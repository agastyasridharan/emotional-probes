"""
Pre-flight check for the 14-model suite -- run this BEFORE committing GPU-days.

It answers, per model, the only questions that abort a long run hours in:

  * Does the HF repo id resolve, and (if gated) do we actually have access?
  * Do the live config's layer-count / hidden-size match the manifest?
  * Is there a FAST tokenizer (the per-token analyses require one)?
  * Is the architecture one whose residual stream ProbedModel can hook?
  * For the FP8 235B: is the FP8 backend importable?
  * Will the weights fit -- one card, or must it shard?

Two modes:

    python scripts/preflight_suite.py                 # --dry on all 14 (metadata only; CPU + proxy, ~seconds each)
    python scripts/preflight_suite.py --key qwen3-32b # one model
    python scripts/preflight_suite.py --load          # ALSO load each on GPU + tiny forward (slow; needs cards)

`--dry` downloads only config.json + the file list (no weights), so it is cheap
and safe to run on a login node with the proxy exported. `--load` actually builds
ProbedModel, confirms the decoder layers resolve, runs a 2-text forward, and
reports real per-card residency -- the definitive "it fits and hooks work" test.

Exit code is non-zero if any model FAILs, so it can gate a launcher script.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emotion_probes.models.suite import CARD_GB, NUM_CARDS, SUITE, SuiteModel, by_key  # noqa: E402

# Architecture classes / model_types we know ProbedModel can hook (clean per-layer
# residual stream). Kept loose -- the definitive check is --load.
_KNOWN_ARCH_SUBSTR = ("ForCausalLM", "ForConditionalGeneration")


def _hf_token() -> str | None:
    """Read the token the cluster setup points at (HF_TOKEN_PATH), or the env."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    p = os.environ.get("HF_TOKEN_PATH")
    if p and Path(p).expanduser().is_file():
        return Path(p).expanduser().read_text().strip() or None
    return None


def _config_dims(cfg: dict) -> tuple[int | None, int | None, dict]:
    """Pull (num_layers, hidden_size, text_sub_config) handling VLM nesting."""
    sub = cfg.get("text_config") or cfg.get("llm_config") or {}
    layers = cfg.get("num_hidden_layers", sub.get("num_hidden_layers"))
    hidden = cfg.get("hidden_size", sub.get("hidden_size"))
    return layers, hidden, sub


def dry_check(m: SuiteModel) -> tuple[str, list[str]]:
    """Metadata-only check. Returns (status, messages) with status PASS/WARN/FAIL."""
    import json

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import (
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
    )

    msgs: list[str] = []
    status = "PASS"
    token = _hf_token()

    # 1) repo resolves + access (a gated repo without access raises here)
    try:
        info = HfApi().model_info(m.model_id, token=token, files_metadata=False)
    except GatedRepoError:
        return "FAIL", [f"GATED and not accessible with the current token "
                        f"({'token present' if token else 'NO token'}); accept terms + login"]
    except RepositoryNotFoundError:
        return "FAIL", ["repo id does not resolve (typo, renamed, or private without access)"]
    except Exception as e:  # network/proxy/etc
        return "FAIL", [f"could not reach HF: {type(e).__name__}: {e}"]

    siblings = {s.rfilename for s in (info.siblings or [])}

    # 2) live config dims vs manifest
    try:
        cfg_path = hf_hub_download(m.model_id, "config.json", token=token)
        cfg = json.loads(Path(cfg_path).read_text())
    except EntryNotFoundError:
        return "FAIL", ["no config.json in repo"]
    layers, hidden, _ = _config_dims(cfg)
    archs = cfg.get("architectures") or []
    if layers != m.num_layers:
        status = "WARN"; msgs.append(f"layers: manifest {m.num_layers} != config {layers}")
    if hidden != m.hidden_size:
        status = "WARN"; msgs.append(f"hidden: manifest {m.hidden_size} != config {hidden}")
    if archs and not any(sub in a for a in archs for sub in _KNOWN_ARCH_SUBSTR):
        status = "WARN"; msgs.append(f"unrecognised architecture {archs} (verify with --load)")
    if archs and archs[0] != m.arch:
        msgs.append(f"note: arch is {archs[0]} (manifest says {m.arch})")

    # 3) fast tokenizer present
    if "tokenizer.json" not in siblings:
        status = "WARN"; msgs.append("no tokenizer.json -> per-token analyses (deflection) may fail")

    # 4) FP8 expectations -- the required backend depends on the *scheme*:
    #   quant_method "fp8"               -> transformers-native FineGrainedFP8 (torch float8 + triton)
    #   quant_method "compressed-tensors"-> the compressed_tensors package
    #   quant_method "fbgemm_fp8"        -> fbgemm_gpu
    qc = cfg.get("quantization_config")
    if m.fp8:
        if not qc:
            status = "WARN"; msgs.append("manifest says fp8 but config has no quantization_config")
        else:
            method = (qc.get("quant_method") or "").lower()
            if method == "fp8":
                import torch
                if not hasattr(torch, "float8_e4m3fn"):
                    status = "WARN"; msgs.append("native FP8 needs a torch build with float8 dtypes")
                else:
                    try:
                        import triton  # noqa: F401
                    except Exception:
                        msgs.append("note: triton missing -> FP8 uses the slower torch fallback")
            elif method == "compressed-tensors":
                try:
                    import compressed_tensors  # noqa: F401
                except Exception:
                    status = "WARN"; msgs.append("compressed-tensors FP8 needs the compressed_tensors package")
            elif method in ("fbgemm_fp8", "fbgemm"):
                try:
                    import fbgemm_gpu  # noqa: F401
                except Exception:
                    status = "WARN"; msgs.append("fbgemm FP8 needs the fbgemm_gpu package")
            else:
                msgs.append(f"note: unrecognised FP8 quant_method {method!r}; verify with --load")
    elif qc:
        msgs.append(f"note: config carries quantization_config ({qc.get('quant_method')})")

    # 5) capacity
    wt = m.weight_gb()
    if m.device_map == "cuda" and not m.fits_one_card():
        status = "WARN"; msgs.append(f"~{wt:.0f}GB won't fit one {CARD_GB}GB card but device_map=cuda; use auto")
    if wt > NUM_CARDS * CARD_GB:
        status = "FAIL"; msgs.append(f"~{wt:.0f}GB exceeds the whole box ({NUM_CARDS}x{CARD_GB}GB)")

    if not msgs:
        msgs.append(f"L={layers} H={hidden} arch={archs[0] if archs else '?'} ~{wt:.0f}GB, fits "
                    f"{'1 card' if m.fits_one_card() else f'{ -(-int(wt)//CARD_GB) } cards (auto)'}")
    return status, msgs


def load_check(m: SuiteModel) -> tuple[str, list[str]]:
    """Actually load the model and run a tiny forward; the definitive test."""
    from emotion_probes.config import Config
    from emotion_probes.models import ProbedModel

    msgs: list[str] = []
    cfg = Config().with_(
        model_id=m.model_id, device_map=m.device_map,
        attn_implementation=m.attn_implementation,
        enable_thinking=m.enable_thinking,
    )
    try:
        pm = ProbedModel(cfg)
    except Exception as e:
        return "FAIL", [f"load failed: {type(e).__name__}: {e}"]

    if pm.num_layers != m.num_layers:
        msgs.append(f"layers resolved {pm.num_layers} != manifest {m.num_layers}")
    if pm.hidden_size != m.hidden_size:
        msgs.append(f"hidden resolved {pm.hidden_size} != manifest {m.hidden_size}")
    if not pm.tokenizer_is_fast:
        msgs.append("tokenizer is NOT fast -> deflection/per-token paths will error")

    # tiny forward through the real capture path
    try:
        means = pm.extract_means(["I feel afraid.", "What a wonderful day."],
                                 layers=[pm.layer_index_for_fraction()])
        msgs.append(f"forward OK: means {means.shape} @ analysis layer {pm.layer_index_for_fraction()}")
    except Exception as e:
        return "FAIL", [f"forward/capture failed: {type(e).__name__}: {e}"]

    # per-card residency
    try:
        import torch
        if torch.cuda.is_available():
            res = [f"{torch.cuda.memory_allocated(i)/1e9:.1f}" for i in range(torch.cuda.device_count())]
            msgs.append(f"resident GB/card: [{', '.join(res)}]")
    except Exception:
        pass
    return ("WARN" if any("!=" in x or "NOT" in x for x in msgs) else "PASS"), msgs


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-flight check for the model suite.")
    ap.add_argument("--key", help="check just this model (default: all 14)")
    ap.add_argument("--load", action="store_true", help="also load on GPU + tiny forward (slow)")
    args = ap.parse_args()

    models = [by_key(args.key)] if args.key else list(SUITE)
    print(f"Pre-flight: {len(models)} model(s), mode={'load' if args.load else 'dry'}, "
          f"token={'yes' if _hf_token() else 'no'}\n")

    worst_fail = False
    for m in models:
        status, msgs = (load_check if args.load else dry_check)(m)
        worst_fail = worst_fail or status == "FAIL"
        mark = {"PASS": "ok  ", "WARN": "WARN", "FAIL": "FAIL"}[status]
        print(f"[{mark}] {m.key:<18} {m.model_id}")
        for x in msgs:
            print(f"         - {x}")
    print("\n" + ("FAILURES present -- fix before launching." if worst_fail else "no hard failures."))
    sys.exit(1 if worst_fail else 0)


if __name__ == "__main__":
    main()
