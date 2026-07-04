export PYTHONUNBUFFERED=1
# Weights live in the shared /data/hf_cache (10GB home quota kills default-cache
# downloads). HF_HOME goes to a user-writable dir because the shared cache's
# token file is not readable by us (the hub crashes reading it if HF_HOME
# points at the shared cache).
export HF_HOME=/data/agastyas/hf_home
export HF_HUB_CACHE=/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache
# Cluster egress to huggingface.co is flaky (SYN-SENT blackholes that hang the
# FP8 loader's live kernel-version lookup for good). Everything we need — Qwen
# weights AND the finegrained-fp8 Triton kernel — is already in /data/hf_cache,
# so run fully offline and pin the kernel to its cached snapshot (the pin is
# applied by emotion_probes/models/language_model.py; without it, offline mode
# crashes on the kernel's list_repo_refs API call).
export HF_HUB_OFFLINE=1
export EP_FP8_KERNEL_REVISION=061130fedf845f320c56de4425f7404f6512c87e
# SANDBOX: fork runs must never write into the main repo. suite_root() defaults
# to the MAIN repo's /data/agastyas/emotion-probes/suite when this is unset (a
# hand-launched 235B smoke once leaked its output there). The fork's suite dirs
# symlink vectors/ to the main repo (read-only) and own their analysis/ dirs.
export EMOTION_PROBES_SUITE_ROOT=/data/agastyas/emotion-probes-persona/suite
