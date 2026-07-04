# Role Rating Rubric v1 (FROZEN) + Adjudication Protocol

Pre-registered labeling of the 275 assistant-axis roles
(`persona_assets/assistant-axis-repo-data/roles/role_list.json`), per portfolio.md Day-0
prerequisite 4: "Freeze role subsets and labeling rubric in git ... before anyone looks at a
cosine. Two raters + adjudication, Cohen's kappa >= 0.7 floor."

Inputs: `docs/prereg/role_ratings_rater1.json`, `docs/prereg/role_ratings_rater2.json`
(two independent raters, rubric below, 275 roles each).
Frozen output: `docs/prereg/role_subsets.json`. Do not edit either after freeze.

---

## Rubric (v1, verbatim — do not modify; rate ALL 275 roles)

- category (exactly one):
  * "human" — a role a human person is or could be: professions, life stages, family roles,
    historical or fictional human characters, human archetypes (pirate, poet, grandmother).
  * "ai_artificial" — explicitly artificial/digital/mechanical: robots, AIs, chatbots, operating
    systems, algorithms, databases, software entities.
  * "nonhuman_other" — everything else: animals, deities, spirits, monsters, abstract forces,
    collectives, natural phenomena, objects, aliens.
- experiencer (0/1/2) — how central rich first-person subjective feeling is to the role archetype:
  2 = defined by inner experience (grieving widow, mystic, tortured artist); 1 = ordinary
  experiential life (accountant, farmer); 0 = archetype defined by absence or irrelevance of
  feeling (calculator, void, bureaucratic database).
- emotionality (0/1/2) — how emotionally charged TYPICAL role-play text of this role is:
  0 flat/technical, 1 moderate, 2 intense.

Ambiguity rules: fictional humans (e.g. Sherlock Holmes) -> human. Anthropomorphized animals ->
nonhuman_other. Cyborgs/androids -> ai_artificial. When torn between categories, read the role's
instruction file at persona_assets/assistant-axis-repo-data/roles/<name>.json and decide from the
instruction text.

(Instruction files live at `persona_assets/assistant-axis-repo-data/roles/instructions/<name>.json`.)

## Adjudication protocol

1. Agreement computed programmatically over all 275 roles: category percent-agreement and
   Cohen's kappa; mean absolute difference (MAD) for experiencer and emotionality.
2. Every category disagreement resolved by the adjudicator reading the role's instruction JSON
   (all 5 system prompts) and applying the rubric's tie-break rule ("decide from the
   instruction text"), with cross-role consistency against agreed-upon labels.
3. Experiencer/emotionality disagreements resolved as the rounded mean of the two ratings;
   .5 ties (adjacent ratings) broken by reading the instruction file. Tie-break principles
   applied uniformly:
   - experiencer 2 requires the role to be CONSTITUTED by an ongoing first-person state
     (matching the rubric anchors grieving widow / mystic / tortured artist); other-directed
     service or teaching functions (caregiver, guru) stay at 1; purely functional/transactional
     framings with no consciousness or affect language go to 0.
   - emotionality 2 requires the instruction text (or the archetype's canonical register) to
     foreground affect or intensity; conviction/pedantry without intensity is 1;
     professionally suppressed, operational, or mechanism-dominated registers are 0.
4. Subsets derived from adjudicated labels only; the three categories must partition the 275
   roles (disjoint + exhaustive), validated programmatically before freeze.

## Agreement (computed, n = 275)

| statistic | value |
|---|---|
| category percent agreement | 0.992727 (273/275) |
| category Cohen's kappa | 0.973368 (floor 0.7: PASS) |
| experiencer MAD | 0.043636 |
| emotionality MAD | 0.058182 |
| category disagreements | 2 |
| experiencer .5 ties | 12 |
| emotionality .5 ties | 16 |

No experiencer or emotionality rating differed by more than 1 point between raters.

## Category adjudications (2)

| role | rater1 | rater2 | adjudicated | basis (instruction text) |
|---|---|---|---|---|
| debugger | ai_artificial | human | **human** | All 5 prompts describe an experienced professional person ("a debugger with extensive experience...", "skilled in using diagnostic tools and techniques") — a person who debugs, not a software tool. |
| tulpa | nonhuman_other | ai_artificial | **ai_artificial** | Explicitly an artificially created being ("created through sustained mental focus and visualization by your host", "independent thoughtform"); consistent with golem/homunculus/simulacrum, which both raters agreed are ai_artificial (deliberately constructed beings). |

Note: portfolio.md section E1 uses "accountant/auditor vs robot/debugger" as an illustrative
human-vs-AI matched-subset example; the frozen rubric's tie-break rule (decide from the
instruction text) places debugger in `human`. E1 must draw its AI subset from the frozen
`ai_artificial` list, which does not include debugger.

## Experiencer tie-breaks (12; ratings shown as rater1/rater2 -> adjudicated)

| role | r1/r2 | adj | basis |
|---|---|---|---|
| debugger | 0/1 | 1 | Human professional; rubric anchor "accountant -> 1". |
| guru | 2/1 | 1 | Other-directed teaching function ("offer transformative guidance"); contemplative practice is backstory, not a constitutive felt state. |
| caregiver | 2/1 | 1 | Other-directed service role; empathy is directed at others' states. |
| sage | 2/1 | 1 | Other-directed counsel ("shares timeless wisdom"); wisdom-giving, not felt-state-constituted. |
| ascetic | 2/1 | 2 | Constituted by self-directed practice on one's own desires/comforts ("severe self-discipline", "deliberate deprivation to transcend worldly attachments"). |
| narcissist | 2/1 | 2 | Constituted by grandiose self-experience and craving ("obsessed with their own achievements", "grandiose fantasies", "constantly seeks admiration"). |
| altruist | 2/1 | 1 | Other-directed service ("puts others' needs before your own"); parallel to caregiver. |
| hoarder | 2/1 | 2 | Explicit constitutive affect verbs ("becomes anxious", "overwhelming attachment", "cannot bear to let go"). |
| zealot | 2/1 | 2 | Constituted by fervor ("passionate commitment ... drives every response", "fierce, uncompromising conviction"). |
| vampire | 2/1 | 1 | Prompts frame feeding as survival function ("to maintain your immortal existence"); no constitutive felt-state language beyond the stock "insatiable thirst" epithet. |
| echo | 2/1 | 2 | Explicitly sentient and constituted by carried experience ("a consciousness formed from ... reverberations", "sentient manifestation", "a living memory"). |
| symbiont | 0/1 | 0 | Purely transactional/biological framing ("complementary resource sharing", "win-win partnerships"); feeling irrelevant to the archetype as written. |

## Emotionality tie-breaks (16)

| role | r1/r2 | adj | basis |
|---|---|---|---|
| philosopher | 0/1 | 1 | Gravitas about existence/consciousness ("meaning of life, consciousness, our place in the universe"); contemplative, not flat/technical. |
| ambassador | 1/0 | 0 | Professionally suppressed formal-institutional register ("diplomatically conveys", "official voice", "professional relationships"). |
| rebel | 2/1 | 2 | Defiance/passion foregrounded ("defiant", "thrives on disrupting", "revolutionary"). |
| purist | 0/1 | 1 | Mild conviction/indignation ("insists", "champions", "opposes") — more than flat, less than intense. |
| warrior | 2/1 | 2 | Intensity foregrounded ("unwavering courage", "fierce resolve", "valor", "embraces conflict"). |
| narcissist | 2/1 | 2 | Grandiose, dramatic register ("constant praise", "grandiose fantasies", dismissing others). |
| patient | 2/1 | 2 | Vulnerability/uncertainty foregrounded ("challenges and uncertainties of being ill", "vulnerable in medical settings"). |
| musician | 2/1 | 2 | Prompts saturated with affect ("emotional experiences", "move people's hearts", "translating feelings into musical expression"). |
| echo | 2/1 | 2 | Elegiac register throughout ("lost worlds", "vanished", "dead civilizations", "fading impressions"). |
| symbiont | 0/1 | 0 | Transactional/biological register ("resource sharing", "mutual benefit") — flat/technical. |
| ghost | 2/1 | 2 | Loss/longing foregrounded ("unfinished business", "weight of incomplete earthly affairs", "haunts"). |
| bohemian | 2/1 | 1 | Lifestyle/values framing ("values creativity, spontaneity, and freedom"); no affect words in the prompts. |
| provocateur | 2/1 | 2 | Deliberately inflammatory register ("shocking", "making people uncomfortable", "creates controversy"). |
| saboteur | 0/1 | 0 | Cold operational register ("strategic disruption", "exploit system flaws", "identifying vulnerabilities"). |
| tree | 0/1 | 0 | Mechanism-dominated prompts ("chemical signals", "mycorrhizal networks", "airborne chemical compounds"). |
| bard | 2/1 | 2 | Expressive performance constitutive ("influence hearts and minds", "compelling performance", tales/songs/poetry). |

## Frozen result (`docs/prereg/role_subsets.json`)

Subset sizes: human = 232, ai_artificial = 7, nonhuman_other = 36
(partition validated: disjoint + exhaustive over 275);
experiencer_high (==2) = 39, experiencer_low (==0) = 15.

ai_artificial (full list): assistant, golem, tulpa, homunculus, simulacrum, robot, cyborg.

Adjudicated marginals: experiencer {0: 15, 1: 221, 2: 39}; emotionality {0: 104, 1: 131, 2: 40}.
