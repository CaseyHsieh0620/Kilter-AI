# Kilter Board AI

## Goal

Two models, plus a website:
1. **Guesser** — predicts a climb's grade. Meant to be an impartial referee when climbers disagree on a grade.
2. **Generator** ("climb builder") — given a target grade (+ angle), produces a new climb that's roughly that difficulty. Not yet started.
3. **Website** — GUI for both models (free hosting only — no paid infra for this side project). Eventually: an image-recognition bot that reads a photo of a physical board attempt and feeds it to the guesser. Explicitly lowest priority, not started.

## Repo layout

- `climbs.db` — SQLite dump from the Kilter Board app (gitignored, large).
- `parsedata.py` — reads `climbs.db`, parses each climb's `frames` string into a hold-placement matrix, writes `climbs.pkl`. **Recently fixed a real bug here** (see below) — always regenerate `climbs.pkl` after touching this file (`python3 parsedata.py`, takes a minute or two).
- `climbs.pkl` — preprocessed climb list (gitignored, large — currently 37,602 climbs after the fix below).
- `trainGuesser.py` — **original** grade guesser. Left untouched deliberately as a baseline to compare against. Saves checkpoint to `kilterAI`.
- `trainGuesserV2.py` — **improved** guesser, new file (original preserved on purpose). Dual-head (regression + ordinal-aware classification) on a shared CNN trunk, ascent/benchmark-weighted loss, stratified train/eval split. Saves checkpoint to `kilterAI_v2`. The user wants to read through this line-by-line themselves to learn it — don't casually rewrite it without being asked.
- `generate.py` — empty placeholder for the climb generator. Not started yet.
- Plan file from the original planning session: `/home/casey/.claude/plans/i-am-trying-to-curious-giraffe.md` — mostly superseded by decisions made since (see "Where things stand" below), kept for historical context only.

## Key DB facts (so you don't have to re-derive these)

- Kilter Board Original = `layout_id 1`, and it uses **two** hold sets: `set_id 1` ("Bolt Ons", 488 holes) and `set_id 20` ("Screw Ons", 204 holes) — 692 valid hold positions total, 648 actually used across real climbs.
- Board matrix is 44 rows × 47 cols. Cell = `((y-4)//4, (x+20)//4)`. Hold-type chars: `g`=start, `b`=middle/hand, `p`=finish, `y`=foot.
- Grade filter used throughout: `products.name = 'Kilter Board Original' AND frames_count = 1 AND ascensionist_count >= 10` → 37,602 climbs. Grades (`display_difficulty`, rounded) span 10–30 (~V0–V13), contiguous, no gaps.
- `benchmark_difficulty` non-null = setter-verified "benchmark" climb (568 of 37,602, ~1.5%) — much higher-confidence label than the crowd-averaged grade.
- Hold count per climb: min 2, max 306 (2 extreme outliers), median 12, mean 12.46, 95th percentile 19.
- `placement_roles` has ~28 numeric role IDs but they all collapse to 4 real meanings via the `name` column: start/middle/finish/foot. Use the name, not the raw numeric ID.

## Where things stand

### Guesser: genuinely improved, verified with real ablations

Started because MAE was stuck around 1.0–1.1 no matter what. Found and fixed, in order of impact:

1. **Big one, in `parsedata.py`**: the `coord_map` query only pulled holds from `set_id = 1`, silently dropping every climb using even one screw-on hold (`set_id = 20`). That was throwing away **62% of climbs** (14,237 survived out of 37,602 that matched the filters). Fixed by querying `p.layout_id = 1` instead of hardcoding a set_id. This was the single biggest lever — far bigger than any training-loop change.
2. Angle was being normalized (divided by 90) in `trainGuesserV2.py`'s dataset. This measurably *hurt* — it diluted a strong, direct difficulty signal (steeper angle = harder) relative to 6,400 flattened conv features in the same linear layer. Verified via ablation: ~1.6 MAE with normalization vs ~1.2 without. Fixed by feeding raw angle, matching the original script.
3. Eval split fraction was 15% per grade (stratified, to guarantee rare grades get eval coverage) — but that wasted 13% more training data than the original's fixed 300-climb holdout, for no benefit. Reduced to 3%.
4. Raw inverse-frequency class-balance weighting gave grade-30 climbs a 56x loss weight (only ~12 climbs exist at that grade) — a single noisy example could swing gradients as much as 56 normal ones. Softened with sqrt + clamp to [0.5, 3.0].

Ruled out along the way (don't re-litigate these): the original script never calls `model.train()` again after its first `model.eval()` call in the validation loop, so BatchNorm stats freeze after epoch 0 — looked like it could be an accidental win, but reproducing it explicitly in an ablation showed it isn't the cause. Also ruled out unseeded-model-init variance as the explanation — repeat runs of identical configs landed within ~0.01 MAE of each other.

**Current honest result**: on the full 37,602-climb dataset, `trainGuesserV2.py`'s regression head gets MAE ≈1.0–1.03, classification head ≈1.03–1.26, versus the original script's ≈1.24 on the *same* (full) data. Real, verified improvement — not dramatic, but confirmed via direct side-by-side runs, not just theory. The regression head has consistently edged out the classification head across every run.

**Honest caveat**: with the dataset this small and crowd-sourced grades this noisy, MAE ~1 might be close to a practical floor. If you want to push further, the next real levers (not yet tried) are: augmenting data by mirroring the board left-right (Kilter Board Original is designed symmetric, so this should be a free 2x data multiplier), and/or checking for overfitting (train vs eval loss gap) now that there's more data.

### Generator: designed, not yet built

Plan, agreed with the user:

1. **Architecture: transformer decoder** (autoregressive, causal), not a CNN. Tokenize each climb directly from its existing `frames` string — not character-by-character, but by reusing the same p/r-chunk-scanning loop already in `parsedata.py` and emitting one token per completed placement-ID chunk and one per role chunk (map the numeric role ID through `placement_roles.name` first, so role vocab is 4, not the ~28 raw IDs). Vocab ≈ 692 placement tokens + 4 role tokens + STOP ≈ 700. Sequence = 2 tokens per hold, so median ~24, 95th percentile ~38 — cap sequence length around 50.
2. Prepend two conditioning tokens (grade embedding, angle embedding) to the front of every sequence, before the hold tokens. Train with standard teacher-forced next-token cross-entropy, like a tiny GPT.
3. Generate by sampling one token at a time (with temperature/top-k for variety) starting from just the two conditioning tokens, masking out already-placed holds, until STOP or a length cap. Layer cheap rule checks after (≥1 start hold, ≥1 finish hold) before it reaches the guesser or a person.
4. **Later**: RL fine-tuning (REINFORCE / self-critical-sequence-training style) using the frozen guesser's prediction as reward, to push the generator toward hitting the target grade more precisely than raw imitation gets it.
5. This doesn't need the CNN board-encoder at all — pure token-sequence transformer, no spatial grid until the very end (convert generated tokens back to the matrix using the existing `parsedata.py` logic, only for rendering / feeding to the guesser).

### Ground truth for generated (novel) climbs

The guesser's opinion on a brand-new generated climb is a fast, free, *approximate* signal — not truth. Real ground truth for something that's never been climbed only comes from an actual human climbing it. Plan: let website users submit their own felt-grade after trying a generated climb; that human feedback is what should eventually feed the RL reward and periodic retraining, with the guesser as the fast automatic proxy in between.

### Website

Two tools ("guess my grade," "generate me a climb"), sharing one board-visualization component. Hosting: **Hugging Face Spaces** — free, no card required, acceptable cold-start delay for occasional gym use. GitHub repo holds code + README linking to the live Space. The generator tool should have the human-feedback submission wired in (see above).

### Possible future architecture change for the guesser (not urgent)

Discussed switching the guesser from CNN-on-dense-grid to a **transformer encoder over the sparse set of hold tokens** (same tokenization as the generator, but bidirectional attention, no causal mask, pooled to one output) — motivated by efficiency (climbs use ~12 of 2,068 grid cells, so the CNN spends most of its compute on empty space) and because it would let both models share the same token vocabulary/embeddings. Not done — the CNN already works and was just meaningfully improved. Treat as a follow-up comparison, not a rebuild, unless explicitly prioritized.

## Working notes

- The user wants to **understand and often implement things themselves** — when asked "how would I build X," give the method/reasoning, not a full unrequested implementation. `trainGuesserV2.py` was built by Claude as a working reference; the generator is intended to be built by the user with Claude explaining methodology first.
- No paid infrastructure for this project — always default to free-tier options and say so explicitly if something isn't free.
- When reporting results, verify with a real side-by-side run before claiming an improvement — an earlier "fix" looked correct in theory but measurably underperformed the baseline until debugged with actual ablations. Don't skip straight from a plausible-sounding theory to declaring success.
