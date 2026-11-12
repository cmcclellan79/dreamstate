# Versions: Claude scaffold vs. combined

This repo has two versions of the code, so it's always clear which parts came from Claude alone and
which reflect Coby's design input.

| Version | Git ref | What it is |
|---|---|---|
| **Claude scaffold** | branch `claude-scaffold` (commit `2f6c7bc`) | The original scaffold Claude wrote from the proposal. Frozen: never edited. |
| **Combined** | branch `main` | The Claude scaffold plus every change Coby asked for. This is the version that runs. |

To see every difference: `git diff claude-scaffold main`

Both branches sit on top of the GitHub "Initial commit" (LICENSE, plus the README line "DNU research lab
experiment, Fall 2026"). That content is shared baseline, not a difference between the versions.

## Change log (combined vs. Claude scaffold)

| # | Date | Area | Change in combined | Claude scaffold has | Source |
|---|---|---|---|---|---|
| 1 | 2026-10-19 | Analysis | Removed `analysis.py` and the `dreamstate analyze` command. Added `tracking.py`, which logs each (condition, seed) to Weights & Biases, grouped by condition. New config keys `use_wandb`, `wandb_project`. `wandb` is now a required dependency; the `analysis` extra (pandas, matplotlib) was removed. `metrics.jsonl` is still written locally. | `analysis.py`: a markdown results table (mean ± std over seeds) and a matplotlib retention plot | Coby: "analysis can be done by wandb" |

## Discussed, not changed

| Date | Suggestion | Decision | Why |
|---|---|---|---|
| 2026-10-19 | Rewrite `training.py` with TensorFlow | Not changed | Training runs on Tinker's GPUs through `forward_backward` / `optim_step`, and TensorFlow can't drive that. Most of `training.py` builds the token-and-mask data Tinker needs, which any framework would also need. Hugging Face transformers has also dropped TensorFlow support, so Qwen models aren't available in TF. Revisit only if the project moves to local GPUs, and even then PyTorch would be the natural choice. |

## Pending (owned by Coby)

| Item | Note |
|---|---|
| `docs/related_work.md` → reframe as "inspired work" | Coby is rewriting this himself. Claude should not edit it. |
