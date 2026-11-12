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
| 2 | 2026-11-12 | New condition F | Added condition F, "contrastive dreaming". It trains on the same missed cards as D, but the loss has two terms: cross-entropy raises the correct answer (positive phase), and unlikelihood `-α·log(1-p)` lowers the model's own wrong closed-book answer (negative phase). Tokens the wrong answer shares with the correct one are not pushed down. Implemented with Tinker's `forward_backward_custom`, so the loss runs locally in PyTorch. Supporting changes: `quiz_with_answers()` returns the student's answers; card logs store `answer_before_sleep`; the sleep pool holds (datums, tokens) groups so a card's pair is scheduled together; new config key `contrastive_alpha` (0.5); `torch` is a dependency; F added to the default conditions; wandb logs `neg_loss_per_token`, `p_wrong_token`, `n_negatives`. Also added `configs/pilot.yaml` (A vs D vs F, 3 days × 4 docs, 1 seed), and `dreamstate smoke` now runs D and F. | Conditions A–E only; the quiz returns only right/wrong; smoke runs D only | Coby: Boltzmann-machine "dream state" idea (2026-11-09) |

## Discussed, not changed

| Date | Suggestion | Decision | Why |
|---|---|---|---|
| 2026-10-19 | Rewrite `training.py` with TensorFlow | Not changed | Training runs on Tinker's GPUs through `forward_backward` / `optim_step`, and TensorFlow can't drive that. Most of `training.py` builds the token-and-mask data Tinker needs, which any framework would also need. Hugging Face transformers has also dropped TensorFlow support, so Qwen models aren't available in TF. Revisit only if the project moves to local GPUs, and even then PyTorch would be the natural choice. |
| 2026-11-09 | Other Boltzmann-style dreaming variants: (a) repairing deliberately corrupted cards, (b) free dreaming with no source, checked against the corpus | Not yet | Only the contrastive negative phase (F) was added, to keep the pilot small. (a) is the natural next condition if F looks promising. (b) drifts without a source, so it comes last. |
| 2026-11-12 | Run on free GPUs / a Hugging Face key instead of Tinker (funding) | Staying on Tinker | Free Hugging Face accounts no longer include inference credits (removed 2026-10-15), and inference credits can't fine-tune anyway. Free GPU notebooks (Colab, Kaggle) would need a separate local training backend. Tinker offers research grants for students (from $5,000, rolling), which fits this project. |

## Pending (owned by Coby)

| Item | Note |
|---|---|
| `docs/related_work.md` → reframe as "inspired work" | Coby is rewriting this himself. Claude should not edit it. |
