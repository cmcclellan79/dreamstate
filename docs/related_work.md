# Related work

Starting reading list. Papers marked ★ are in the project files.

## Rehearsal and generative replay (forgetting)

- ★ **Huang et al., 2024. Mitigating Catastrophic Forgetting in LLMs with Self-Synthesized Rehearsal (SSR).**
  arXiv:2403.01244. The base LLM generates synthetic instances by in-context learning, and these are rehearsed
  during continual fine-tuning because the original training data is unavailable. This is the closest LLM
  precedent for training on self-generated data to protect old knowledge. Our difference: we target *new*
  knowledge consolidation and select what to rehearse by closed-book self-quiz failures.
- ★ **Pomponi, Scardapane & Uncini. Pseudo-Rehearsal for Continual Learning with Normalizing Flows.**
  Combines generative rehearsal with regularization, using a task-conditioned normalizing flow over internal
  embeddings so memory overhead stays constant. Useful framing for "replay vs. regularization."
- Shin et al., 2017. Continual Learning with Deep Generative Replay.
- Hinton et al., 1995. The wake-sleep algorithm for unsupervised neural networks.
- McClelland, McNaughton & O'Reilly, 1995. Complementary learning systems: hippocampal replay and slow
  cortical consolidation. This is the biological motivation for the wake/sleep split.

## Self-generated training data

- Zelikman et al., 2022. STaR: bootstrapping reasoning by training on self-generated correct rationales.
- Wang et al., 2023. Self-Instruct.
- Zweiger et al., 2025. SEAL: Self-Adapting Language Models. The model writes its own "self-edits" (study notes)
  for new documents and fine-tunes on them. This is the nearest neighbour for knowledge incorporation. Compare
  directly.
- Ellis et al. DreamCoder: an explicit dream phase that invents practice problems between waking tasks.

## Risks

- Shumailov et al., 2024 (Nature). AI models collapse when trained on recursively generated data. This motivates
  our source-grounded card verification step.

## Spaced repetition

- Leitner box system; Settles & Meeder, 2016, "A Trainable Spaced Repetition Model for Language Learning."

TODO: search for 2025–2026 work on "knowledge injection," "continued pretraining on synthetic QA," and
"sleep consolidation LLM" before the write-up.
