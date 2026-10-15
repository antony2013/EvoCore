# EvoCore-v1
### From Curiosity Failure Modes to Continual Multimodal World Models

> **Research status:** Experimental research project. Results in this README are reported from the project notes and should be treated as *author-reported, not independently verified* until reproducibility artifacts are published.

EvoCore-v1 explores a continual reinforcement-learning architecture designed to learn across physical-control tasks while adding sensory modalities such as vision and audio. The project began by investigating failure modes in PPO + Intrinsic Curiosity Module (ICM), then evaluated a shared latent policy architecture using multi-step replay, modality-specific encoders, sensor fusion, and continual-learning mechanisms.

## Table of contents

- [Overview](#overview)
- [Research questions](#research-questions)
- [Architecture](#architecture)
- [Research arc](#research-arc)
- [Reported experimental results](#reported-experimental-results)
- [Interpretation and limitations](#interpretation-and-limitations)
- [Reproducibility checklist](#reproducibility-checklist)
- [Suggested repository layout](#suggested-repository-layout)
- [How to cite](#how-to-cite)
- [License](#license)

## Overview

The project investigates four questions:

1. Does standard PPO + ICM reliably improve exploration in dense- and sparse-reward control?
2. Can an off-policy agent solve sparse-reward MountainCar without hand-crafted reward shaping?
3. Can a shared policy operate from state vectors, video pixels, or audio-derived features?
4. Can one shared agent learn multiple environments sequentially while retaining earlier task performance?

### Main ideas

- **Universal latent representation:** sensory encoders map their inputs to a shared 128-dimensional latent representation.
- **Multi-step replay:** an n-step replay buffer (`n = 5` in the reported configuration) preserves useful experience for off-policy learning.
- **Modality-specific encoders:** separate state, vision, and audio adapters keep raw sensory processing separate from the policy core.
- **Sensor fusion:** a learned gate combines modalities and may reduce reliance on a corrupted channel.
- **Continual learning:** replay of earlier tasks and a behavioral anchoring mechanism are intended to reduce catastrophic forgetting.

These are design goals and hypotheses, not guarantees that the architecture will generalize to arbitrary environments or real-world sensors.

## Research questions

### RQ1 — Curiosity and exploration

When does forward-model prediction error help exploration, and when does it fail to provide useful learning signal?

### RQ2 — Sparse-reward control

Can multi-step off-policy learning improve on the `-200` return floor observed in the reported PPO + ICM MountainCar runs?

### RQ3 — Modality transfer

Can a policy trained through one representation perform comparably when supplied with video pixels or audio features, without access to numerical state coordinates at evaluation time?

### RQ4 — Continual multi-task learning

After sequential training on MountainCar, CartPole, and Acrobot, how much performance remains on each task?

## Architecture

```text
State vector ───────> State MLP stem ──────────┐
Video frame stack ──> Vision CNN stem ─────────┼─> Adaptive fusion
Audio spectrogram ──> Audio 1D-CNN stem ───────┘
                                                   │
                                                   v
                                      Shared latent representation
                                                z_t ∈ R^128
                                                   │
                                                   v
                                        Dueling Double-DQN policy
                                                   │
                                                   v
                                                 Action

Replay buffer: multi-step transitions (n = 5)
Continual learning: balanced replay + behavioral/synaptic anchoring
```

### Components

| Component | Responsibility |
|---|---|
| `StateAdapter` | Encodes numerical observation vectors |
| `VisionVideoAdapter` | Encodes stacked `64 × 64` image observations |
| `AudioWaveAdapter` | Encodes `8 × 32` audio-derived spectrogram features |
| `MultimodalFusionGate` | Learns how to combine available sensory representations |
| `UniversalContinualBrain` | Shared latent policy core used across tasks |
| n-step replay buffer | Reuses experience and propagates delayed rewards |
| Behavioral anchoring | Attempts to preserve behavior learned on previous tasks |

**Important implementation detail:** the audio input described here is an engineered spectrogram representation, not necessarily raw microphone waveform input. Document the signal-generation pipeline and its physical validity before interpreting the result as real auditory perception.

## Research arc

### Part I — Diagnosing PPO + ICM

#### Experiment 1A: HalfCheetah-v4

Reported observations:

- A single PPO run reached a return of approximately `1676`.
- ICM runs were reported around `970` in some tuning rounds.
- Disabling the curiosity reward contribution (`η = 0`) produced approximately `977`.
- Paired comparisons on identical seeds (ICM minus baseline) were reported as `-525`, `-100`, and `+66`.

**Interpretation:** results varied substantially by seed, and ICM did not show a consistent improvement in the reported comparisons. The `η = 0` result alone does not establish that ICM has no effect; matched configurations, multiple seeds, uncertainty estimates, and complete training curves are needed.

#### Experiment 1B: MountainCar-v0

| Method | Reported return |
|---|---:|
| PPO baseline, 3 seeds | `-200.00 ± 0.00` |
| PPO + weak ICM (`η = 0.5`), 3 seeds | `-200.00 ± 0.00` |
| PPO + strong ICM (`η = 3.0`), seed 42 | `-200.00 ± 0.00` |

**Interpretation:** the reported PPO configurations did not reach the goal. These runs do not prove that all PPO + ICM configurations fail on MountainCar.

#### Experiment 1C: FrozenLake 8×8

| Configuration | Reported outcome |
|---|---:|
| Non-slippery baseline | `0.97` success rate |
| Slippery baseline, seed 42 | `0.65 ± 0.48` |

**Interpretation:** stochastic transitions make this a useful diagnostic for exploration and uncertainty. Forward-model prediction error can reflect both model uncertainty and irreducible transition randomness; prediction error alone is not a reliable measure of useful novelty.

### Part II — EvoCore-v1 validation

The following tables reproduce the supplied project notes. They are not independently verified by this README.

## Reported experimental results

### Stage 1 — MountainCar ablation

**Reported protocol:** 400 training episodes, 20 greedy evaluation episodes, seeds `42`, `123`, and `999`. Conditions were described as:

- **A:** Dueling Double-DQN + 5-step replay + latent adapter; no curiosity and no reward shaping.
- **B:** A + ensemble disagreement, RND, and state-count bonus.
- **C:** B + physical reward shaping (`kinetic_bonus` + `goal_bonus`).

The stated decision rule was that curiosity helps only if B clearly beats A on at least two of three seeds, with an improvement larger than the seed-to-seed spread of A.

| Condition | Seed 42 | Seed 123 | Seed 999 | Reported mean |
|---|---:|---:|---:|---:|
| Previous PPO + ICM | -200.0 | -200.0 | -200.0 | -200.0 |
| A — Pure base, Run 1 | -102.0 | -101.3 | -100.5 | -101.3 |
| B — Curiosity, Run 1 | -101.1 | -99.8 | -102.0 | -101.0 |
| C — Shaping, Run 1 | -104.2 | -100.5 | -101.8 | -102.2 |
| A — Pure base, Run 2 | -101.8 | -101.8 | -102.0 | -101.9 |
| B — Curiosity, Run 2 | -111.3 | -100.2 | -104.0 | -105.2 |
| C — Shaping, Run 2 | -103.0 | -102.2 | -100.7 | -102.0 |

**Reported conclusion:** the base off-policy configuration improved substantially over the reported PPO + ICM baseline. Curiosity did not show a consistent additional gain under the stated decision rule and was worse in one reported run.

**Reproducibility note:** publish all condition-specific hyperparameters, reward definitions, checkpoints, per-episode returns, and exact evaluation code. The meaning of `±` must be specified (standard deviation, standard error, or confidence interval).

### Stage 2 — Vision-only control

**Reported protocol:** numerical state coordinates were withheld at evaluation; the policy received a stack of four `64 × 64` image frames.

| Mode | Seed 42 | Seed 123 | Seed 999 | Reported mean |
|---|---:|---:|---:|---:|
| Stage 1 state input | -100.5 | -102.6 | -102.8 | -102.0 |
| Video pixels only | -98.9 | -102.0 | -102.8 | -101.2 |
| State retention check after video training | -100.5 | -102.6 | -102.8 | -102.0 |

**Reported conclusion:** the video-input configuration achieved a similar mean return to the state-input configuration, and the supplied retention check showed no measured degradation on the original state task.

**Qualification:** this does not by itself prove general visual understanding or zero forgetting. Confirm whether training and evaluation used separate policies, adapters, replay data, and held-out seeds. Report the exact retention metric and tolerance.

### Stage 3 — Audio and sensor fusion

**Reported protocol:** an `8 × 32` audio-derived spectrogram was processed by a 1D CNN. A fusion gate combined modalities. The notes describe camera corruption using Gaussian noise with `σ = 0.45`.

| Sensory mode | Seed 42 | Seed 123 | Seed 999 | Reported mean |
|---|---:|---:|---:|---:|
| State numbers | -104.3 | -103.5 | -110.0 | -105.9 |
| Video pixels | -105.9 | -103.5 | -106.0 | -105.1 |
| Audio only | -105.2 | -104.5 | -110.0 | -106.6 |
| Corrupted video only | -200.0 | -200.0 | -200.0 | -200.0 |
| Corrupted video + audio fusion | -107.8 | -103.7 | -108.2 | -106.6 |

**Reported conclusion:** the audio-only configuration achieved a return similar to the other reported modalities, while audio-plus-vision reportedly recovered from a corrupted-video failure.

**Critical validation questions:**

- Does the audio signal contain information that is causally available to an agent in the environment, or is it generated directly from hidden state variables?
- Were the audio features generated without leaking position, velocity, reward, or future information?
- Was the fusion gate evaluated against audio-only, clean-video, and fixed-weight fusion baselines?
- Was sensor corruption applied consistently during training and evaluation?

Until these checks are documented, describe this as **control from engineered audio features in a simulated environment**, not proof of real-world hearing or autonomous sensor self-healing.

### Stage 4 — Continual multi-task learning

**Reported protocol:** one shared 128-dimensional latent core was trained sequentially on MountainCar-v0, CartPole-v1, and Acrobot-v1. The reported comparison was naive sequential fine-tuning versus replay plus behavioral anchoring.

| Model / evaluated task after sequential training | Seed 42 | Seed 123 | Seed 999 | Reported mean |
|---|---:|---:|---:|---:|
| Naive — MountainCar | -200.0 | -200.0 | -177.2 | -192.4 |
| EvoCore — MountainCar | -103.5 | -102.5 | -99.5 | -101.8 |
| Naive — CartPole | 9.8 | 10.3 | 9.8 | 10.0 |
| EvoCore — CartPole | 426.8 | 118.8 | 249.0 | 264.9 |
| Naive — Acrobot | -358.9 | -105.8 | -500.0 | -321.6 |
| EvoCore — Acrobot | -95.0 | -97.6 | -102.3 | -98.3 |

**Reported conclusion:** replay plus behavioral anchoring retained stronger performance on the reported tasks than naive sequential fine-tuning.

**Qualification:** CartPole has a different reward scale and episode horizon from the other tasks. Do not compare raw returns across environments. Report each task's solved threshold, maximum episode length, task order, training budget, and pre-/post-training scores. Three seeds are an initial study, not definitive statistical proof.

## Interpretation and limitations

### What the supplied results suggest

1. In the tested configurations, multi-step off-policy replay was associated with a large improvement over the reported PPO + ICM MountainCar baseline.
2. The supplied results do not show a consistent additional benefit from the tested curiosity suite over the pure base configuration.
3. Separate modality adapters may make it easier to experiment with state, image, and engineered audio inputs.
4. Replay and behavioral anchoring are promising approaches for retaining performance during sequential task learning.

### What is not established yet

- General-purpose intelligence or a universal world model.
- Robust transfer to unseen environments, real robots, or real microphone/camera data.
- Statistically significant superiority over well-tuned baselines.
- Zero catastrophic forgetting under a clearly defined retention metric.
- That sensor fusion is the cause of recovery without controlled ablations.
- That the results are independently reproducible.

### Methodological risks to address

- **Unequal training budgets:** match environment steps, gradient updates, compute, and hyperparameter search budgets across baselines.
- **Reward leakage:** ensure engineered audio or visual inputs do not encode hidden state or future outcomes unavailable to the intended agent.
- **Evaluation leakage:** separate training seeds from evaluation seeds and freeze model selection before final evaluation.
- **Unclear uncertainty:** define all `±` values and provide raw episode-level returns.
- **Task-scale mismatch:** evaluate each environment with its own normalized score or solved criterion.
- **Missing artifacts:** publish code commit, dependency lockfile, configs, logs, checkpoints, and scripts to reproduce every table.

## Reproducibility checklist

Before presenting this work as a verified research paper, publish:

- [ ] Repository commit hash and tagged release
- [ ] Python version, OS, Gymnasium/environment versions, and dependency lockfile
- [ ] Exact algorithm implementations and hyperparameters for every baseline
- [ ] Training step counts, episode limits, and checkpoint-selection rules
- [ ] Fixed random seeds and separate evaluation seeds
- [ ] Raw per-episode evaluation results in CSV/JSON
- [ ] Learning curves with uncertainty intervals
- [ ] Definitions for every reward term, curiosity bonus, and sensor transformation
- [ ] Ablations for replay, curiosity, fusion, and behavioral anchoring individually
- [ ] Scripts that regenerate every README table from raw results
- [ ] Clear license and instructions for reproducing experiments

## Suggested repository layout

```text
EvoCore-v1/
├── README.md
├── pyproject.toml
├── configs/
│   ├── mountaincar/
│   ├── halfcheetah/
│   ├── frozenlake/
│   └── continual/
├── src/
│   └── evocore/
│       ├── agents/
│       ├── encoders/
│       ├── fusion/
│       ├── replay/
│       ├── continual/
│       └── environments/
├── experiments/
│   ├── ablations/
│   ├── vision/
│   ├── audio/
│   └── continual/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── reproduce_tables.py
├── results/
│   ├── raw/
│   └── figures/
├── tests/
└── docs/
    ├── methodology.md
    └── experiment_protocol.md
```

This is a suggested structure, not a claim that these files already exist.

## Next steps

1. **Make results reproducible:** save raw evaluation returns and generate tables programmatically.
2. **Run stronger baselines:** tune Double-DQN and PPO fairly; include a replay-only baseline and an anchoring-only ablation.
3. **Quantify retention:** measure each task before and after every subsequent task, and report forgetting per task.
4. **Validate sensory grounding:** test audio/video inputs with shuffled, zeroed, delayed, and corrupted signals to detect leakage.
5. **Test generalization:** use held-out seeds and environment variations; then consider a real-world dataset or simulator.
6. **Only then move to a real-world application:** financial market data, robotics, or another domain needs a separate protocol, leakage controls, and domain-specific evaluation. MountainCar results alone do not establish trading performance.

## How to cite

No formal publication or DOI is specified yet. Until one exists, cite the repository URL and release/commit hash.

```text
EvoCore-v1 contributors. "EvoCore-v1: From Curiosity Failure Modes to Continual
Multimodal World Models." Research repository, version/commit: <INSERT HASH>,
year: <INSERT YEAR>.
```

## License

No license is specified in the supplied material. Add a `LICENSE` file before inviting reuse or redistribution.
