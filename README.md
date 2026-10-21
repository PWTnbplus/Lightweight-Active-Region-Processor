# Lightweight Active Region Processor

This repository contains one runnable demo of **active region localization
with a frozen EfficientNet-B0 encoder and a Deep Q-Network (DQN)**. The demo
has no dataset download: it creates a small synthetic image, trains the policy
online, evaluates it on a held-out image, and saves the predicted region.

> This is a compact proof-of-concept for the EfficientNet + reinforcement
> learning idea. It is not a pretrained detector or a benchmark result.

## Run

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python demo.py
```

The default run uses the EfficientNet-B0 architecture with random weights, so
it works without network access. If ImageNet weights are available locally (or
the machine can download them), use:

```bash
python demo.py --weights pretrained --device auto
```

The output is written to `outputs/active_region_demo.png`. Green is the hidden
ground-truth box used only for scoring, red is the final policy prediction,
and blue boxes show intermediate region proposals.

Useful options:

```bash
python demo.py --episodes 48 --max-steps 16 --weights pretrained
python demo.py --episodes 8 --device cpu --output outputs/cpu_demo.png
```

## Principle

The system is an **active localization loop** rather than a one-shot box
regressor:

```text
image + current region
          │
          ▼
 frozen EfficientNet-B0 ──► visual feature (1280-D)
          │
          ├── current box geometry (x1, y1, x2, y2)
          └── last 10 action one-hot vectors
                              │
                              ▼
                  DQN policy head ──► 9 discrete actions
                              │
                              ▼
                  updated region proposal
```

The initial proposal covers the whole image. At each step, the policy chooses
one action from:

| ID | Action | Effect |
|---:|---|---|
| 0 | `move_left` | shift the region left |
| 1 | `move_right` | shift the region right |
| 2 | `move_up` | shift the region up |
| 3 | `move_down` | shift the region down |
| 4 | `shrink_width` | move both vertical sides inward |
| 5 | `shrink_height` | move both horizontal sides inward |
| 6 | `shrink_both` | shrink width and height |
| 7 | `expand_both` | expand width and height |
| 8 | `stop` | terminate the episode |

For a box `b_t` and target `g`, the environment computes

\[
 r_t = 10[\operatorname{IoU}(b_{t+1},g)-\operatorname{IoU}(b_t,g)] - 0.01.
\]

The `stop` action receives a positive terminal bonus when IoU is at least
0.55 and a penalty otherwise. The DQN minimizes the standard one-step TD
loss:

\[
 y_t = r_t + \gamma(1-d_t)\max_a Q_{\text{target}}(s_{t+1},a),
 \qquad
 L = \operatorname{Huber}(Q(s_t,a_t), y_t).
\]

The target network is periodically synchronized with the online network, and
the replay buffer breaks the correlation between consecutive region edits.
EfficientNet is frozen; only the small DQN head is optimized. This keeps the
policy lightweight while retaining visual information from the current crop.

## Information boundary

During training, the target box is part of the **environment**, because it is
needed to calculate the reward. It is deliberately excluded from the state
passed to the DQN. During evaluation, the policy sees only the image crop,
current proposal geometry, and action history. The target box is read after
the rollout only to calculate the displayed IoU and draw the green reference
box.

## Project scope

There is intentionally only one demo in this repository: `demo.py`. The
synthetic scene makes the execution reproducible and avoids bundling datasets,
large model checkpoints, experiment logs, or generated artifacts.
