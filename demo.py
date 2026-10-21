"""One-file demo of active region localization with EfficientNet + DQN.

The demo trains a small DQN on synthetic images at runtime.  EfficientNet-B0
is a frozen visual encoder; the policy learns which geometric action to apply
to the current region proposal.  Ground-truth boxes are used only by the
training environment to calculate reward and by the final evaluator to report
IoU.  They are never part of the policy state.
"""

from __future__ import annotations

import argparse
from collections import deque, namedtuple
import copy
import random
from pathlib import Path
from typing import Deque, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torchvision import transforms as T
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


Box = Tuple[float, float, float, float]
NUM_ACTIONS = 9
HISTORY_LENGTH = 10
ACTION_NAMES = (
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "shrink_width",
    "shrink_height",
    "shrink_both",
    "expand_both",
    "stop",
)


def clamp_box(box: Sequence[float]) -> Box:
    """Keep a normalized xyxy box valid and non-empty."""

    x1, y1, x2, y2 = (float(value) for value in box)
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 - x1 < 1e-3:
        centre = (x1 + x2) / 2.0
        x1, x2 = max(0.0, centre - 5e-4), min(1.0, centre + 5e-4)
    if y2 - y1 < 1e-3:
        centre = (y1 + y2) / 2.0
        y1, y2 = max(0.0, centre - 5e-4), min(1.0, centre + 5e-4)
    return x1, y1, x2, y2


def intersection_over_union(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def apply_action(box: Sequence[float], action: int, step_fraction: float = 0.15) -> Box:
    """Apply one of the eight region edits; action 8 is the STOP action."""

    x1, y1, x2, y2 = clamp_box(box)
    width, height = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    dx, dy = step_fraction * width, step_fraction * height

    if action == 0:
        x1, x2 = x1 - dx, x2 - dx
    elif action == 1:
        x1, x2 = x1 + dx, x2 + dx
    elif action == 2:
        y1, y2 = y1 - dy, y2 - dy
    elif action == 3:
        y1, y2 = y1 + dy, y2 + dy
    elif action == 4:
        x1, x2 = x1 + dx, x2 - dx
    elif action == 5:
        y1, y2 = y1 + dy, y2 - dy
    elif action == 6:
        x1, x2, y1, y2 = x1 + dx, x2 - dx, y1 + dy, y2 - dy
    elif action == 7:
        x1, x2, y1, y2 = x1 - dx, x2 + dx, y1 - dy, y2 + dy
    return clamp_box((x1, y1, x2, y2))


def make_scene(seed: int, image_size: int = 224) -> Tuple[Image.Image, Box]:
    """Create a deterministic toy image so the demo needs no dataset download."""

    rng = random.Random(seed)
    background = Image.new("RGB", (image_size, image_size), (18, 24, 38))
    draw = ImageDraw.Draw(background)

    for _ in range(16):
        x = rng.randrange(image_size)
        y = rng.randrange(image_size)
        radius = rng.randrange(max(3, image_size // 50), max(5, image_size // 18))
        colour = tuple(rng.randrange(20, 90) for _ in range(3))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour)

    width = rng.randint(image_size // 4, image_size // 2)
    height = rng.randint(image_size // 4, image_size // 2)
    x1 = rng.randint(image_size // 10, image_size - width - image_size // 10)
    y1 = rng.randint(image_size // 10, image_size - height - image_size // 10)
    x2, y2 = x1 + width, y1 + height
    target_colour = (235, 174, 40)
    draw.rounded_rectangle((x1, y1, x2, y2), radius=max(3, image_size // 40),
                           fill=target_colour, outline=(255, 244, 190), width=3)
    draw.line((x1 + width // 5, y1 + height // 2, x2 - width // 5, y1 + height // 2),
              fill=(112, 56, 18), width=max(2, image_size // 45))
    draw.line((x1 + width // 2, y1 + height // 5, x1 + width // 2, y2 - height // 5),
              fill=(112, 56, 18), width=max(2, image_size // 45))
    return background, (x1 / image_size, y1 / image_size, x2 / image_size, y2 / image_size)


class EfficientNetEncoder:
    """Frozen EfficientNet-B0 feature extractor for the current crop."""

    def __init__(self, device: torch.device, weights_mode: str = "none") -> None:
        self.device = device
        weights = None
        if weights_mode == "pretrained":
            try:
                weights = EfficientNet_B0_Weights.DEFAULT
                self.backbone = efficientnet_b0(weights=weights).features
                print("[encoder] using ImageNet-pretrained EfficientNet-B0")
            except Exception as error:  # offline machines should still run the demo
                print(f"[encoder] pretrained weights unavailable ({error}); using random weights")
                weights = None
                self.backbone = efficientnet_b0(weights=None).features
        else:
            self.backbone = efficientnet_b0(weights=None).features
            print("[encoder] using EfficientNet-B0 architecture with random weights")

        self.backbone = self.backbone.to(device).eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        self.feature_dim = 1280
        self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        if weights is not None:
            self.mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(3, 1, 1)
            self.std = torch.tensor((0.229, 0.224, 0.225), device=device).view(3, 1, 1)
        else:
            self.mean = None
            self.std = None

    @torch.inference_mode()
    def encode(self, image: Image.Image, box: Sequence[float]) -> np.ndarray:
        width, height = image.size
        x1, y1, x2, y2 = box
        crop = image.crop((x1 * width, y1 * height, x2 * width, y2 * height))
        if crop.width <= 1 or crop.height <= 1:
            crop = image
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        if self.mean is not None and self.std is not None:
            tensor = (tensor - self.mean) / self.std
        features = self.pool(self.backbone(tensor)).flatten(1)
        return features[0].float().cpu().numpy()


class QNetwork(nn.Module):
    """Small DQN head; EfficientNet remains frozen and is never updated."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, NUM_ACTIONS),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


Transition = namedtuple("Transition", "state action reward next_state done")


def build_state(encoder: EfficientNetEncoder, image: Image.Image, box: Box,
                action_history: np.ndarray) -> np.ndarray:
    """State = visual crop feature + current box geometry + action history."""

    visual = encoder.encode(image, box).astype(np.float32)
    geometry = np.asarray(box, dtype=np.float32)
    return np.concatenate((visual, geometry, action_history), dtype=np.float32)


def select_action(policy: QNetwork, state: np.ndarray, epsilon: float,
                  rng: random.Random, device: torch.device) -> int:
    if rng.random() < epsilon:
        return rng.randrange(NUM_ACTIONS)
    with torch.inference_mode():
        tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        return int(policy(tensor).argmax(dim=1).item())


def optimize(policy: QNetwork, target: QNetwork, replay: Deque[Transition],
             optimizer: torch.optim.Optimizer, batch_size: int, gamma: float,
             device: torch.device) -> float | None:
    if len(replay) < batch_size:
        return None
    batch = random.sample(replay, batch_size)
    states = torch.from_numpy(np.stack([item.state for item in batch])).to(device)
    actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=device)
    next_states = torch.from_numpy(np.stack([item.next_state for item in batch])).to(device)
    done = torch.tensor([item.done for item in batch], dtype=torch.float32, device=device)

    current_q = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target(next_states).max(dim=1).values
        expected_q = rewards + gamma * (1.0 - done) * next_q
    loss = nn.functional.smooth_l1_loss(current_q, expected_q)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def train(policy: QNetwork, target: QNetwork, encoder: EfficientNetEncoder,
          scenes: Sequence[Tuple[Image.Image, Box]], episodes: int, max_steps: int,
          device: torch.device, seed: int) -> None:
    rng = random.Random(seed)
    replay: Deque[Transition] = deque(maxlen=4000)
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-4)
    batch_size, gamma = 32, 0.98
    losses: List[float] = []
    global_step = 0

    policy.train()
    target.eval()
    for episode in range(episodes):
        image, target_box = scenes[episode % len(scenes)]
        box: Box = (0.0, 0.0, 1.0, 1.0)
        history = np.zeros(NUM_ACTIONS * HISTORY_LENGTH, dtype=np.float32)
        state = build_state(encoder, image, box, history)
        total_reward = 0.0

        for step in range(max_steps):
            fraction = episode / max(episodes - 1, 1)
            epsilon = max(0.05, 1.0 - 0.95 * fraction)
            action = select_action(policy, state, epsilon, rng, device)
            next_box = apply_action(box, action)
            previous_iou = intersection_over_union(box, target_box)
            next_iou = intersection_over_union(next_box, target_box)
            reward = 10.0 * (next_iou - previous_iou) - 0.01
            done = action == 8 or step == max_steps - 1
            if action == 8:
                reward += 2.0 if previous_iou >= 0.55 else -0.5

            if done:
                next_state = np.zeros_like(state)
            else:
                one_hot = np.zeros(NUM_ACTIONS, dtype=np.float32)
                one_hot[action] = 1.0
                history = np.concatenate((history[NUM_ACTIONS:], one_hot))
                next_state = build_state(encoder, image, next_box, history)

            replay.append(Transition(state, action, reward, next_state, done))
            loss = optimize(policy, target, replay, optimizer, batch_size, gamma, device)
            if loss is not None:
                losses.append(loss)
            total_reward += reward
            global_step += 1
            if global_step % 50 == 0:
                target.load_state_dict(policy.state_dict())
            if done:
                break
            box, state = next_box, next_state

        if (episode + 1) % max(1, episodes // 4) == 0 or episode == 0:
            mean_loss = sum(losses[-20:]) / max(len(losses[-20:]), 1)
            print(f"[train] episode {episode + 1:>3}/{episodes}  "
                  f"reward={total_reward:>7.3f}  epsilon={epsilon:.3f}  "
                  f"loss={mean_loss:.4f}")


def evaluate(policy: QNetwork, encoder: EfficientNetEncoder, image: Image.Image,
             target_box: Box, max_steps: int, device: torch.device) -> Tuple[Box, List[int], List[Box]]:
    policy.eval()
    box: Box = (0.0, 0.0, 1.0, 1.0)
    history = np.zeros(NUM_ACTIONS * HISTORY_LENGTH, dtype=np.float32)
    actions: List[int] = []
    trajectory: List[Box] = [box]
    with torch.inference_mode():
        for _ in range(max_steps):
            state = build_state(encoder, image, box, history)
            action = select_action(policy, state, 0.0, random.Random(0), device)
            actions.append(action)
            box = apply_action(box, action)
            trajectory.append(box)
            if action == 8:
                break
            one_hot = np.zeros(NUM_ACTIONS, dtype=np.float32)
            one_hot[action] = 1.0
            history = np.concatenate((history[NUM_ACTIONS:], one_hot))
    return box, actions, trajectory


def save_visualization(image: Image.Image, target_box: Box, predicted_box: Box,
                       trajectory: Sequence[Box], path: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    def pixels(box: Box) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

    draw.rectangle(pixels(target_box), outline=(55, 220, 100), width=4)
    for index, box in enumerate(trajectory[::2]):
        colour = (80, 150, 255) if index < len(trajectory[::2]) - 1 else (255, 70, 70)
        draw.rectangle(pixels(box), outline=colour, width=2)
    draw.rectangle(pixels(predicted_box), outline=(255, 70, 70), width=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=24,
                        help="online DQN training episodes (default: 24)")
    parser.add_argument("--max-steps", type=int, default=12,
                        help="maximum region edits per episode (default: 12)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="synthetic image side length (default: 224)")
    parser.add_argument("--weights", choices=("none", "pretrained"), default="none",
                        help="EfficientNet weights; pretrained falls back to none offline")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/active_region_demo.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 0 or args.max_steps < 1:
        raise ValueError("--episodes must be non-negative and --max-steps must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    print(f"[setup] device={device}  actions={NUM_ACTIONS}  max_steps={args.max_steps}")

    encoder = EfficientNetEncoder(device, args.weights)
    state_dim = encoder.feature_dim + 4 + NUM_ACTIONS * HISTORY_LENGTH
    policy = QNetwork(state_dim).to(device)
    target = copy.deepcopy(policy).to(device)

    scenes = [make_scene(args.seed + 100 + index, args.image_size)
              for index in range(max(8, args.episodes))]
    train(policy, target, encoder, scenes, args.episodes, args.max_steps, device, args.seed)

    evaluation_image, target_box = make_scene(args.seed + 10000, args.image_size)
    predicted_box, actions, trajectory = evaluate(
        policy, encoder, evaluation_image, target_box, args.max_steps, device)
    final_iou = intersection_over_union(predicted_box, target_box)
    save_visualization(evaluation_image, target_box, predicted_box, trajectory, args.output)
    action_text = " -> ".join(ACTION_NAMES[action] for action in actions)
    print(f"[eval] initial_iou={intersection_over_union(trajectory[0], target_box):.3f}  "
          f"final_iou={final_iou:.3f}  steps={len(actions)}")
    print(f"[eval] actions={action_text or '<none>'}")
    print(f"[saved] {args.output.resolve()}")


if __name__ == "__main__":
    main()
