import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_CHECKPOINT_DIR = Path(
    "logs/retail/btgear/lightning_logs/version_0/checkpoints"
)
DEFAULT_BEHAVIORS = ["cart", "pv", "fav", "buy"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a BT-GEAR Lightning checkpoint and export per-head and "
            "head-averaged 4x4 behavior-transition decay matrices."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint to visualize. If omitted, the newest checkpoint under "
            f"{DEFAULT_CHECKPOINT_DIR} is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/transition_decay"),
        help="Directory for the PNG, CSV, and metadata files.",
    )
    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=DEFAULT_BEHAVIORS,
        help=(
            "Behavior labels in embedding-index order, excluding padding. "
            "Retail defaults to: cart pv fav buy."
        ),
    )
    return parser.parse_args()


def resolve_checkpoint(checkpoint):
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    candidates = list(DEFAULT_CHECKPOINT_DIR.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint was provided and no checkpoint was found under "
            f"{DEFAULT_CHECKPOINT_DIR}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_decay_coefficients(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    matching_keys = [
        key for key in state_dict if key.endswith("transition_decay_logits")
    ]
    if len(matching_keys) != 1:
        raise KeyError(
            "Expected exactly one transition_decay_logits tensor, found "
            f"{matching_keys}."
        )

    logits = state_dict[matching_keys[0]]
    if logits.ndim != 3 or logits.shape[1] != logits.shape[2]:
        raise ValueError(
            "Expected [num_heads, num_behaviors+1, num_behaviors+1], got "
            f"{tuple(logits.shape)}."
        )

    coefficients = F.softplus(logits).detach().cpu().numpy()[:, 1:, 1:]
    return coefficients, payload, matching_keys[0]


def write_matrix_csv(path, matrix, labels):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["query_behavior \\ key_behavior", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{value:.6f}" for value in row]])


def annotate_matrix(axis, matrix, vmin, vmax):
    midpoint = vmin + (vmax - vmin) / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value < midpoint else "black"
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )


def plot_heatmaps(coefficients, labels, output_path, epoch):
    mean_matrix = coefficients.mean(axis=0)
    matrices = [*coefficients, mean_matrix]
    titles = [
        *[f"Attention head {index + 1}" for index in range(len(coefficients))],
        "Mean across heads",
    ]

    vmin = float(coefficients.min())
    vmax = float(coefficients.max())
    figure, axes = plt.subplots(
        1,
        len(matrices),
        figsize=(4.4 * len(matrices), 4.2),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    image = None
    for axis, matrix, title in zip(axes, matrices, titles):
        image = axis.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks(range(len(labels)), labels=labels)
        axis.set_yticks(range(len(labels)), labels=labels)
        axis.set_xlabel(r"Historical key behavior $b_j$")
        axis.set_ylabel(r"Current query behavior $b_i$")
        annotate_matrix(axis, matrix, vmin, vmax)

    epoch_text = "unknown" if epoch is None else str(epoch)
    figure.suptitle(
        f"BT-GEAR behavior-transition temporal decay (Epoch {epoch_text})",
        fontsize=14,
    )
    figure.colorbar(image, ax=axes.tolist(), shrink=0.78, label="Decay coefficient")
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return mean_matrix


def main():
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    coefficients, payload, state_key = load_decay_coefficients(checkpoint)

    num_behaviors = coefficients.shape[1]
    if len(args.behaviors) != num_behaviors:
        raise ValueError(
            f"Checkpoint contains {num_behaviors} non-padding behaviors, but "
            f"{len(args.behaviors)} labels were provided: {args.behaviors}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    epoch = payload.get("epoch")
    step = payload.get("global_step")
    figure_path = args.output_dir / "transition_decay_heatmaps.png"
    mean_matrix = plot_heatmaps(
        coefficients, args.behaviors, figure_path, epoch
    )

    for head_index, matrix in enumerate(coefficients, start=1):
        write_matrix_csv(
            args.output_dir / f"transition_decay_head_{head_index}.csv",
            matrix,
            args.behaviors,
        )
    write_matrix_csv(
        args.output_dir / "transition_decay_mean.csv",
        mean_matrix,
        args.behaviors,
    )

    metadata = {
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_epoch_zero_based": epoch,
        "global_step": step,
        "state_dict_key": state_key,
        "behavior_order": args.behaviors,
        "matrix_semantics": "row=query/current behavior, column=key/historical behavior",
        "coefficient_min": float(coefficients.min()),
        "coefficient_max": float(coefficients.max()),
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    np.set_printoptions(precision=4, suppress=True)
    print(f"Checkpoint: {checkpoint}")
    print(f"Behavior order: {args.behaviors}")
    print("Mean transition-decay matrix (row=query, column=key):")
    print(mean_matrix)
    print(f"Saved visualization and matrices to: {args.output_dir}")


if __name__ == "__main__":
    main()
