
# BT-GEAR: Behavior-Transition-Aware Temporal Decay for GEAR

[English](README.md) | [简体中文](README_zh.md)

BT-GEAR is a research prototype for multi-behavior sequential recommendation. It extends the official implementation of **GEAR: Generalized Alternating Regressor for Multi-Behavior Sequential Recommendation** with a behavior-transition-aware temporal attention bias.

## Architecture

![BT-GEAR architecture](BT-GEAR.png)

BT-GEAR preserves GEAR's item-attention branch, behavior-attention branch, alternating cross-signal fusion, and autoregressive prediction heads. The architectural change is localized to the temporal bias in behavior-specific attention, where the fixed head-wise coefficient is replaced by a learnable behavior-transition-aware decay matrix.

## Motivation

GEAR models a head-wise temporal decay in its behavior-sequence attention. Its temporal bias can be summarized as

$$
\Phi_{ij}^{(h)}=-\alpha_h\log(1+\Delta t_{ij}),
$$

where the decay coefficient $\alpha_h$ depends on the attention head but not on the behavior types. Consequently, the same head applies the same temporal decay to different transitions such as `pv -> buy`, `cart -> buy`, and `fav -> buy`.

BT-GEAR conditions the decay coefficient on both the current query behavior $b_i$ and the historical key behavior $b_j$:

$$
\Phi_{ij}^{(h)}=-\mathrm{softplus}(\theta_{h,b_i,b_j})\log(1+\Delta t_{ij}).
$$

The `softplus` function keeps every decay coefficient non-negative. The transition parameters are initialized from GEAR's original head-wise slopes, so BT-GEAR starts from the same temporal bias as GEAR and learns transition-specific differences during training.

Only the temporal bias of the lower behavior-attention blocks is changed. The item branch, upper alternating Transformer, prediction heads, and training loss remain unchanged, making the comparison with GEAR controlled and interpretable.

## Ablation Study Design

The ablation study uses four controlled variants. They share the same item branch, behavior branch, alternating Transformer, prediction heads, loss functions, dataset split, and evaluation protocol. Only the parameterization of the temporal decay coefficient is changed.

### GEAR

The original baseline uses one fixed decay coefficient for each attention head:

$$
\Phi_{ij}^{(h)}=-\alpha_h\log(1+\Delta t_{ij}).
$$

The coefficients are not updated by gradient descent and do not depend on the behavior types.

### GEAR-T

GEAR-T replaces the fixed head-wise coefficients with learnable positive coefficients:

$$
\Phi_{ij}^{(h)}=-\mathrm{softplus}(\theta_h)\log(1+\Delta t_{ij}).
$$

This variant tests whether an improvement comes only from making GEAR's temporal slopes trainable.

### BT-GEAR-S

BT-GEAR-S learns one behavior-transition matrix shared by all attention heads:

$$
\Phi_{ij}^{(h)}=-\mathrm{softplus}(\theta_{b_i,b_j})\log(1+\Delta t_{ij}).
$$

The row represents the current query behavior $b_i$, and the column represents the historical key behavior $b_j$. This variant tests whether distinguishing behavior transitions is useful without head-specific matrices. The suffix `S` means that the transition matrix is shared across heads.

### BT-GEAR

The full model learns a separate behavior-transition matrix for every attention head:

$$
\Phi_{ij}^{(h)}=-\mathrm{softplus}(\theta_{h,b_i,b_j})\log(1+\Delta t_{ij}).
$$

For Retail with two attention heads and four valid behavior types, GEAR-T learns 2 valid decay coefficients, BT-GEAR-S learns $4\times4=16$, and BT-GEAR learns $2\times4\times4=32$. The implementation also stores a padding row and column, so the actual tensors have shapes $5\times5$ and $2\times5\times5$; the padding entries are excluded from interpretation and visualization.

| Comparison | Question answered |
|---|---|
| GEAR vs. GEAR-T | Does learning the original head-wise temporal slope help? |
| GEAR-T vs. BT-GEAR | Does conditioning decay on the behavior transition add value? |
| BT-GEAR-S vs. BT-GEAR | Is a separate transition matrix for each attention head useful? |
| GEAR vs. BT-GEAR | Does the complete behavior-transition-aware design improve the baseline? |

## Main Files

- `src/models/BTGEAR.py`: behavior-transition-aware temporal decay model.
- `src/models/BTGEARS.py`: behavior-transition matrix shared across attention heads.
- `src/models/GEART.py`: learnable head-wise temporal decay ablation.
- `src/configs/retail_btgear.yaml`: BT-GEAR configuration for the Retail dataset.
- `src/configs/retail_btgear_s.yaml`: BT-GEAR-S configuration for the Retail dataset.
- `src/configs/retail_geart.yaml`: GEAR-T configuration for the Retail dataset.
- `scripts/visualize_transition_decay.py`: checkpoint-to-heatmap visualization script.
- `src/models/GEAR.py`: unchanged upstream GEAR implementation.
- `src/configs/retail.yaml`: memory-safe baseline configuration used on an 8 GB GPU.

## Environment

Install PyTorch for your CUDA version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
pip install -U "jsonargparse[signatures]>=4.27.7"
```

## Dataset

Download the datasets from the [dataset folder provided by the original GEAR repository](https://drive.google.com/drive/folders/1RxTTZtcjdcK063pkRblRxzVDqVZpZX-R?usp=sharing) and place the Retail data at:

```text
data/retail.txt
```

The dataset and generated preprocessing files are intentionally excluded from Git.

## Training

Train the memory-safe GEAR baseline:

```bash
python run.py --config src/configs/retail.yaml fit
```

Train BT-GEAR:

```bash
python run.py --config src/configs/retail_btgear.yaml fit
```

The supplied Retail configurations use `train_batch_size=16` and `accumulate_grad_batches=8`, retaining an effective batch size of 128 while avoiding the full-vocabulary CUDA out-of-memory error on an 8 GB GPU.

To inspect the curves:

```bash
tensorboard --logdir logs --port 6006
```

Then open `http://localhost:6006`.

## Visualizing the Learned Transition Decay

The visualization script automatically loads the newest BT-GEAR checkpoint and exports the per-head matrices, their mean matrix, CSV files, and metadata:

```bash
python scripts/visualize_transition_decay.py \
  --output-dir figures/transition_decay
```

An explicit checkpoint can also be supplied:

```bash
python scripts/visualize_transition_decay.py \
  --checkpoint "path/to/checkpoint.ckpt" \
  --output-dir figures/transition_decay
```

Rows represent the current query behavior $b_i$, while columns represent the historical key behavior $b_j$. A larger coefficient produces faster temporal decay and a stronger attention penalty for an old interaction; a smaller coefficient preserves that transition's influence for longer.

The following heatmaps were extracted from the Epoch 31 Retail checkpoint:

![Learned behavior-transition decay heatmaps](figures/transition_decay_epoch31/transition_decay_heatmaps.png)

The coefficients vary substantially across both behavior transitions and attention heads (approximately 0.006 to 2.901 in this checkpoint). This confirms that the model has moved away from its behavior-agnostic initialization and learned head-specific transition patterns. The heatmap is an interpretation of learned parameters rather than evidence of a causal behavioral relationship.

## Preliminary Results

The following are best validation results from the current single-seed (`seed=42`) Retail pre-study. The original GEAR run was trained through Epoch 82; BT-GEAR was still training when these values were recorded.

| Metric | GEAR best | Epoch | BT-GEAR best | Epoch | Relative change |
|---|---:|---:|---:|---:|---:|
| NDCG@10 | 0.715395 | 80 | **0.715942** | 26 | **+0.08%** |
| Recall@10 | 0.830635 | 80 | **0.832336** | 27 | **+0.20%** |
| NDCG@5 | 0.696784 | 80 | **0.697569** | 26 | **+0.11%** |
| Recall@5 | 0.773137 | 80 | **0.775801** | 27 | **+0.34%** |
| NDCG@1 | **0.608378** | 76 | 0.608017 | 26 | -0.06% |

BT-GEAR currently improves four of the five best-to-best ranking metrics and improves all five metrics when compared with GEAR at the same Epoch 27. These results support the feasibility of behavior-transition-aware temporal decay, but they are preliminary: multiple random seeds, additional datasets, and statistical significance tests are still required for a publication-level conclusion.

## Reproducibility Notes

- Training outputs, checkpoints, TensorBoard events, and datasets are excluded by `.gitignore`.
- The Windows-compatible datamodule type annotation and 8 GB GPU configuration changes are included in the repository.
- For an exact resume, pass a checkpoint using `fit --ckpt_path <checkpoint>`; the checkpoint itself is not uploaded.
