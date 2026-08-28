import torch
import torch.nn as nn
import torch.nn.functional as F

from ._temporal_ablation import TemporalBiasGEAR, inverse_softplus


class GEART(TemporalBiasGEAR):
    def __init__(
        self,
        max_len: int = None,
        num_items: int = None,
        n_layer: int = None,
        n_head: int = None,
        num_users: int = None,
        n_b: int = None,
        d_model: int = None,
        dropout: float = 0.0,
    ):
        super().__init__(
            max_len=max_len,
            num_items=num_items,
            n_layer=n_layer,
            n_head=n_head,
            num_users=num_users,
            n_b=n_b,
            d_model=d_model,
            dropout=dropout,
        )

        initial_decay = self.slopes.clone().detach()
        self.head_decay_logits = nn.Parameter(inverse_softplus(initial_decay))

    def decay_coefficients(self):
        return F.softplus(self.head_decay_logits)

    def _build_time_bias(self, behavior_seqs, time_gaps):
        del behavior_seqs
        decay = self.decay_coefficients().view(1, self.n_head, 1, 1)
        log_time_gaps = torch.log1p(torch.clamp(time_gaps, min=0))
        return -decay * log_time_gaps.unsqueeze(1)
