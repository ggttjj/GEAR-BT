import torch
import torch.nn as nn
import torch.nn.functional as F

from ._temporal_ablation import TemporalBiasGEAR, inverse_softplus


class BTGEARS(TemporalBiasGEAR):
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

        self.n_b = n_b

        initial_decay = self.slopes.mean().expand(n_b + 1, n_b + 1).clone()
        self.transition_decay_logits = nn.Parameter(
            inverse_softplus(initial_decay)
        )

    def decay_coefficients(self):
        return F.softplus(self.transition_decay_logits)

    def _build_time_bias(self, behavior_seqs, time_gaps):
        seq_len = behavior_seqs.size(1)
        query_behaviors = behavior_seqs.unsqueeze(2).expand(-1, -1, seq_len)
        key_behaviors = behavior_seqs.unsqueeze(1).expand(-1, seq_len, -1)

        transition_decay = self.decay_coefficients()[
            query_behaviors, key_behaviors
        ]
        transition_decay = transition_decay.unsqueeze(1).expand(
            -1, self.n_head, -1, -1
        )

        log_time_gaps = torch.log1p(torch.clamp(time_gaps, min=0))
        return -transition_decay * log_time_gaps.unsqueeze(1)
