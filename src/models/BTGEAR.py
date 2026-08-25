import torch
import torch.nn as nn
import torch.nn.functional as F

from .GEAR import GEAR


class BTGEAR(GEAR):
    """GEAR with behavior-transition-aware temporal decay.

    For attention head h, query behavior b_i, and key behavior b_j, the
    temporal bias is

        -softplus(alpha[h, b_i, b_j]) * log(1 + delta_t[i, j]).

    Index 0 is reserved for padding, matching GEAR's behavior embedding.
    """

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

        # Initialize every behavior transition with GEAR's original
        # head-specific slope. This makes BTGEAR equivalent to GEAR at the
        # start of training, before transition-specific parameters diverge.
        initial_decay = self.slopes[:, None, None].expand(
            n_head, n_b + 1, n_b + 1
        ).clone()
        inverse_softplus = torch.log(torch.expm1(initial_decay))
        self.transition_decay_logits = nn.Parameter(inverse_softplus)

    def decay_coefficients(self):
        """Return the positive learned coefficients [head, query, key]."""
        return F.softplus(self.transition_decay_logits)

    def _build_transition_time_bias(self, behavior_seqs, time_gaps):
        """Build a [batch, head, query_pos, key_pos] temporal bias tensor."""
        batch_size, seq_len = behavior_seqs.shape

        query_behaviors = behavior_seqs.unsqueeze(2).expand(-1, -1, seq_len)
        key_behaviors = behavior_seqs.unsqueeze(1).expand(-1, seq_len, -1)

        decay = self.decay_coefficients()
        transition_decay = decay[
            :,
            query_behaviors.reshape(-1),
            key_behaviors.reshape(-1),
        ]
        transition_decay = transition_decay.transpose(0, 1).reshape(
            batch_size, seq_len, seq_len, self.n_head
        )
        transition_decay = transition_decay.permute(0, 3, 1, 2).contiguous()

        log_time_gaps = torch.log1p(torch.clamp(time_gaps, min=0))
        return -transition_decay * log_time_gaps.unsqueeze(1)

    def log2feats(self, item_seqs, behavior_seqs, time_bias):
        device = item_seqs.device
        time_bias = self._build_transition_time_bias(behavior_seqs, time_bias)

        # The remaining architecture is unchanged from the official GEAR.
        seqs_i = self.item_emb(item_seqs)
        seqs_b = self.behavior_emb(behavior_seqs)

        seq_len_i = item_seqs.size(1)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len_i).to(device)
        seqs_i = self.dropout(seqs_i)
        seqs_i = self.LayerNorm(seqs_i)
        for block in self.item_attention_blocks:
            seqs_i = block(seqs_i, src_mask=attn_mask)

        seq_len_b = behavior_seqs.size(1)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len_b).to(device)
        seqs_b = self.dropout(seqs_b)
        seqs_b = self.LayerNorm(seqs_b)
        for block in self.behavior_attention_blocks:
            seqs_b = block(seqs_b, src_mask=attn_mask, time_bias=time_bias)

        seqs_i = self.dropout(seqs_i)
        seqs_b = self.dropout(seqs_b)

        seqs_alt = torch.empty(
            item_seqs.size(0),
            seq_len_b + seq_len_i,
            self.d_model,
            dtype=seqs_i.dtype,
            device=device,
        )
        seqs_alt[:, 1::2, :] = seqs_i
        seqs_alt[:, 0::2, :] = seqs_b

        attn_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len_b + seq_len_i
        ).to(device)
        for block in self.upper_transformer:
            seqs_alt = block(seqs_alt, src_mask=attn_mask)

        return seqs_alt
