import torch
import torch.nn as nn

from .GEAR import GEAR


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value))


class TemporalBiasGEAR(GEAR):
    def _build_time_bias(self, behavior_seqs, time_gaps):
        raise NotImplementedError

    def log2feats(self, item_seqs, behavior_seqs, time_bias):
        device = item_seqs.device
        time_bias = self._build_time_bias(behavior_seqs, time_bias)

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
