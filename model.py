import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding


class FFN(nn.Sequential):
    def __init__(self, emb_dim: int, ffn_factor: int):
        super().__init__(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim * ffn_factor),
            nn.ReLU(),
            nn.LayerNorm(emb_dim * ffn_factor),
            nn.Linear(emb_dim * ffn_factor, emb_dim),
            nn.ReLU(),
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, head_dim: int, ffn_factor: int):
        super().__init__()
        self.head_dim = head_dim
        self.qk_proj = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, 2 * head_dim, bias=False)
        )
        self.rotary = RotaryEmbedding(head_dim)
        self.ffn = FFN(emb_dim, ffn_factor)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        x_head = rearrange(x, "b t (h d) -> b h t d", d=self.head_dim)
        qk = self.qk_proj(x_head)
        q, k = torch.chunk(qk, 2, dim=-1)
        q = self.rotary.rotate_queries_or_keys(q)
        k = self.rotary.rotate_queries_or_keys(k)
        attn = F.scaled_dot_product_attention(q, k, x_head, is_causal=is_causal)
        attn = rearrange(attn, "b h t d -> b t (h d)")
        return x + self.ffn(attn)


class PTR(nn.Module):
    def __init__(self, emb_dim: int, vocab_size: int, head_dim: int, num_attn_layers: int, ffn_factor: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.mhas = nn.ModuleList(
            [
                MultiHeadAttention(emb_dim, head_dim, ffn_factor)
                for _ in range(num_attn_layers)
            ]
        )
        self.out_proj = nn.Linear(emb_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        emb = self.embedding(token_ids)
        for mha in self.mhas:
            emb = mha(emb, is_causal=is_causal)
        logits = self.out_proj(emb)
        return logits
