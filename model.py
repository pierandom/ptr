import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from transformers import PreTrainedModel, PretrainedConfig


class PTRConfig(PretrainedConfig):
    model_type = "PTR"

    def __init__(
        self,
        emb_dim: int,
        vocab_size: int,
        head_dim: int,
        num_attn_layers: int,
        ffn_factor: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.head_dim = head_dim
        self.num_attn_layers = num_attn_layers
        self.ffn_factor = ffn_factor


class FFN(nn.Sequential):
    def __init__(self, emb_dim: int, ffn_factor: int):
        super().__init__(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim * ffn_factor, bias=False),
            nn.ReLU(),
            nn.LayerNorm(emb_dim * ffn_factor),
            nn.Linear(emb_dim * ffn_factor, emb_dim, bias=False),
            nn.ReLU(),
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, head_dim: int, ffn_factor: int):
        super().__init__()
        self.head_dim = head_dim
        self.qk_proj = nn.Sequential(
            nn.LayerNorm(head_dim), nn.Linear(head_dim, 2 * head_dim, bias=False)
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


class PTR(PreTrainedModel):
    config_class = PTRConfig

    def __init__(self, config: PTRConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.mhas = nn.ModuleList(
            [
                MultiHeadAttention(config.emb_dim, config.head_dim, config.ffn_factor)
                for _ in range(config.num_attn_layers)
            ]
        )
        self.out_proj = nn.Linear(config.emb_dim, config.vocab_size)

    def forward(self, token_ids: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        emb = self.embedding(token_ids)
        for mha in self.mhas:
            emb = mha(emb, is_causal=is_causal)
        logits = self.out_proj(emb)
        return logits
