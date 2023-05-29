from typing import Optional
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
        pos_encoding: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.head_dim = head_dim
        self.num_attn_layers = num_attn_layers
        self.ffn_factor = ffn_factor
        self.pos_encoding = pos_encoding


class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, 2 * out_dim, bias=False)
        )

    def forward(self, x):
        x0, x1 = torch.chunk(self.layer(x), 2, dim=-1)
        return F.silu(x0) * x1


class FFN(nn.Sequential):
    def __init__(self, emb_dim: int, ffn_factor: int):
        super().__init__(
            SwiGLU(emb_dim, ffn_factor * emb_dim), SwiGLU(ffn_factor * emb_dim, emb_dim)
        )
        self.emb_dim = emb_dim
        self.ffn_factor = ffn_factor


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, head_dim: int, ffn_factor: int, pos_encoding: str):
        super().__init__()
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        self.ffn_factor = ffn_factor
        self.pos_encoding = pos_encoding
        self.qkv_proj = SwiGLU(head_dim, 3 * head_dim)
        self.out_proj = SwiGLU(emb_dim, emb_dim)
        self.ffn = FFN(emb_dim, ffn_factor)
        if self.pos_encoding == "rotary":
            self.rotary = RotaryEmbedding(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x_head = rearrange(x, "b t (h d) -> b h t d", d=self.head_dim)
        qkv = self.qkv_proj(x_head)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        if self.pos_encoding == "rotary":
            q = self.rotary.rotate_queries_or_keys(q)
            k = self.rotary.rotate_queries_or_keys(k)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        attn = rearrange(attn, "b h t d -> b t (h d)")
        attn_out = self.out_proj(attn)
        x_attn = x + attn_out
        return x_attn + self.ffn(x_attn)


class PTR(PreTrainedModel):
    config_class = PTRConfig

    def __init__(self, config: PTRConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.mhas = nn.ModuleList(
            [
                MultiHeadAttention(config.emb_dim, config.head_dim, config.ffn_factor, config.pos_encoding)
                for _ in range(config.num_attn_layers)
            ]
        )
        self.out_proj = nn.Linear(config.emb_dim, config.vocab_size)

    def alibi(self, x: torch.Tensor):
        ctx_len = x.shape[-2]
        num_heads = self.config.emb_dim // self.config.head_dim
        mask = (
            torch.flip(torch.arange(ctx_len + 1), dims=(0,))
            .repeat(ctx_len)
            .view(ctx_len + 1, ctx_len)[1:, :]
        )
        mask = torch.tril(mask) + torch.triu(
            torch.inf * torch.ones(ctx_len, ctx_len), 1
        )
        m = torch.pow(2, -1.0 * torch.arange(0, num_heads // 2, 0.5))
        scaled_mask = -1.0 * rearrange(m, "h -> 1 h 1 1") * rearrange(mask, "s t -> 1 1 s t")
        return scaled_mask.to(x)

    def forward(
        self,
        token_ids: torch.Tensor,
        is_causal: bool = False,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        emb = self.embedding(token_ids)
        if self.config.pos_encoding == "alibi":
            attn_mask = self.alibi(emb)
        for mha in self.mhas:
            emb = mha(emb, attn_mask=attn_mask, is_causal=is_causal)
        logits = self.out_proj(emb)
        return logits

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        use_entropy_weights: bool = False,
    ) -> torch.Tensor:
        if use_entropy_weights:
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = rearrange(log_probs, "b t c -> b c t")
            entropy = self.entropy(logits, targets)
            log_probs = rearrange(entropy, "b t -> b 1 t") * log_probs
            return F.nll_loss(log_probs, targets)
        else:
            # faster and use less memory
            return F.cross_entropy(rearrange(logits, "b t c -> b c t"), targets)

    @torch.no_grad()
    def entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        eps = torch.finfo(logits.dtype).eps
        entropy = -torch.sum(
            probs * torch.log(torch.clamp(probs + eps, max=1)), dim=-1
        ) / torch.log(torch.tensor(logits.shape[-1]))
        preds = torch.argmax(logits, dim=-1)
        entropy = torch.where(preds == targets, entropy, 1.0)
        return entropy
