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
        dropout_p: float,
        attention_heads_type: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.head_dim = head_dim
        self.num_attn_layers = num_attn_layers
        self.ffn_factor = ffn_factor
        self.pos_encoding = pos_encoding
        self.dropout_p = dropout_p
        self.attention_heads_type = attention_heads_type


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


class MultiHeadProjection(nn.Module):
    def __init__(self, num_heads: int, head_dim_in: int, head_dim_out: int):
        super().__init__()
        self.ln = nn.LayerNorm(head_dim_in)
        init_tensor = torch.randn(
            num_heads, head_dim_in, 2 * head_dim_out
        ) / torch.sqrt(torch.tensor(head_dim_in))
        self.mh_linear = nn.Parameter(init_tensor)

    def forward(self, x: torch.Tensor):
        x_norm = self.ln(x)
        x_proj = torch.einsum("b h t d, h d k -> b h t k", x_norm, self.mh_linear)
        x0, x1 = torch.chunk(x_proj, 2, dim=-1)
        return F.silu(x0) * x1


class LeXRotary(nn.Module):
    """Length eXtrapolable positional embedding
    from paper https://arxiv.org/pdf/2212.10554v1.pdf
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.gamma = 1
        range_tensor = torch.repeat_interleave(
            torch.arange(head_dim // 2) / (head_dim // 2), repeats=2
        ).view(1, 1, 1, head_dim)
        self.register_buffer(
            name="theta",
            tensor=torch.pow(1 / 10_000, range_tensor),
            persistent=False,
        )
        self.register_buffer(
            name="zeta",
            tensor=(range_tensor + self.gamma) / (1 + self.gamma),
            persistent=False,
        )
        self.register_buffer(
            name="rot", tensor=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]), persistent=False
        )

    def rotate(self, x: torch.Tensor):
        x_pairs = rearrange(x, "b h t (d p) -> b h t d p", p=2)
        x_rotated = torch.matmul(x_pairs, self.rot)
        return rearrange(x_rotated, "b h t d p -> b h t (d p)")

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        L = q.shape[-2]
        pos = torch.arange(L).view(1, 1, L, 1).to(q)
        cos = torch.cos(pos * self.theta)
        sin = torch.sin(pos * self.theta)
        q_rotated = (q * cos + self.rotate(q) * sin) * self.zeta
        k_rotated = (k * cos + self.rotate(k) * sin) / self.zeta
        return q_rotated, k_rotated


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        head_dim: int,
        ffn_factor: int,
        pos_encoding: str,
        dropout_p: float,
        attention_heads_type: str,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        self.ffn_factor = ffn_factor
        self.pos_encoding = pos_encoding
        self.dropout_p = dropout_p
        self.attention_heads_type = attention_heads_type
        num_heads = emb_dim // head_dim
        match attention_heads_type:
            case "simple":
                self.qkv_proj = SwiGLU(head_dim, 3 * head_dim)
            case "multi_query":
                self.q_proj = MultiHeadProjection(num_heads, head_dim, head_dim)
                self.kv_proj = SwiGLU(head_dim, 2 * head_dim)
            case "classic":
                self.qkv_proj = MultiHeadProjection(num_heads, head_dim, 3 * head_dim)
            case _:
                raise ValueError(attention_heads_type)
        self.out_proj = SwiGLU(emb_dim, emb_dim)
        self.ffn = FFN(emb_dim, ffn_factor)
        match pos_encoding:
            case "rotary":
                self.rotary = RotaryEmbedding(head_dim)
            case "lex_rotary":
                self.lex_rotary = LeXRotary(head_dim)
            case "alibi":
                pass
            case _:
                raise ValueError(pos_encoding)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x_head = rearrange(x, "b t (h d) -> b h t d", d=self.head_dim)
        match self.attention_heads_type:
            case "simple" | "classic":
                qkv = self.qkv_proj(x_head)
                q, k, v = torch.chunk(qkv, 3, dim=-1)
            case "multi_query":
                q = self.q_proj(x_head)
                kv = self.kv_proj(x_head)
                k, v = torch.chunk(kv, 2, dim=-1)
        match self.pos_encoding:
            case "rotary":
                q = self.rotary.rotate_queries_or_keys(q)
                k = self.rotary.rotate_queries_or_keys(k)
            case "lex_rotary":
                q, k = self.lex_rotary(q, k)
            case "alibi":
                pass
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout_p if self.training else 0,
        )
        attn = rearrange(attn, "b h t d -> b t (h d)")
        x_attn = x + self.out_proj(attn)
        return x_attn + self.ffn(x_attn)


class PTR(PreTrainedModel):
    config_class = PTRConfig

    def __init__(self, config: PTRConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.mhas = nn.ModuleList(
            [
                MultiHeadAttention(
                    emb_dim=config.emb_dim,
                    head_dim=config.head_dim,
                    ffn_factor=config.ffn_factor,
                    pos_encoding=config.pos_encoding,
                    dropout_p=config.dropout_p,
                    attention_heads_type=config.attention_heads_type,
                )
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
        scaled_mask = (
            -1.0 * rearrange(m, "h -> 1 h 1 1") * rearrange(mask, "s t -> 1 1 s t")
        )
        return scaled_mask.to(x)

    def forward(
        self,
        token_ids: torch.Tensor,
        is_causal: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
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
        temperature: float = 1.0,
        use_entropy_weights: bool = False,
    ) -> torch.Tensor:
        logits = logits / temperature
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
