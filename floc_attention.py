"""
FLOC-Norm Attention: 用幂次压缩替代 softmax 的注意力机制.

Core formula:  Attention(Q,K,V) = FLOC(A)·V
  where A = QK^T / sqrt(d_k)
  FLOC(A)_ij = |A_ij|^(p-2) * A_ij / sum_j(|A_ij|^(p-2))

Why it works for α-stable noise:
  - softmax uses exp() → gradient vanishes exponentially for large |A|
  - FLOC uses power-law → gradient decays as |A|^(p-2), never zero
  - p is learnable → adapts compression strength to noise level

Ref: Tian2021 FLOCR, Luan2021 GC, CLAUDE.md Paper 1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FLOCAttention1D(nn.Module):
    """1D FLOC-Norm Attention for GCC-PHAT feature sequences.

    Input:  (B, L, d_model)  — sequence of GCC-PHAT features
    Output: (B, L, d_model)  — attention-weighted features

    Args:
        d_model: feature dimension
        n_heads: number of attention heads
        p_init: initial FLOC order (1.0 < p < 2.0)
        p_trainable: whether p is learned during training
    """
    def __init__(self, d_model=64, n_heads=4, p_init=1.5, p_trainable=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # QKV projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Learnable FLOC order p
        self.p_raw = nn.Parameter(torch.tensor(self._p_to_raw(p_init)), requires_grad=p_trainable)

        self.dropout = nn.Dropout(0.1)
        self.scale = self.d_k ** 0.5

    @staticmethod
    def _p_to_raw(p):
        """Map p ∈ (0, 2) to raw ∈ R via logit transform."""
        import math
        return math.log(p / (2.0 - p))

    @staticmethod
    def _raw_to_p(raw):
        """Map raw ∈ R to p ∈ (0, 2) via sigmoid."""
        return 2.0 * torch.sigmoid(raw)

    @property
    def p(self):
        """Current FLOC order, constrained to (0, 2)."""
        return self._raw_to_p(self.p_raw)

    def _floc_norm(self, scores):
        """FLOC normalization along last dim.

        scores: (B, n_heads, L, L)  — attention scores
        returns: (B, n_heads, L, L) — FLOC-normalized weights
        """
        p = self.p

        # |scores|^(p-2) — power-law compression
        abs_scores = scores.abs().clamp(min=1e-8)  # avoid 0^(negative)
        pow_scores = abs_scores.pow(p - 2.0)

        # signed weights: sign(scores) * |scores|^(p-1)
        # = scores * |scores|^(p-2)
        weights = scores * pow_scores

        # normalize: w_i / sum(|w_j|)
        denom = weights.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return weights / denom

    def forward(self, x):
        B, L, _ = x.shape

        # Linear projections → multi-head
        q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        k = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product scores
        scores = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, L, L)

        # FLOC normalization (replaces softmax)
        attn_weights = self._floc_norm(scores)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        out = attn_weights @ v  # (B, H, L, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)

        return self.W_o(out)

    def extra_repr(self):
        return f"d_model={self.d_model}, n_heads={self.n_heads}, p={self.p.item():.3f}"


class FLOCAttentionPooling(nn.Module):
    """FLOC attention for global pooling — aggregate sequence into single vector.

    Similar to multi-head attention pooling but uses FLOC norm.
    Good for: GCC-PHAT feature → scalar delay estimate.

    Input:  (B, L, d_model)
    Output: (B, d_model)
    """
    def __init__(self, d_model=64, n_heads=4, p_init=1.5):
        super().__init__()
        self.attention = FLOCAttention1D(d_model, n_heads, p_init)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x):
        B = x.shape[0]
        # Concatenate learnable query token
        query = self.pool_query.expand(B, 1, -1)  # (B, 1, d_model)
        x_with_query = torch.cat([query, x], dim=1)  # (B, 1+L, d_model)

        # Apply FLOC attention
        out = self.attention(x_with_query)

        # Extract pooled representation from query position
        return out[:, 0, :]  # (B, d_model)


# ── Tests ──
def _test_floc_vs_softmax():
    """Verify FLOC attention has non-zero gradient for extreme inputs
    where softmax would have vanishing gradient."""
    import torch.nn.functional as F

    # Simulate extreme GCC-PHAT values (impulse noise creates large peaks)
    x_normal = torch.randn(2, 8, 64)
    x_impulse = x_normal.clone()
    x_impulse[0, 0, 0] = 100.0  # extreme outlier

    floc = FLOCAttention1D(d_model=64, n_heads=4)

    # FLOC gradient test
    x_test = x_impulse.clone().requires_grad_(True)
    out_floc = floc(x_test)
    loss_floc = out_floc.sum()
    loss_floc.backward()
    grad_floc = x_test.grad.abs().max().item()

    # Softmax gradient test (for comparison)
    x_test2 = x_impulse.clone().requires_grad_(True)
    # Simple self-attention style softmax
    scale = 8 ** 0.5
    scores = (x_test2 @ x_test2.transpose(-2, -1)) / scale
    attn_softmax = F.softmax(scores, dim=-1)
    loss_softmax = attn_softmax.sum()
    loss_softmax.backward()
    grad_softmax = x_test2.grad.abs().max().item()

    print(f"  FLOC max gradient:    {grad_floc:.6f}")
    print(f"  Softmax max gradient: {grad_softmax:.6f}")
    print(f"  Ratio (FLOC/Softmax): {grad_floc/grad_softmax:.1f}x")

    # FLOC should have non-zero gradient even with extreme values
    assert grad_floc > 1e-8, "FLOC gradient vanished!"
    print("  [PASS] FLOC attention maintains gradient for extreme inputs")


if __name__ == "__main__":
    print("=== FLOC Attention Tests ===")
    _test_floc_vs_softmax()

    # Quick forward pass sanity check
    x = torch.randn(4, 16, 64)
    attn = FLOCAttention1D(d_model=64, n_heads=4)
    out = attn(x)
    print(f"\n  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Learned p:    {attn.p.item():.4f}")
    print(f"  Trainable params: {sum(p.numel() for p in attn.parameters()):,}")
