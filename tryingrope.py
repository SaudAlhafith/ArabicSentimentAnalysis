import torch

def rotate_half(x):
    """Splits x into two halves and applies (x1, x2) -> (-x2, x1) transformation."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, position_ids, theta=10000.0):
    """
    Applies RoPE (Rotary Positional Embeddings) to a tensor x.

    Args:
        x: (batch, seq_len, dim) tensor (Q or K matrix in attention).
        position_ids: (batch, seq_len) tensor containing position indices.
        theta: scaling factor for frequencies (default 10,000).

    Returns:
        Tensor of the same shape as x with RoPE applied.
    """
    dim = x.shape[-1]
    half_dim = dim // 2  # RoPE operates on half the dimensions

    # Compute frequencies
    freq_seq = torch.arange(half_dim, dtype=torch.float32, device=x.device)
    freqs = 1.0 / (theta ** (freq_seq / half_dim))  # Compute theta decay

    # Compute angles based on positions
    angle = torch.einsum("i,j->ij", position_ids.flatten(), freqs)  # (batch * seq_len, half_dim)
    angle = angle.view(*position_ids.shape, half_dim)  # (batch, seq_len, half_dim)

    # Apply rotation
    emb_cos = torch.cos(angle)
    emb_sin = torch.sin(angle)
    
    x1, x2 = x.chunk(2, dim=-1)  # Split Q/K into two halves
    x_rot = x1 * emb_cos + rotate_half(x2) * emb_sin  # Rotate the embeddings

    return torch.cat([x_rot, x2], dim=-1)  # Reassemble rotated Q/K

# Example Usage
batch_size = 1
seq_len = 5
dim = 4  # Must be even for RoPE

x = torch.randint(10, (batch_size, seq_len, dim))  # Dummy tensor
position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)  # Position indices

x_rope = apply_rope(x, position_ids)
print("Original x:\n", x)
print("\nRoPE Applied x:\n", x_rope)
