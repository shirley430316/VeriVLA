import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Stochastic depth (inlined from timm, to avoid a timm dependency for
    this one helper). Standard ViT regularizer, not paper-specific."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


def build_action_block_causal_attention_mask(T, H, W, add_tokens=1, value_tokens=0):
    """Block-causal attention mask for the action-conditioned verifier.

    Args:
        T:            number of sub-chunks (rollout steps) currently in the sequence.
        H, W:         visual patch grid height/width (HW = H*W visual tokens/frame).
        add_tokens:   size of the "cond" prefix per frame = action tokens (K) + 1 state
                      token (+1 extrinsics token if used). This is the DYNAMICS
                      conditioning prefix in Fig. 5's per-sub-chunk block.
        value_tokens: number of trailing Delta-p (task-conditioned progress query)
                      tokens per frame. Paper Sec 3.4 uses exactly 1.

    Returns:
        BoolTensor [T*N_T, T*N_T], True = attend, False = blocked.
    """
    cond = add_tokens
    N_T = cond + (H * W) + value_tokens  # tokens in one sub-chunk block
    N = T * N_T

    # ---- (R1) block-causal across sub-chunks -------------------------------
    # PAPER: Sec 3.4 "Architecture"
    mask = torch.zeros(N, N).bool()
    mask_block = torch.ones(N_T, N_T).bool()
    for t1 in range(T):
        for t2 in range(0, t1 + 1):
            mask[t1 * N_T : (t1 + 1) * N_T, t2 * N_T : (t2 + 1) * N_T] = mask_block

    # ---- (R2) Delta-p columns are private ----------------------------------
    if value_tokens > 0:
        val_local = torch.arange(cond + H * W, N_T)
        val_global = torch.cat([t * N_T + val_local for t in range(T)])
        mask[:, val_global] = False                     
        for t in range(T):                              
            v = t * N_T + val_local                     
            mask[v.unsqueeze(1), v.unsqueeze(0)] = True

    return mask


def rotate_queries_or_keys(x, pos):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2), outer product

    emb_sin = freq.sin()
    emb_cos = freq.cos()
    # NOTE: this expansion duplicates frequencies across the vector pair; kept
    # as-is for bit-compatibility with pretrained V-JEPA 2 checkpoints.
    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)

    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class DropPath(nn.Module):
    """Stochastic depth (standard ViT regularizer, not paper-specific)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, wide_silu=True
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x, mask=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob if self.training else 0.0,
                is_causal=self.is_causal, attn_mask=attn_mask,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RoPEAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        tokens_per_frame = int((H_patches or self.grid_size) * (W_patches or self.grid_size))
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        tokens_per_frame = int((H_patches or self.grid_size) * (W_patches or self.grid_size))
        tokens_per_row = W_patches or self.grid_size
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        tokens_per_frame = int((H_patches or self.grid_size) * (W_patches or self.grid_size))
        tokens_per_row = W_patches or self.grid_size
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        B, N, C = x.size()
        grid_depth = int(N // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        s = 0
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim

        if s < self.head_dim:
            qr, kr = q[..., s:], k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob if self.training else 0.0,
                is_causal=self.is_causal, attn_mask=attn_mask,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = RoPEAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                use_sdpa=use_sdpa, is_causal=is_causal, grid_size=grid_size, proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                use_sdpa=use_sdpa, is_causal=is_causal, proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        if isinstance(self.attn, RoPEAttention):
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        else:
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ACRoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal

    def _get_frame_pos(self, ids, H_patches, W_patches):
        tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches, W_patches):
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches, W_patches):
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(
        self, x, mask=None, attn_mask=None,
        T=None, H=None, W=None,
        action_tokens=0, value_tokens=0,
        kv_cache=None,  # PAPER Sec 3.4 (O1): reused KV cache across the autoregressive rollout
    ):
        B, N, C = x.size()

        # -- KV-cache frame offset: with caching, ``x`` holds only the NEW
        # sub-chunk(s), so RoPE positions must be shifted by how many
        # sub-chunks are already cached (paper O1 optimization).
        tokens_per_frame = action_tokens + H * W + value_tokens
        if kv_cache is not None:
            T_offset = kv_cache[0].size(2) // tokens_per_frame
        else:
            T_offset = 0
        frame_pos = torch.arange(T_offset, T_offset + T, device=x.device)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)
        else:
            mask = torch.arange(T * H * W, device=x.device) + T_offset * H * W
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)

        h_mask *= self.grid_size / H
        w_mask *= self.grid_size / W

        if action_tokens > 0 or value_tokens > 0:
            x = x.view(B, T, action_tokens + H * W + value_tokens, C)

            if action_tokens > 0:
                a = x[:, :, :action_tokens, :].reshape(B, T * action_tokens, C)
                a_qkv = self.qkv(a).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
                aq, ak, av = a_qkv[0], a_qkv[1], a_qkv[2]
                apos = frame_pos.repeat_interleave(action_tokens)
                aqd = rotate_queries_or_keys(aq[..., : self.d_dim], pos=apos)
                akd = rotate_queries_or_keys(ak[..., : self.d_dim], pos=apos)
                action_q = torch.cat([aqd, aq[..., self.d_dim :]], dim=-1)
                action_k = torch.cat([akd, ak[..., self.d_dim :]], dim=-1)
                action_v = av
            else:
                action_q = action_k = action_v = None

            patch_x = x[:, :, action_tokens : action_tokens + H * W, :].flatten(1, 2)
            value_x = x[:, :, action_tokens + H * W :, :]  # PAPER Fig. 5: the Delta-p token(s)

            if value_tokens > 0:
                value_qkv = (
                    self.qkv(value_x.flatten(1, 2))
                    .unflatten(-1, (3, self.num_heads, -1))
                    .permute(2, 0, 3, 1, 4)
                )
                value_q, value_k, value_v = value_qkv[0], value_qkv[1], value_qkv[2]
                vframe_pos = frame_pos.repeat_interleave(value_tokens)
                vqd = rotate_queries_or_keys(value_q[..., : self.d_dim], pos=vframe_pos)
                vkd = rotate_queries_or_keys(value_k[..., : self.d_dim], pos=vframe_pos)
                vqr, vkr = value_q[..., self.d_dim :], value_k[..., self.d_dim :]
                value_q = torch.cat([vqd, vqr], dim=-1).view(B, self.num_heads, T, value_tokens, -1)
                value_k = torch.cat([vkd, vkr], dim=-1).view(B, self.num_heads, T, value_tokens, -1)
                value_v = value_v.view(B, self.num_heads, T, value_tokens, -1)

            x = patch_x

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        s = 0
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim

        if s < self.head_dim:
            qr, kr = q[..., s:], k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if action_tokens > 0 or value_tokens > 0:

            def merge_(tx, ta, tv=None):
                tx = tx.view(B, self.num_heads, T, H * W, -1)
                ta = ta.view(B, self.num_heads, T, action_tokens, -1)
                tokens = [ta, tx]
                if tv is not None:
                    tokens.append(tv)
                return torch.cat(tokens, dim=3).flatten(2, 3)

            q = merge_(q, action_q, value_q if value_tokens > 0 else None)
            k = merge_(k, action_k, value_k if value_tokens > 0 else None)
            v = merge_(v, action_v, value_v if value_tokens > 0 else None)

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        # PAPER Sec 3.4 (O1): concatenate with the cached K/V from earlier
        # sub-chunks so only the new sub-chunk needs a forward pass.
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_cache = (k, v)

        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.proj_drop_prob if self.training else 0.0,
                is_causal=self.is_causal and kv_cache is None,
                attn_mask=attn_mask,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, new_cache


class ValueACBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = ACRoPEAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                use_sdpa=use_sdpa, is_causal=is_causal, grid_size=grid_size, proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                use_sdpa=use_sdpa, is_causal=is_causal, proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0, value_tokens=0, kv_cache=None):
        y = self.norm1(x)
        new_cache = None
        if isinstance(self.attn, ACRoPEAttention):
            y, new_cache = self.attn(
                y, mask=mask, attn_mask=attn_mask, T=T, H=H, W=W,
                action_tokens=action_tokens, value_tokens=value_tokens, kv_cache=kv_cache,
            )
        else:
            y, new_cache = self.attn(y, mask=mask, attn_mask=attn_mask, kv_cache=kv_cache)
        x = x + self.drop_path(y)
        y = self.norm2(x)
        x = x + self.drop_path(self.mlp(y))
        return x, new_cache
