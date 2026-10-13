import math
from functools import partial

import torch
import torch.nn as nn

from .attention import ValueACBlock, build_action_block_causal_attention_mask
from .utils import trunc_normal_


class VisionTransformerPredictorAC(nn.Module):
    """Action-Conditioned latent world model verifier predictor.
    """

    _ACTION_ENCODERS = {
        7: "action_encoder_7d_eef",
        14: "action_encoder",
        16: "action_encoder_16d",
    }
    _STATE_ENCODERS = {
        7: "state_encoder_7d_eef",
        14: "state_encoder",
        16: "state_encoder_16d",
    }

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=True,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_activation_checkpointing=False,
        use_rope=True,
        action_embed_dim=14,
        action_chunk_size=10,   # K: actions per sub-chunk (paper Sec 3.4, "M sub-chunks")
        value_dim=1,            # Delta-p head output dim (1 = scalar delta-progress)
        max_frames=32,           # max sub-chunks: sizes the causal-mask horizon
        num_tasks=40,            # number of task-specific Delta-p query identities (Sec 3.4)
        use_extrinsics=False,
        **kwargs,
    ):
        super().__init__()
        self.is_frame_causal = is_frame_causal
        self.use_extrinsics = use_extrinsics
        self.max_frames = max_frames

        # action and state encoders
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.action_encoder_7d_eef = nn.Linear(7, predictor_embed_dim, bias=True)
        self.action_encoder_16d = nn.Linear(16, predictor_embed_dim, bias=True)
        self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.state_encoder_7d_eef = nn.Linear(7, predictor_embed_dim, bias=True)
        self.state_encoder_16d = nn.Linear(16, predictor_embed_dim, bias=True)
        self.extrinsics_encoder = nn.Linear(action_embed_dim - 1, predictor_embed_dim, bias=True)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = self.img_height // self.patch_size
        self.grid_width = self.img_width // self.patch_size
        self.use_activation_checkpointing = use_activation_checkpointing
        self.uniform_power = uniform_power
        self.use_rope = use_rope

        self.num_task_tokens = 1

        # ---- predictor transformer blocks (shared dynamics + Delta-p path,
        # separated only by the attention mask -- Fig. 5) ----
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList(
            [
                ValueACBlock(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # one learnable embedding per task
        self.num_tasks = num_tasks
        self.task_query_embedding = nn.Embedding(num_tasks, predictor_embed_dim)
        self.value_mlp = nn.Linear(predictor_embed_dim, value_dim, bias=True)

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def build_attn_mask(self, action_len, max_frames=10, grid_height=None, grid_width=None, task_token=1):
        """PAPER Fig. 5: builds the mask that isolates the Delta-p path from
        the dynamics (latent-prediction) path. See attention.py for the
        exact rule."""
        grid_height = self.grid_height if grid_height is None else grid_height
        grid_width = self.grid_width if grid_width is None else grid_width
        return build_action_block_causal_attention_mask(
            max_frames,
            grid_height,
            grid_width,
            add_tokens=action_len + 1,   # action tokens + 1 state token (cond prefix)
            value_tokens=task_token,     # trailing per-sub-chunk Delta-p token
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # action/state encoding (shared by _assemble / rollout_kv_cached)
    def _encode_action_state(self, actions, states):
        """
        actions: [..., K, action_dim]   (action_dim in {7, 14, 16})
        states:  [..., state_dim]       (state_dim in {7, 14, 16}, independent of action_dim)
        Returns:
            a: [..., K, D]
            s: [..., 1, D]
        """
        action_dim, state_dim = actions.shape[-1], states.shape[-1]
        try:
            action_attr = self._ACTION_ENCODERS[action_dim]
        except KeyError:
            raise ValueError(f"unsupported action dim: {action_dim} "
                              f"(expected one of {sorted(self._ACTION_ENCODERS)})")
        try:
            state_attr = self._STATE_ENCODERS[state_dim]
        except KeyError:
            raise ValueError(f"unsupported state dim: {state_dim} "
                              f"(expected one of {sorted(self._STATE_ENCODERS)})")
        a = getattr(self, action_attr)(actions)
        s = getattr(self, state_attr)(states).unsqueeze(-2)
        return a, s

    def _delta_progress_query(self, task_id, T, B, device):
        tid = torch.as_tensor(task_id, device=device, dtype=torch.long).view(-1)
        if tid.numel() == 1 and B > 1:
            tid = tid.expand(B)
        task_term = self.task_query_embedding(tid)                      # [B, D]
        D = task_term.size(-1)
        return task_term.view(B, 1, 1, D).expand(B, T, 1, D)            # [B, T, 1, D]

    def _assemble(self, visual, actions, states, extrinsics, task_id):
        """Build the flattened token sequence for ``T`` sub-chunks.

        Args:
            visual:      [B, T, HW, D]  (encoder output per sub-chunk, or
                         predicted z_hat for sub-chunks > 0 during rollout)
            actions:     [B, T, K, action_embed_dim]
            states:      [B, T, action_embed_dim]
            extrinsics:  [B, T, action_embed_dim - 1] or None
            task_id:     int or [B] long tensor (paper Sec 3.4 task query),
                         broadcast unchanged across all T sub-chunks

        Returns:
            [B, T * tokens_per_frame, D]
        """
        B, T, HW, D = visual.shape
        a, s = self._encode_action_state(actions, states)   # a: [B,T,K,D], s: [B,T,1,D]

        v = self._delta_progress_query(task_id, T, B, visual.device)  # [B, T, 1, D]

        if self.use_extrinsics:
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)  # [B, T, 1, D]
            seq = torch.cat([a, s, e, visual, v], dim=2)
        else:
            seq = torch.cat([a, s, visual, v], dim=2)
        return seq.flatten(1, 2)                            # [B, T * tokens_per_frame, D]

    @staticmethod
    def _expand_states(states, steps):
        if states is not None and states.dim() == 2:
            states = states.unsqueeze(1).repeat(1, steps, 1)
        return states

    def _run_blocks(self, h, attn_mask, T, cond_tokens):
        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing:
                h = torch.utils.checkpoint.checkpoint(
                    blk, h,
                    mask=None, attn_mask=attn_mask,
                    T=T, H=self.grid_height, W=self.grid_width,
                    action_tokens=cond_tokens,
                    value_tokens=self.num_task_tokens,
                    use_reentrant=False,
                )[0]
            else:
                h = blk(
                    h, mask=None, attn_mask=attn_mask,
                    T=T, H=self.grid_height, W=self.grid_width,
                    action_tokens=cond_tokens,
                    value_tokens=self.num_task_tokens,
                )[0]
        return h

    def forward(self, x, actions, states, extrinsics=None, return_value=True, task_id=0):
        """
        x:       [B, T * HW, D] visual latent tokens (T sub-chunks, from encoder.py)
        actions: [B, T, K, action_embed_dim]
        states:  [B, T, action_embed_dim]
        task_id: int or [B] long tensor selecting the task-specific Delta-p query

        Returns:
            out:          [B, T*HW, embed_dim]  -- predicted latents z_hat (per sub-chunk)
            value_logits: [B, T, value_dim]      -- predicted Delta-p_hat (per sub-chunk)
        """
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        HW = self.grid_height * self.grid_width
        T = N_ctxt // HW

        action_len = actions.shape[2]
        cond_tokens = action_len + 1
        tokens_per_frame = cond_tokens + HW + self.num_task_tokens

        visual = x.view(B, T, HW, D)
        seq = self._assemble(visual, actions, states, extrinsics, task_id)

        L = seq.size(1)
        attn_mask = self.build_attn_mask(action_len, max_frames=T)[:L, :L].to(x.device, non_blocking=True)
        h = self._run_blocks(seq, attn_mask, T, cond_tokens=cond_tokens)

        h = h.view(B, T, tokens_per_frame, D)
        # PAPER Fig. 5: z^[i] is read out first ...
        value_tok = h[:, :, cond_tokens + HW, :]                      # [B, T, D]  (Delta-p^[i] token, post-attention)

        out = h[:, :, cond_tokens: cond_tokens + HW, :].flatten(1, 2)
        out = self.predictor_proj(self.predictor_norm(out))           # [B, T * HW, embed_dim] == z_hat

        if return_value:
            # ... "and then predicts the corresponding progress increment
            # Delta-p_hat" (Sec 3.4) -- decoding the (already latent-aware,
            # via attention) Delta-p token with the Delta-p head.
            value_logits = self.value_mlp(value_tok)                  # [B, T, value_dim] == Delta-p_hat
            return out, value_logits
        return out

    @torch.no_grad()
    def rollout(self, x, actions, states, extrinsics=None, task_id=0):
        """
        x:       [B, HW, D] initial visual latent (encoder output at o_t)
        actions: [B, M, K, action_embed_dim]   -- M sub-chunks of the candidate action chunk
        states:  [B, M, action_embed_dim] (or [B, action_embed_dim])
        task_id: int or [B] long tensor selecting the task-specific Delta-p query

        Returns:
            frame_preds: [B, M + 1, HW, embed_dim]  -- z_hat_0 (context) .. z_hat_M
            values:      [M, B, value_dim]           -- Delta-p_hat^[1] .. Delta-p_hat^[M]
                          (sum over dim 0 = the cumulative predicted progress
                          used to rank candidates, paper Sec 3.4 "Inference":
                          "executes the action chunk with the largest
                          cumulative predicted progress, computed as the sum
                          of these M estimates.")
        """
        B, N_ctxt, D = x.shape
        HW = self.grid_height * self.grid_width
        steps = actions.size(1)
        states = self._expand_states(states, steps)

        visual = x
        frame_preds = [x.view(B, -1, HW, D)]
        values = []

        for t in range(steps):
            Tcur = t + 1
            extr = extrinsics[:, :Tcur] if self.use_extrinsics else None
            predictions, progress = self.forward(
                visual, actions[:, :Tcur], states[:, :Tcur],
                extrinsics=extr, task_id=task_id,
            )
            nxt = predictions[:, -HW:]
            values.append(progress[:, -1])
            frame_preds.append(nxt.unsqueeze(1))
            visual = torch.cat([visual, nxt], dim=1)

        return torch.cat(frame_preds, dim=1), torch.stack(values)

    @torch.no_grad()
    def rollout_kv_cached(self, x, actions, states, extrinsics=None, task_id=0):
        """Same semantics as ``rollout`` but processes one new sub-chunk per
        step and reuses cached keys/values, i.e. O(M) instead of O(M^2)."""
        B, N_ctxt, embed_dim = x.shape
        HW = self.grid_height * self.grid_width
        T_ctxt = N_ctxt // HW
        frame_preds = [x.view(B, T_ctxt, HW, embed_dim)]
        x = self.predictor_embed(x)
        D = x.size(-1)
        steps = actions.size(1)
        states = self._expand_states(states, steps)

        action_len = actions.shape[2]
        cond_tokens = action_len + 1

        kv_caches = [None] * len(self.predictor_blocks)
        visual = x.view(B, T_ctxt, HW, D)
        seq_so_far = 0
        values = []

        for t in range(steps):
            a_t, s_t = self._encode_action_state(actions[:, t:t + 1], states[:, t:t + 1])
            # Same task query token at every step; temporal position comes from
            # RoPE (vframe_pos) inside the attention block (paper Fig. 5).
            v_t = self._delta_progress_query(task_id, 1, B, x.device)  # [B, 1, 1, D]
            vis_t = visual[:, -1:]                                                # [B, 1, HW, D]

            if self.use_extrinsics:
                e_t = self.extrinsics_encoder(extrinsics[:, t]).view(B, 1, 1, D)
                new_tokens = torch.cat([a_t, s_t, e_t, vis_t, v_t], dim=2)
            else:
                new_tokens = torch.cat([a_t, s_t, vis_t, v_t], dim=2)
            new_tokens = new_tokens.flatten(1, 2)                                # [B, tokens_per_frame, D]

            total = seq_so_far + new_tokens.size(1)
            attn_mask = self.build_attn_mask(action_len, max_frames=steps)[seq_so_far:total, :total].to(
                x.device, non_blocking=True
            )

            h = new_tokens
            for i, blk in enumerate(self.predictor_blocks):
                h, kv_caches[i] = blk(
                    h, mask=None, attn_mask=attn_mask,
                    T=1, H=self.grid_height, W=self.grid_width,
                    action_tokens=cond_tokens,
                    value_tokens=self.num_task_tokens,
                    kv_cache=kv_caches[i],
                )
            seq_so_far = total

            values.append(self.value_mlp(h[:, cond_tokens + HW, :]))
            nxt = self.predictor_proj(self.predictor_norm(h[:, cond_tokens: cond_tokens + HW, :]))
            frame_preds.append(nxt.unsqueeze(1))
            visual = self.predictor_embed(nxt).unsqueeze(1)

        return torch.cat(frame_preds, dim=1), torch.stack(values)


def vit_ac_predictor(**kwargs):
    return VisionTransformerPredictorAC(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
