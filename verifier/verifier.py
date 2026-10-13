from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoder import VisionTransformer
from .ac_predictor import VisionTransformerPredictorAC


@dataclass
class VerifierOutput:
    """One candidate's verifier result.

    predicted_latents: [M+1, HW, embed_dim]  z_hat_0 (context) .. z_hat_M
    delta_progress:    [M]                   Delta-p_hat^[1] .. Delta-p_hat^[M]
    cumulative_progress: scalar              sum of delta_progress (paper
    """

    predicted_latents: torch.Tensor
    delta_progress: torch.Tensor
    cumulative_progress: torch.Tensor


class LatentWorldModelVerifier(nn.Module):

    def __init__(self, encoder: VisionTransformer, predictor: VisionTransformerPredictorAC):
        super().__init__()
        assert encoder.embed_dim == predictor.predictor_embed.in_features, (
            "encoder.embed_dim must match the predictor's expected input dim "
            f"(got encoder={encoder.embed_dim}, predictor expects={predictor.predictor_embed.in_features})"
        )
        self.encoder = encoder
        self.predictor = predictor

    @torch.no_grad()
    def encode_observation(self, obs):
        """PAPER Eq. 4: o_t -> latent. obs: [B, C, H, W] -> [B, HW, embed_dim]."""
        return self.encoder(obs)

    @torch.no_grad()
    def score_candidate(self, obs_latent, actions, states, extrinsics=None, task_id=0, use_kv_cache=True):
        """Run the predictor's autoregressive rollout for ONE candidate
        action chunk (already split into M sub-chunks) and return its
        verifier output (paper Eq. 4 + Sec 3.4 "Inference").

        obs_latent: [B, HW, embed_dim]           encoder output at o_t
        actions:    [B, M, K, action_embed_dim]  M sub-chunks of this candidate
        states:     [B, M, state_dim] or [B, state_dim]
        """
        rollout_fn = self.predictor.rollout_kv_cached if use_kv_cache else self.predictor.rollout
        frame_preds, values = rollout_fn(obs_latent, actions, states, extrinsics=extrinsics, task_id=task_id)
        # values: [M, B, value_dim] -> per-batch-item [M] delta-progress trace
        delta_progress = values.squeeze(-1)  # [M, B]
        cumulative = delta_progress.sum(dim=0)  # [B]  paper: "sum of these M estimates"
        return VerifierOutput(
            predicted_latents=frame_preds,
            delta_progress=delta_progress,
            cumulative_progress=cumulative,
        )

    @torch.no_grad()
    def rank_candidates(self, obs_latent, candidate_actions, states, extrinsics=None, task_id=0, use_kv_cache=True):
        """PAPER Sec 3.2
        
        obs_latent:        [1, HW, embed_dim]
        candidate_actions: [K, M, action_chunk_size, action_embed_dim] -- K
                            candidates, each pre-split into M sub-chunks (paper Sec 3.4)
        states:             [M, state_dim] or [state_dim], shared across candidates
                             (same o_t / s_t is being re-evaluated K times).

        Returns:
            best_idx: int
            cumulative_progress: [K] tensor
        """
        K = candidate_actions.shape[0]
        obs_latent_rep = obs_latent.expand(K, -1, -1)
        states_rep = states.unsqueeze(0).expand(K, *states.shape) if states.dim() < 3 else states
        extr_rep = extrinsics.unsqueeze(0).expand(K, *extrinsics.shape) if extrinsics is not None else None

        out = self.score_candidate(
            obs_latent_rep, candidate_actions, states_rep, extrinsics=extr_rep,
            task_id=task_id, use_kv_cache=use_kv_cache,
        )
        best_idx = int(torch.argmax(out.cumulative_progress).item())
        return best_idx, out.cumulative_progress
