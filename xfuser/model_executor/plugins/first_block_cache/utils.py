import contextlib
import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict
from xfuser.core.distributed import (
    get_sp_group,
    get_sequence_parallel_world_size,
)

import torch


@dataclasses.dataclass
class CacheContext:
    first_hidden_states_residual: torch.Tensor = None
    hidden_states_residual: torch.Tensor = None
    encoder_hidden_states_residual: torch.Tensor = None

    def clear_buffers(self):
        self.first_hidden_states_residual = None
        self.hidden_states_residual = None
        self.encoder_hidden_states_residual = None


class FBCachedTransformerBlocks(torch.nn.Module):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_states_first=True,
        enable_fbcache=True,
    ):
        super().__init__()
        self.transformer = transformer
        self.transformer_blocks = transformer_blocks
        self.single_transformer_blocks = single_transformer_blocks
        self.rel_l1_thresh = rel_l1_thresh
        self.return_hidden_states_first = return_hidden_states_first
        self.enable_fbcache = enable_fbcache
        self.cache_context = CacheContext()

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        if not self.enable_fbcache:
            # the branch to disable cache
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
                if not self.return_hidden_states_first:
                    hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
            if self.single_transformer_blocks is not None:
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                for block in self.single_transformer_blocks:
                    hidden_states = block(hidden_states, *args, **kwargs)
                hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :]
            return (
                (hidden_states, encoder_hidden_states)
                if self.return_hidden_states_first
                else (encoder_hidden_states, hidden_states)
            )

        # run first block of transformer
        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]
        hidden_states, encoder_hidden_states = first_transformer_block(
            hidden_states, encoder_hidden_states, *args, **kwargs
        )
        if not self.return_hidden_states_first:
            hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        first_hidden_states_residual = hidden_states - original_hidden_states
        del original_hidden_states

        prev_first_hidden_states_residual = self.cache_context.first_hidden_states_residual

        if prev_first_hidden_states_residual is None:
            use_cache = False
        else:
            mean_diff = (first_hidden_states_residual-prev_first_hidden_states_residual).abs().mean()
            mean_t1 = prev_first_hidden_states_residual.abs().mean()
            if get_sequence_parallel_world_size() > 1:
                mean_diff = get_sp_group().all_gather(mean_diff.unsqueeze(0)).mean()
                mean_t1 = get_sp_group().all_gather(mean_t1.unsqueeze(0)).mean()
            diff = mean_diff / mean_t1
            use_cache = diff < self.rel_l1_thresh

        if use_cache:
            del first_hidden_states_residual
            hidden_states += self.cache_context.hidden_states_residual
            encoder_hidden_states += self.cache_context.encoder_hidden_states_residual
        else:
            original_hidden_states = hidden_states
            original_encoder_hidden_states = encoder_hidden_states
            self.cache_context.first_hidden_states_residual = first_hidden_states_residual
            for block in self.transformer_blocks[1:]:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
                if not self.return_hidden_states_first:
                    hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
            if self.single_transformer_blocks is not None:
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                for block in self.single_transformer_blocks:
                    hidden_states = block(hidden_states, *args, **kwargs)
                encoder_hidden_states, hidden_states = hidden_states.split(
                    [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
                )
            self.cache_context.hidden_states_residual = hidden_states - original_hidden_states
            self.cache_context.encoder_hidden_states_residual = encoder_hidden_states - original_encoder_hidden_states

        return (
            (hidden_states, encoder_hidden_states)
            if self.return_hidden_states_first
            else (encoder_hidden_states, hidden_states)
        )
