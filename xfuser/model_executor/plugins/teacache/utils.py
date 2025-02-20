import contextlib
import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict
from xfuser.core.distributed import (
    get_sp_group,
    get_sequence_parallel_world_size,
)

import torch

class TorchPoly1D:
    def __init__(self, coefficients):
        self.coefficients = torch.tensor(coefficients, dtype=torch.float32)
        self.degree = len(coefficients) - 1

    def __call__(self, x):
        result = torch.zeros_like(x)
        for i, coef in enumerate(self.coefficients):
            result += coef * (x ** (self.degree - i))
        return result

class TeaCachedTransformerBlocks(torch.nn.Module):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        enable_teacache=True,
        num_steps=8,
        rel_l1_thresh=0.6,
        return_hidden_states_first=True,
        coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
    ):
        super().__init__()
        self.transformer = transformer
        self.transformer_blocks = transformer_blocks
        self.single_transformer_blocks = single_transformer_blocks
        self.cnt = 0
        self.enable_teacache = enable_teacache
        self.num_steps = num_steps
        self.rel_l1_thresh = rel_l1_thresh
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.previous_residual_encoder = None
        self.coefficients = coefficients
        self.return_hidden_states_first = return_hidden_states_first
        self.rescale_func = TorchPoly1D(coefficients)

    def forward(self, hidden_states, encoder_hidden_states, temb, *args, **kwargs):
        if not self.enable_teacache:
            # the branch to disable cache
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, temb, *args, **kwargs)
                if not self.return_hidden_states_first:
                    hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
            if self.single_transformer_blocks is not None:
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                for block in self.single_transformer_blocks:
                    hidden_states = block(hidden_states, temb, *args, **kwargs)
                hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :]
            return (
                (hidden_states, encoder_hidden_states)
                if self.return_hidden_states_first
                else (encoder_hidden_states, hidden_states)
            )

        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        first_transformer_block = self.transformer_blocks[0]
        inp = hidden_states.clone()
        temb_ = temb.clone()
        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = first_transformer_block.norm1(inp, emb=temb_)
        if self.cnt == 0 or self.cnt == self.num_steps-1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            mean_diff = (modulated_inp-self.previous_modulated_input).abs().mean()
            mean_t1 = self.previous_modulated_input.abs().mean()
            if get_sequence_parallel_world_size() > 1:
                mean_diff = get_sp_group().all_gather(mean_diff.unsqueeze(0)).mean()
                mean_t1 = get_sp_group().all_gather(mean_t1.unsqueeze(0)).mean()
            self.accumulated_rel_l1_distance += self.rescale_func(mean_diff / mean_t1)
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

        if not should_calc:
            hidden_states += self.previous_residual
            encoder_hidden_states += self.previous_residual_encoder
        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, temb, *args, **kwargs)
                if not self.return_hidden_states_first:
                    hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
            if self.single_transformer_blocks is not None:
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                for block in self.single_transformer_blocks:
                    hidden_states = block(hidden_states, temb, *args, **kwargs)
                encoder_hidden_states, hidden_states = hidden_states.split(
                    [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
                )
            self.previous_residual = hidden_states - original_hidden_states
            self.previous_residual_encoder = encoder_hidden_states - original_encoder_hidden_states

        return (
            (hidden_states, encoder_hidden_states)
            if self.return_hidden_states_first
            else (encoder_hidden_states, hidden_states)
        )
