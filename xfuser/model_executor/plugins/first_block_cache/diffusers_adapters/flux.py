import functools
import unittest

import torch
from diffusers import DiffusionPipeline, FluxTransformer2DModel

from xfuser.model_executor.plugins.first_block_cache import utils


def apply_cache_on_transformer(
    transformer: FluxTransformer2DModel,
    *,
    rel_l1_thresh=0.6,
    use_cache=True,
    return_hidden_states_first=False,
):
    cached_transformer_blocks = torch.nn.ModuleList(
        [
            utils.FBCachedTransformerBlocks(
                transformer.transformer_blocks,
                transformer.single_transformer_blocks,
                transformer=transformer,
                rel_l1_thresh=rel_l1_thresh,
                return_hidden_states_first=return_hidden_states_first,
                enable_fbcache=use_cache,
            )
        ]
    )
    dummy_single_transformer_blocks = torch.nn.ModuleList()

    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def new_forward(
        self,
        *args,
        **kwargs,
    ):
        with unittest.mock.patch.object(
            self,
            "transformer_blocks",
            cached_transformer_blocks,
        ), unittest.mock.patch.object(
            self,
            "single_transformer_blocks",
            dummy_single_transformer_blocks,
        ):
            return original_forward(
                *args,
                **kwargs,
            )

    transformer.forward = new_forward.__get__(transformer)

    return transformer

