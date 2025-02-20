import importlib

from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxTransformer2DWrapper


def apply_teacache_on_transformer(transformer, *args, **kwargs):
    if isinstance(transformer, (FluxTransformer2DModel, xFuserFluxTransformer2DWrapper)):
        adapter_name = "flux"
    else:
        raise ValueError(f"Unknown transformer class: {transformer.__class__.__name__}")

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    apply_cache_on_transformer_fn = getattr(adapter_module, "apply_cache_on_transformer")
    return apply_cache_on_transformer_fn(transformer, *args, **kwargs)
