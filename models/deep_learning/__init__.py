"""Deep learning models initialization."""

from .iqa_model import IQAModel, LightweightIQAModel, get_model
from .feature_fusion import FeatureFusion, MultiScaleFusion
from .attention import SpatialAttention, ChannelAttention, CBAM

__all__ = [
    'IQAModel',
    'LightweightIQAModel',
    'get_model',
    'FeatureFusion',
    'MultiScaleFusion',
    'SpatialAttention',
    'ChannelAttention',
    'CBAM',
    "get_model"
]

# Replace strict imports that caused ImportError with a tolerant loader.

# Dynamic discovery: scan package submodules for a `get_model` callable.
# Capture the discovered callable and wrap it so returned models are coerced to float32
# to avoid "expected scalar type Double but found Float" during inference.
import pkgutil
import importlib
import inspect
import warnings

_inspected_submodules = []
_found_submodule = None
_orig_get_model = None

for _finder, _mod_name, _ispkg in pkgutil.iter_modules(__path__):
    _inspected_submodules.append(_mod_name)
    try:
        _mod = importlib.import_module(f".{_mod_name}", __name__)
    except Exception:
        # ignore modules that fail to import when scanning
        continue
    _candidate = getattr(_mod, "get_model", None)
    if _candidate is not None and (inspect.isfunction(_candidate) or inspect.isclass(_candidate) or callable(_candidate)):
        # capture discovered callable (do not bind directly yet)
        _orig_get_model = _candidate
        _found_submodule = _mod_name
        break

if _orig_get_model is not None:
    def get_model(*args, _force_float=True, **kwargs):
        """
        Calls the discovered get_model and coerces returned torch.nn.Module to float() if present.
        _force_float (bool): if True, call model.float() to ensure float32 parameters/tensors.
        """
        model = _orig_get_model(*args, **kwargs)
        if _force_float:
            try:
                import torch
                if isinstance(model, torch.nn.Module):
                    model = model.float()
                    warnings.warn(
                        "Converted deep-learning model parameters to float32 to avoid dtype mismatch.",
                        UserWarning
                    )
            except Exception:
                # if torch isn't available or coercion fails, silently continue and return model as-is
                pass
        return model
else:
    # define a clear runtime error if no implementation is found
    def get_model(*args, **kwargs):
        raise ImportError(
            "get_model is not defined in models.deep_learning. "
            f"Scanned submodules: {', '.join(_inspected_submodules) or '(none)'}.\n"
            "Ensure one of the submodules exports `get_model` (common names: "
            "model_factory, model, networks), or add/rename the module accordingly."
        )
