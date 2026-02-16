from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch
from torch import nn

if TYPE_CHECKING:
    from inference_clt.inference_clt import InferenceCLT


def _nested_getattr(obj: object, attr: str) -> Any:
    value = obj
    for name in attr.split("."):
        value = getattr(value, name)
    return value


def _extract_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x:
        first = x[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Expected activation tensor, got {type(x)}")


class ActivationLoader:
    """Lightweight forward-hook wrapper for Hugging Face transformer activations.

    The wrapper returns layer-first tensors for compatibility with `InferenceCLT`:
    `(n_layers, n_tokens, d_model_like)`, where `n_tokens = batch * seq_len` when
    `flatten_tokens=True`.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_attr: str = "model.layers",
        per_layer_attrs: tuple[str, ...] = ("mlp.input", "mlp.output"),
        layer_indices: Optional[Sequence[int]] = None,
        flatten_tokens: bool = True,
    ):
        self.model = model
        self.layer_attr = layer_attr
        self.per_layer_attrs = per_layer_attrs
        self.flatten_tokens = flatten_tokens

        layers = _nested_getattr(model, layer_attr)
        n_layers = len(layers)
        self.layer_indices = list(range(n_layers)) if layer_indices is None else list(layer_indices)

        self._parsed_attrs = [self._parse_attr(a) for a in per_layer_attrs]
        self._cache: dict[int, dict[str, dict[str, torch.Tensor]]] = {
            i: defaultdict(dict) for i in self.layer_indices
        }
        self._hooks: list[Any] = []

        for layer_idx in self.layer_indices:
            layer = layers[layer_idx]
            for component_attr, port in self._parsed_attrs:
                component = _nested_getattr(layer, component_attr)

                if port == "input":

                    def hook(_module, args, _output, layer_idx=layer_idx, component_attr=component_attr):
                        act = _extract_tensor(args)
                        self._cache[layer_idx][component_attr]["input"] = self._convert_activation(act)

                    self._hooks.append(component.register_forward_hook(hook))
                elif port == "output":

                    def hook(_module, _args, output, layer_idx=layer_idx, component_attr=component_attr):
                        act = _extract_tensor(output)
                        self._cache[layer_idx][component_attr]["output"] = self._convert_activation(act)

                    self._hooks.append(component.register_forward_hook(hook))
                else:
                    raise ValueError(f"Unknown activation port '{port}' in attr '{component_attr}.{port}'")

    @classmethod
    def from_inference_clt(
        cls,
        model: nn.Module,
        inference_clt: InferenceCLT,
        *,
        flatten_tokens: bool = True,
    ) -> "ActivationLoader":
        if inference_clt.input_attr is None or inference_clt.output_attr is None:
            raise ValueError("InferenceCLT is missing input_attr/output_attr metadata")

        return cls(
            model=model,
            layer_attr=inference_clt.layer_attr,
            per_layer_attrs=(inference_clt.input_attr, inference_clt.output_attr),
            layer_indices=inference_clt.layer_indices,
            flatten_tokens=flatten_tokens,
        )

    def _parse_attr(self, attr: str) -> tuple[str, str]:
        parts = attr.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid hook attr '{attr}'. Expected '<module_path>.input' or '<module_path>.output'."
            )
        return ".".join(parts[:-1]), parts[-1]

    def _convert_activation(self, act: torch.Tensor) -> torch.Tensor:
        out = act.detach()
        if self.flatten_tokens and out.ndim >= 3:
            out = out.reshape(-1, out.shape[-1])
        return out

    def clear_cache(self) -> None:
        self._cache = {i: defaultdict(dict) for i in self.layer_indices}

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def close(self) -> None:
        self.remove_hooks()

    def __enter__(self) -> "ActivationLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def run(self, **model_inputs) -> tuple[torch.Tensor, ...]:
        """Run the model and retrieve activations from the hooked points.
        
        Returns a tuple of tensors of shape (n_layers, ...), one tensor for each
        hook point in self.per_layer_attrs.
        """
        with torch.inference_mode():
            self.model(**model_inputs)

        saved_acts: list[torch.Tensor] = []
        for component_attr, port in self._parsed_attrs:
            per_layer = [self._cache[i][component_attr][port] for i in self.layer_indices]
            saved_acts.append(torch.stack(per_layer, dim=0))

        self.clear_cache()
        return tuple(saved_acts)


TorchActivationWrapper = ActivationLoader
