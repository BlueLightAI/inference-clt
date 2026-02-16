from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

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
        self._cache: dict[int, dict[int, dict[str, dict[str, torch.Tensor]]]] = {}
        self._current_call_idx = -1

        self._hooks: list[Any] = []
        self._model_pre_hook = self.model.register_forward_pre_hook(
            self._on_model_pre_forward,
            with_kwargs=True,
        )

        for layer_idx in self.layer_indices:
            layer = layers[layer_idx]
            for component_attr, port in self._parsed_attrs:
                component = _nested_getattr(layer, component_attr)

                if port == "input":

                    def hook(_module, args, _output, layer_idx=layer_idx, component_attr=component_attr):
                        act = _extract_tensor(args)
                        converted = self._convert_activation(act)
                        if self._current_call_idx >= 0 and self._current_call_idx in self._cache:
                            self._cache[self._current_call_idx][layer_idx][component_attr][
                                "input"
                            ] = converted

                    self._hooks.append(component.register_forward_hook(hook))
                elif port == "output":

                    def hook(_module, _args, output, layer_idx=layer_idx, component_attr=component_attr):
                        act = _extract_tensor(output)
                        converted = self._convert_activation(act)
                        if self._current_call_idx >= 0 and self._current_call_idx in self._cache:
                            self._cache[self._current_call_idx][layer_idx][component_attr][
                                "output"
                            ] = converted

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

    def _new_layer_cache(self) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
        return {i: defaultdict(dict) for i in self.layer_indices}

    def _on_model_pre_forward(self, _module, _args, kwargs) -> None:
        self._current_call_idx += 1
        self._cache[self._current_call_idx] = self._new_layer_cache()

    def _reset_cache(self) -> None:
        self._cache = {}
        self._current_call_idx = -1

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        if self._model_pre_hook is not None:
            self._model_pre_hook.remove()
            self._model_pre_hook = None

    def close(self) -> None:
        self.remove_hooks()

    def __enter__(self) -> "ActivationLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _assemble_single_call_capture(
        self,
        call_cache: dict[int, dict[str, dict[str, torch.Tensor]]],
    ) -> tuple[torch.Tensor, ...]:
        saved_acts: list[torch.Tensor] = []
        for component_attr, port in self._parsed_attrs:
            per_layer = [call_cache[i][component_attr][port] for i in self.layer_indices]
            saved_acts.append(torch.stack(per_layer, dim=0))
        return tuple(saved_acts)

    def run(
        self,
        return_model_output: bool = False,
        **model_inputs: Any,
    ) -> tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], Any]:
        """Run the model and retrieve activations from the hooked points.

        Args:
            return_model_output: If True, also return the model forward output.
            **model_inputs: Inputs forwarded to `model(**model_inputs)`.

        Returns a tuple of tensors of shape (n_layers, ...), one tensor for each
        hook point in self.per_layer_attrs.

        If `return_model_output=True`, returns `(model_output, activations)`.
        """
        self._reset_cache()
        with torch.inference_mode():
            start_call_idx = self._current_call_idx
            model_output = self.model(**model_inputs)

            call_ids = sorted(call_id for call_id in self._cache.keys() if call_id > start_call_idx)
        if not call_ids:
            raise RuntimeError("No activations were captured during run()")
        saved_acts = list(self._assemble_single_call_capture(self._cache[call_ids[-1]]))

        self._reset_cache()
        activations = tuple(saved_acts)
        if return_model_output:
            return model_output, activations
        return activations

    def generate_with_activations(
        self,
        generation_kwargs: dict[str, Any] | None = None,
        capture_mode: Literal["all", "prefill_only", "decode_only"] = "all",
        **model_inputs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Run model.generate() and capture activations from prefill and decode passes.

        Args:
            generation_kwargs: Optional kwargs passed through to `model.generate(...)`.
            capture_mode: One of {"all", "prefill_only", "decode_only"}.
            **model_inputs: Inputs forwarded to `model.generate(...)`.

        Returns:
            A tuple `(generation_output, captures)` where `captures` has keys:
            - `prefill`: tuple[Tensor, ...] | None
            - `decode_steps`: list[tuple[Tensor, ...]]
        """
        if not hasattr(self.model, "generate"):
            raise ValueError("Model does not expose generate(); use a generation-capable HF model")

        if capture_mode not in {"all", "prefill_only", "decode_only"}:
            raise ValueError(
                "capture_mode must be one of {'all', 'prefill_only', 'decode_only'}"
            )

        generation_kwargs = dict(generation_kwargs or {})

        self._reset_cache()

        with torch.inference_mode():
            start_call_idx = self._current_call_idx
            generation_output = self.model.generate(**model_inputs, **generation_kwargs)

            call_ids = sorted(call_id for call_id in self._cache.keys() if call_id > start_call_idx)
        if not call_ids:
            self._reset_cache()
            captures = {"prefill": None, "decode_steps": []}
            return generation_output, captures

        prefill_call_id = call_ids[0]
        prefill_capture = self._assemble_single_call_capture(self._cache[prefill_call_id])
        decode_steps: list[tuple[torch.Tensor, ...]] = []

        for call_id in call_ids[1:]:
            decode_steps.append(self._assemble_single_call_capture(self._cache[call_id]))

        if capture_mode == "prefill_only":
            captures = {"prefill": prefill_capture, "decode_steps": []}
        elif capture_mode == "decode_only":
            captures = {"prefill": None, "decode_steps": decode_steps}
        else:
            captures = {"prefill": prefill_capture, "decode_steps": decode_steps}

        self._reset_cache()

        return generation_output, captures
