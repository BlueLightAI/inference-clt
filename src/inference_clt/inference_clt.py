from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file
from torch import nn


_STR_TO_DTYPE: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


_LAYER_KEY_TO_STATE_KEY = {
    "W_enc": "encoder.W",
    "b_enc": "encoder.b",
    "norm_factor_in": "norm_factor_in",
    "norm_factor_out": "norm_factor_out",
    "log_threshold": "activation.log_threshold",
    "b_dec": "decoder.b",
    "W_dec_shared": "decoder.decoder.W",
}


def _saved_key_to_state_dict_key(
    key: str,
    layout_type: str,
    layer_index_map: dict[int, int] | None = None,
) -> str:
    if key in _LAYER_KEY_TO_STATE_KEY:
        return _LAYER_KEY_TO_STATE_KEY[key]

    if layout_type in ("input_layer", "output_layer"):
        decoders_name = "decoders"
    else:
        decoders_name = "layer_connections"

    try:
        param, _, layer = key.split("_")
        layer_idx = int(layer)
    except Exception as exc:
        raise ValueError(f"Unknown key {key}") from exc

    if layer_index_map is not None:
        layer_idx = layer_index_map[layer_idx]

    return f"decoder.{decoders_name}.{layer_idx}.{param}"


def _load_state_dict_from_checkpoint(
    path: Path, device: str | torch.device
) -> dict[str, torch.Tensor]:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return load_file(path / "clt.safetensors", device=str(device))

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if manifest.get("format") != "layer_weights":
        raise ValueError(f"Unsupported format: {manifest.get('format')}")

    decoder_type = manifest["decoder_type"]
    layer_index_map = None
    if manifest.get("layer_indices") is not None:
        layer_index_map = {
            original_idx: i
            for i, original_idx in enumerate(manifest["layer_indices"])
        }

    state_dict: dict[str, torch.Tensor] = {}
    for filename in manifest["files"]:
        file_path = path / filename
        if not file_path.exists():
            continue
        saved_part = load_file(file_path, device=str(device))
        for key, value in saved_part.items():
            mapped_key = _saved_key_to_state_dict_key(
                key,
                decoder_type,
                layer_index_map,
            )
            state_dict[mapped_key] = value

    return state_dict


def _detect_checkpoint_format(path: Path) -> Literal["clt", "circuit_tracer"]:
    if (path / "model_config.json").exists():
        return "clt"
    if (path / "config.yaml").exists() and (path / "W_enc_0.safetensors").exists():
        return "circuit_tracer"
    raise FileNotFoundError(
        f"Could not detect weight format in {path}. "
        "Expected either CLT format (model_config.json) or circuit-tracer format (config.yaml + W_enc_0.safetensors)."
    )


def _load_circuit_tracer_checkpoint(
    path: Path,
    device: str | torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    enc_files = sorted(path.glob("W_enc_*.safetensors"), key=lambda p: int(p.stem.split("_")[-1]))
    dec_files = sorted(path.glob("W_dec_*.safetensors"), key=lambda p: int(p.stem.split("_")[-1]))

    if not enc_files or not dec_files:
        raise FileNotFoundError(
            f"Missing circuit-tracer weight files in {path}. "
            "Expected W_enc_*.safetensors and W_dec_*.safetensors."
        )

    n_layers = len(enc_files)
    if len(dec_files) != n_layers:
        raise ValueError(
            f"Expected {n_layers} decoder files, found {len(dec_files)}"
        )

    W_enc_layers: list[torch.Tensor] = []
    b_enc_layers: list[torch.Tensor] = []
    b_dec_layers: list[torch.Tensor] = []
    thresholds: list[torch.Tensor] = []
    has_threshold = True

    for layer_idx, enc_file in enumerate(enc_files):
        layer_data = load_file(enc_file, device=str(device))
        W_enc_key = f"W_enc_{layer_idx}"
        b_enc_key = f"b_enc_{layer_idx}"
        b_dec_key = f"b_dec_{layer_idx}"
        threshold_key = f"threshold_{layer_idx}"

        if W_enc_key not in layer_data or b_enc_key not in layer_data or b_dec_key not in layer_data:
            raise ValueError(f"Malformed encoder file {enc_file.name}")

        W_enc_layers.append(layer_data[W_enc_key].transpose(0, 1).contiguous())
        b_enc_layers.append(layer_data[b_enc_key])
        b_dec_layers.append(layer_data[b_dec_key])

        if threshold_key in layer_data:
            thresholds.append(layer_data[threshold_key])
        else:
            has_threshold = False

    W_enc = torch.stack(W_enc_layers, dim=0)
    b_enc = torch.stack(b_enc_layers, dim=0)
    b_dec = torch.stack(b_dec_layers, dim=0)

    d_in = int(W_enc.shape[1])
    d_latent = int(W_enc.shape[2])

    state_dict: dict[str, torch.Tensor] = {
        "encoder.W": W_enc,
        "encoder.b": b_enc,
        "decoder.b": b_dec,
        "norm_factor_in": torch.ones(n_layers, 1, 1, device=W_enc.device, dtype=W_enc.dtype),
        "norm_factor_out": torch.ones(n_layers, 1, 1, device=W_enc.device, dtype=W_enc.dtype),
    }

    for layer_idx, dec_file in enumerate(dec_files):
        layer_data = load_file(dec_file, device=str(device))
        W_dec_key = f"W_dec_{layer_idx}"
        if W_dec_key not in layer_data:
            raise ValueError(f"Malformed decoder file {dec_file.name}")
        W_dec = layer_data[W_dec_key]
        if W_dec.shape[0] != d_latent:
            raise ValueError(f"Decoder latent mismatch in {dec_file.name}")
        # circuit-tracer saves (d_latent, n_layers-i, d_in); CLT input-layer expects (n_layers-i, d_latent, d_in)
        state_dict[f"decoder.decoders.{layer_idx}.W"] = W_dec.permute(1, 0, 2).contiguous()

    if has_threshold:
        threshold = torch.stack(thresholds, dim=0).clamp_min(1e-12)
        state_dict["activation.log_threshold"] = torch.log(threshold)

    config_yaml_path = path / "config.yaml"
    if config_yaml_path.exists():
        try:
            import yaml

            with open(config_yaml_path, "r") as f:
                cfg_yaml = yaml.safe_load(f) or {}
        except Exception:
            cfg_yaml = {}
    else:
        cfg_yaml = {}

    config_dict: dict[str, Any] = {
        "config_version": 1,
        "n_layers": n_layers,
        "d_in": d_in,
        "d_latent": d_latent,
        "activation": "jumprelu" if has_threshold else "relu",
        "k": 0,
        "decoder_layout": "input_layer",
        "base_model_name": cfg_yaml.get("model_name"),
        "input_attr": cfg_yaml.get("feature_input_hook"),
        "output_attr": cfg_yaml.get("feature_output_hook"),
    }

    return config_dict, state_dict


class InferenceCLT(nn.Module):
    """Inference-only CLT wrapper for saved weights.

    This class supports:
    - loading CLT weights in BluelightAI or circuit-tracer format
    - encoding activations to feature values
    - decoding feature values to reconstructed activations
    """

    def __init__(self, config_dict: dict[str, Any], state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.raw_config = config_dict

        self.threshold: nn.Parameter | None = None
        self.decoder_bias: nn.Parameter
        self.decoder_weights = nn.ParameterList()

        config_version = int(config_dict.get("config_version", 1))
        self.config_version = config_version

        if config_version == 2:
            self._init_from_v2_config(config_dict)
        else:
            self._init_from_v1_config(config_dict)

        self.n_training_layers = (
            len(self.layer_indices)
            if self.layer_indices is not None
            else self.base_model_n_layers
        )

        self.W_enc = nn.Parameter(state_dict["encoder.W"], requires_grad=False)
        self.b_enc = nn.Parameter(state_dict["encoder.b"], requires_grad=False)

        if "norm_factor_in" not in state_dict or "norm_factor_out" not in state_dict:
            raise ValueError("Saved weights are missing normalization factors")

        self.norm_factor_in = nn.Parameter(
            state_dict["norm_factor_in"].reshape((-1, 1, 1)),
            requires_grad=False,
        )
        self.norm_factor_out = nn.Parameter(
            state_dict["norm_factor_out"].reshape((-1, 1, 1)),
            requires_grad=False,
        )

        if self.activation_type == "jumprelu":
            if "activation.log_threshold" not in state_dict:
                raise ValueError("JumpReLU model missing activation.log_threshold")
            self.threshold = nn.Parameter(
                torch.exp(state_dict["activation.log_threshold"]),
                requires_grad=False,
            )

        self._load_decoder_weights(state_dict)

    def _init_from_v2_config(self, config_dict: dict[str, Any]) -> None:
        base = config_dict["base_model"]
        hook_points = base["hook_points"]
        arch = config_dict["architecture"]
        activation = arch["activation"]
        decoder = arch["decoder"]

        self.base_model_name = base["name"]
        self.base_model_n_layers = int(base["n_layers"])
        self.layer_attr = hook_points.get("layer_attr", "layers")
        self.input_attr = hook_points["input"]["attr"]
        self.output_attr = hook_points["output"]["attr"]
        self.layer_indices = hook_points.get("layer_indices")

        self.d_in = int(hook_points["input"]["dim"])
        self.d_out = int(hook_points["output"]["dim"])
        self.d_latent = int(arch["d_latent"])

        self.activation_type = activation["type"]
        self.k = int(activation.get("k", 0))

        self.decoder_layout = decoder.get("layout", "output_layer")

    def _init_from_v1_config(self, config_dict: dict[str, Any]) -> None:
        self.base_model_name = config_dict.get("base_model_name")
        self.base_model_n_layers = int(config_dict["n_layers"])
        self.layer_attr = "layers"
        self.input_attr = config_dict.get("input_attr")
        self.output_attr = config_dict.get("output_attr")
        self.layer_indices = config_dict.get("layer_indices")

        self.d_in = int(config_dict["d_in"])
        self.d_out = int(config_dict["d_in"])
        self.d_latent = int(config_dict["d_latent"])

        self.activation_type = config_dict["activation"]
        self.k = int(config_dict.get("k", 0))

        self.decoder_layout = config_dict.get("decoder_layout", "output_layer")

    def _load_decoder_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        n_layers = self.n_training_layers

        source_layout = self.decoder_layout
        if source_layout == "input_layer":
            for layer_idx in range(n_layers):
                w_key = f"decoder.decoders.{layer_idx}.W"
                self.decoder_weights.append(
                    nn.Parameter(state_dict[w_key], requires_grad=False)
                )
        elif source_layout == "output_layer":
            for src_layer in range(n_layers):
                rows = []
                for dst_layer in range(src_layer, n_layers):
                    out_W = state_dict[f"decoder.decoders.{dst_layer}.W"]
                    rows.append(out_W[src_layer : src_layer + 1])
                self.decoder_weights.append(
                    nn.Parameter(torch.cat(rows, dim=0), requires_grad=False)
                )
        else:
            raise ValueError(f"Unsupported decoder layout: {source_layout}")

        self.source_decoder_layout = source_layout
        self.decoder_layout = "input_layer"

        if "decoder.b" in state_dict:
            self.decoder_bias = nn.Parameter(
                state_dict["decoder.b"],
                requires_grad=False,
            )
        else:
            self.decoder_bias = nn.Parameter(
                torch.zeros((n_layers, self.d_out), dtype=out_W.dtype), requires_grad=False,
            )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "base_model_name": self.base_model_name,
            "base_model_n_layers": self.base_model_n_layers,
            "layer_indices": self.layer_indices,
            "layer_attr": self.layer_attr,
            "input_attr": self.input_attr,
            "output_attr": self.output_attr,
            "d_in": self.d_in,
            "d_out": self.d_out,
            "d_latent": self.d_latent,
            "activation": self.activation_type,
            "decoder_layout": self.decoder_layout,
        }

    def _apply_activation(self, pre_activations: torch.Tensor) -> torch.Tensor:
        if self.activation_type == "relu":
            return pre_activations.relu()

        if self.activation_type == "jumprelu":
            if self.threshold is None:
                raise ValueError("JumpReLU model missing threshold")
            threshold = self.threshold.unsqueeze(1).to(pre_activations.dtype)
            return pre_activations * (pre_activations > threshold)

        if self.activation_type == "topk":
            values, indices = pre_activations.relu().topk(self.k, dim=-1)
            return torch.zeros_like(pre_activations).scatter(-1, indices, values)

        if self.activation_type == "topk_all":
            flat = pre_activations.relu().flatten(-2, -1)
            values, indices = flat.topk(self.k, dim=-1)
            return torch.zeros_like(flat).scatter(-1, indices, values).reshape(pre_activations.shape)

        raise ValueError(f"Unsupported activation type: {self.activation_type}")

    def encode(self, input_activations: torch.Tensor) -> torch.Tensor:
        """Map input activations (n_layers, batch, d_in) to feature activations."""
        normalized = input_activations / self.norm_factor_in
        pre = torch.bmm(normalized, self.W_enc) + self.b_enc.unsqueeze(1).to(normalized.dtype)
        return self._apply_activation(pre)

    def _decode_input_layer(self, features: torch.Tensor) -> torch.Tensor:
        if len(self.decoder_weights) != self.n_training_layers:
            raise ValueError("decoder_weights are not initialized correctly")

        out = torch.matmul(features[0], self.decoder_weights[0])
        for layer_idx in range(1, self.n_training_layers):
            decoded = torch.matmul(features[layer_idx], self.decoder_weights[layer_idx])
            out[layer_idx:] += decoded

        out = out + self.decoder_bias.unsqueeze(1).to(out.dtype)
        return out

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Map feature activations (n_layers, batch, d_latent) to output activations."""
        if features.shape[0] != self.n_training_layers:
            raise ValueError(
                f"Expected features with first dim {self.n_training_layers}, got {features.shape[0]}"
            )
        decoded = self._decode_input_layer(features)

        return decoded * self.norm_factor_out

    def reconstruct(self, input_activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features, reconstructed_outputs) from input activations."""
        features = self.encode(input_activations)
        reconstructed = self.decode(features)
        return features, reconstructed

    @classmethod
    def load_from_disk(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "InferenceCLT":
        path = Path(path)

        checkpoint_format = _detect_checkpoint_format(path)
        if checkpoint_format == "clt":
            with open(path / "model_config.json", "r") as f:
                config_dict = json.load(f)
            saved_dtype = _STR_TO_DTYPE[config_dict.get("dtype", "float32")]
            state_dict = _load_state_dict_from_checkpoint(path, device=device)
        else:
            config_dict, state_dict = _load_circuit_tracer_checkpoint(path, device=device)
            saved_dtype = next(iter(state_dict.values())).dtype

        model = cls(config_dict=config_dict, state_dict=state_dict)
        model = model.to(device=device, dtype=dtype or saved_dtype)
        model.eval()
        return model
