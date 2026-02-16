# inference-clt

Simple inference package for loading pretrained CLT models and running:

- feature extraction (`encode`)
- activation reconstruction (`decode`)
- one-step reconstruction (`reconstruct`)

## Installation

Install directly from the git repo with

```bash
pip install git+https://github.com/BlueLightAI/inference-clt
```

You can also clone the repository and install in editable mode:

```bash
git clone https://github.com/BlueLightAI/inference-clt
cd inference_clt
pip install -e .
```

## Quickstart

```python
import torch
from transformers import AutoModel, AutoTokenizer
from inference_clt import ActivationLoader, InferenceCLT

# Load CLT local checkpoint directory
clt = InferenceCLT.load_from_disk("/path/to/checkpoint", device="cpu")

# Load the base model
model = AutoModel.from_pretrained(clt.base_model_name)
tokenizer = AutoTokenizer.from_pretrained(clt.base_model_name)

# Create an activation loader
act_loader = ActivationLoader.from_inference_clt(model, clt)

# Get input activations: (n_layers, tokens, d_in)
prompt = "Let's get some CLT feature activations!"
tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(model.device)
mlp_in, mlp_out = act_loader.run(**tokenized_prompt)

# Feature activations: (n_layers, tokens, d_latent)
f = clt.encode(mlp_in)

# Reconstructed outputs: (n_layers, tokens, d_out)
reconst_mlp_out = clt.decode(f)

# Or get both in one step
f2, reconst_mlp_out2 = clt.reconstruct(mlp_in)
```

## Loading from local checkpoints

`InferenceCLT.load_from_disk(path, ...)` auto-detects supported checkpoint layouts:

1. BluelightAI native CLT formats
   - `single_file` format (`clt.safetensors` + `model_config.json`)
   - `layer_weights` format (`manifest.json` + per-layer safetensors + `model_config.json`)

2. `circuit-tracer` format
   - `config.yaml`
   - `W_enc_*.safetensors`
   - `W_dec_*.safetensors`

Example:

```python
from inference_clt import InferenceCLT

model = InferenceCLT.load_from_disk(
    "/path/to/model_dir",
    device="cuda:0",
)
```

## Loading directly from Hugging Face

Use `InferenceCLT.load_from_huggingface(...)` to download and load in one step.

Note: `load_from_huggingface()` currently downloads a full repo snapshot via `huggingface_hub.snapshot_download`.

```python
from inference_clt import InferenceCLT

model = InferenceCLT.load_from_huggingface(
    repo_id="bluelightai/clt-qwen3-0.6b-base-20k",
    revision="main",          # optional
    device="cpu",
)
```

## Encoder-only loading

If you only need features (not reconstruction), you can skip loading decoder
weights. This can be a substantial memory savings:

```python
clt = InferenceCLT.load_from_disk(
    "/path/to/model_dir",
    device="cuda:0",
    encoder_only=True,
)

features = clt.encode(mlp_in)
# clt.decode(...) and clt.reconstruct(...) will raise RuntimeError
```

## Runtime encoder layer subset

You can encode only a subset of CLT layers by passing `layer_indices` to `encode()`.
When you do this, `input_activations` must contain only those layers in the same order:

```python
subset = [0, 2, 5]
subset_acts = mlp_in[subset]                  # shape: (len(subset), tokens, d_in)
subset_features = clt.encode(subset_acts, layer_indices=subset)
```

## Tensor shapes

This package uses layer-first tensors:

- input activations: `(n_layers, batch, d_in)`
- features: `(n_layers, batch, d_latent)`
- reconstructed outputs: `(n_layers, batch, d_out)`

## Useful attributes

The loaded model exposes metadata and dimensions:

- `model.d_in`, `model.d_out`, `model.d_latent`
- `model.n_training_layers`
- `model.layer_indices` (if subset-trained)
- `model.base_model_name`
- `model.input_attr`, `model.output_attr`
