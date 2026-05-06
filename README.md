# PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models

[![Weights](https://img.shields.io/badge/🤗-Weights-yellow)](https://huggingface.co/nvidia/personaplex-7b-v1)
[![Paper](https://img.shields.io/badge/📄-Paper-blue)](https://arxiv.org/abs/2602.06053)
[![Demo](https://img.shields.io/badge/🎮-Demo-green)](https://research.nvidia.com/labs/adlr/personaplex/)
[![Discord](https://img.shields.io/badge/Discord-Join-purple?logo=discord)](https://discord.gg/5jAXrrbwRb)

PersonaPlex is a real-time, full-duplex speech-to-speech conversational model that enables persona control through text-based role prompts and audio-based voice conditioning. Trained on a combination of synthetic and real conversations, it produces natural, low-latency spoken interactions with a consistent persona. PersonaPlex is based on the [Moshi](https://arxiv.org/abs/2410.00037) architecture and weights.

<p align="center">
  <img src="assets/architecture_diagram.png" alt="PersonaPlex Model Architecture">
  <br>
  <em>PersonaPlex Architecture</em>
</p>

## Usage

### Prerequisites

Install the [Opus audio codec](https://github.com/xiph/opus) development library:
```bash
# Ubuntu/Debian
sudo apt install libopus-dev

# Fedora/RHEL
sudo dnf install opus-devel

# macOS (via Homebrew)
brew install opus
```

### Installation

Download this repository and install with:
```bash
pip install moshi/.
```

Extra step for Blackwell based GPUs as suggested in (See https://github.com/NVIDIA/personaplex/issues/2):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```


### Accept Model License
Log in to your Huggingface account and accept the PersonaPlex model license [here](https://huggingface.co/nvidia/personaplex-7b-v1). <br>
Then set up your Huggingface authentication:
```bash
export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
```

### Launch Server

Launch server for live interaction (temporary SSL certs for https):
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
```

**CPU Offload:** If your GPU has insufficient memory, use the `--cpu-offload` flag to offload model layers to CPU. This requires the `accelerate` package (`pip install accelerate`):
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --cpu-offload
```

Access the Web UI from a browser at `localhost:8998` if running locally, otherwise look for the access link printed by the script:
```
Access the Web UI directly at https://11.54.401.33:8998
```

### Offline Evaluation

For offline evaluation use the offline script that streams in an input wav file and produces an output wav file from the captured output stream. The output file will be the same duration as the input file.

Add `--cpu-offload` to any command below if your GPU has insufficient memory (requires `accelerate` package). Or install cpu-only PyTorch for offline evaluation on pure CPU.

**Assistant example:**
```bash
HF_TOKEN=<TOKEN> \
python -m moshi.offline \
  --voice-prompt "NATF2.pt" \
  --input-wav "asset
```
