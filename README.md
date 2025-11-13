# 🚀 Llama-cpp-python RTX 3090 Optimization Script

**Run Qwen3-32B at 96% GPU utilization with only 4GB RAM** — Hyper-optimized inference setup for maximum performance.

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12.3+](https://img.shields.io/badge/python-3.12.3-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.9](https://img.shields.io/badge/CUDA-12.9-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![NVIDIA-SMI 575.57.08](https://img.shields.io/badge/DRIVER-575.57.08-green.svg)](https://www.nvidia.com/en-us/drivers)

---

## 🙏 Credits

This project builds upon the excellent work by [@drakerfire98](https://github.com/drakerfire98/gpu-only-mode-guide) with GPU-only mode guide (https://github.com/drakerfire98/gpu-only-mode-guide). I've changed the Direct CUDA method, reorganized the parameters to my needs, and pushed RAM and VRAM optimization to the maximum.

---

## 📊 Performance Stats

- **Speed**: 31-35 tokens/second (average)
- **GPU Utilization**: 96%+ 
- **Context Window**: 7,168 tokens (~5,300 words)
- **System RAM Usage**: 4GB (yes, really!)
- **VRAM Usage**: 24GB (fully optimized)
- **Model**: hf.co/unsloth/Qwen3-32B-GGUF:Q5_K_M

---

## 🛠️ Hardware Requirements

- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **RAM**: 4GB (recommended), 8GB comfortable
- **CUDA**: 12.9
- **NVIDIA Driver**: 575.57.08
- **OS**: Ubuntu 24.04 (tested) or similar Linux distro
---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## ⭐ Show Your Support

If this helped you optimize your LLM setup, give it a star! It helps others discover these optimizations.

---

**Built with 💪 and a lot of VRAM optimization**
