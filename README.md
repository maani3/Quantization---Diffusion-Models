# Quantizing Diffusion Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation of advanced quantization techniques (SmoothQuant and AWQ) for Stable Diffusion models. This repository enables efficient model compression while maintaining high-quality image generation across SD 1.5, SDXL, and SD3.5 architectures.

## 🌟 Features

- **Multiple Quantization Methods**: 
  - SmoothQuant implementation for diffusion models
  - AWQ (Activation-aware Weight Quantization) support
  - RTN for both Activations and Weights
- **Wide Model Support**: 
  - Stable Diffusion 1.5
  - Stable Diffusion XL (SDXL)
  - Stable Diffusion 3.5 (SD3.5)
- **HuggingFace Integration**: Seamless integration with HuggingFace model hub
- **Performance Benchmarks**: Comprehensive LPIPS evaluation results included

## 📊 Results

Our quantization implementations achieve significant model compression with minimal quality degradation:

### W8 Quantization Results (LPIPS - Lower is Better)

| Configuration | LPIPS Score |
|--------------|-------------|
| W8_QuantX | 0.0534 |
| W8_lin_QuantX | 0.0523 |
| **W8_TorchAO** | **0.0682** |
| W8_QuantO | 0.0902 |

### W4 Quantization Results (LPIPS - Lower is Better)

| Configuration | LPIPS Score |
|--------------|-------------|
| **W4_TorchAO** | **0.3274** |
| W4A16_Smooth_QuantX | 0.3416 |
| W4_lin_Smooth_QuantX | 0.3357 |
| W4_lin_QuantX | 0.3380 |
| W4_QuantX | 0.5713 |

*Lower LPIPS scores indicate better perceptual similarity to the original images.*

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU
- Git

### Setup Instructions

1. **Clone AutoAWQ Repository**
   ```bash
   git clone https://github.com/casper-hansen/AutoAWQ.git
   cd AutoAWQ
   ```

2. **Install AutoAWQ**
   ```bash
   pip install -e .
   ```

3. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   *Note: You may need to reinstall `flash_attn` if dependency conflicts occur.*

4. **Add LMMMS-eval Submodule**
   ```bash
   git submodule add https://github.com/EvolvingLMMs-Lab/lmms-eval.git
   git submodule update --init --recursive
   ```

5. **Configure Environment Variables**
   
   Add your authentication tokens to `.studiorc` or export them:
   ```bash
   export HF_TOKEN="your_huggingface_token"
   export OPENAI_API_KEY="your_openai_token"
   ```

## 💻 Usage

### Basic Quantization Example

```python
from quantize_diffusion import SmoothQuant, AWQQuant
from diffusers import StableDiffusionPipeline

# Load base model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Apply SmoothQuant
quantizer = SmoothQuant(model=pipe.unet, bits=8)
quantized_model = quantizer.quantize()

# Generate images with quantized model
pipe.unet = quantized_model
image = pipe("A beautiful sunset over mountains").images[0]
```

### AWQ Quantization

```python
# Apply AWQ quantization
awq_quantizer = AWQQuant(model=pipe.unet, bits=4)
quantized_model = awq_quantizer.quantize()
```

## 📁 Project Structure

```
Quantizing-Diffusion-Models/
├── quantization/
│   ├── smooth_quant.py
│   ├── awq_quant.py
│   └── utils.py
├── evaluation/
│   ├── lpips_eval.py
│   └── metrics.py
├── models/
│   ├── sd15.py
│   ├── sdxl.py
│   └── sd35.py
├── examples/
│   └── basic_usage.py
├── results/
│   ├── w4_results.png
│   └── w8_results.png
└── requirements.txt
```

## 🔬 Evaluation Metrics

This implementation uses LPIPS (Learned Perceptual Image Patch Similarity) as the primary evaluation metric, which better correlates with human perception of image quality compared to traditional metrics like PSNR or SSIM.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{quantizing-diffusion-models,
  author = {Muhammad Mahasin Irfan},
  title = {Quantizing Diffusion Models: SmoothQuant and AWQ Implementation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/maani3/Quantizing-Diffusion-Models}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) for the quantization framework
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) for the diffusion models
- [LMMMS-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for evaluation tools

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact mahasin.irfan@gmail.com

---

**Note**: This is an active research project. Results and implementations may be updated as we continue to improve the quantization techniques.
