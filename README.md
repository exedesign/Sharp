# Sharp Monocular View Synthesis in Less Than a Second

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://apple.github.io/ml-sharp/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10685-b31b1b.svg)](https://arxiv.org/abs/2512.10685)

> **⚠️ IMPORTANT:** This project requires **Python 3.11 or 3.12**. Python 3.13 is NOT compatible due to PyTorch limitations.
> 
> **⚠️ ÖNEMLİ:** Bu proje **Python 3.11 veya 3.12** gerektirir. Python 3.13, PyTorch kısıtlamaları nedeniyle uyumlu DEĞİLDİR.

This software project accompanies the research paper: _Sharp Monocular View Synthesis in Less Than a Second_
by _Lars Mescheder, Wei Dong, Shiwei Li, Xuyang Bai, Marcel Santos, Peiyun Hu, Bruno Lecouat, Mingmin Zhen, Amaël Delaunoy,
Tian Fang, Yanghai Tsin, Stephan Richter and Vladlen Koltun_.

![](data/teaser.jpg)

We present SHARP, an approach to photorealistic view synthesis from a single image. Given a single photograph, SHARP regresses the parameters of a 3D Gaussian representation of the depicted scene. This is done in less than a second on a standard GPU via a single feedforward pass through a neural network. The 3D Gaussian representation produced by SHARP can then be rendered in real time, yielding high-resolution photorealistic images for nearby views. The representation is metric, with absolute scale, supporting metric camera movements. Experimental results demonstrate that SHARP delivers robust zero-shot generalization across datasets. It sets a new state of the art on multiple datasets, reducing LPIPS by 25–34% and DISTS by 21–43% versus the best prior model, while lowering the synthesis time by three orders of magnitude.

---

## 🎨 Gradio Web Interface

This repository includes a **CUDA-optimized Gradio web interface** for the SHARP model (`app.py`).

![Gradio Interface](data/gradio_interface.jpg)
*User-friendly web interface for single-image to 3D model generation / Tek fotoğraftan 3D model oluşturma için kullanıcı dostu web arayüzü*

### ✨ Features

- **⚡ CUDA & FP16 Optimization**: Maximum speed on RTX GPUs
- **🎯 channels_last Memory Format**: Optimized memory layout for Conv2D operations
- **🔥 TF32 Support**: Acceleration using Tensor Cores
- **🧠 CuDNN Benchmark**: Automatic kernel selection
- **🎨 Minimal Interface**: User-friendly, clean Gradio design
- **📊 Real-time Status**: Live display of processing steps
- **💾 Automatic Model Download**: Model is automatically downloaded on first run
- **🧹 Memory Management**: Automatic cleanup of VRAM and temporary files
- **🎬 Centered 3D Viewer**: Models are automatically centered and scaled
- **📦 PLY Export**: 3D models in standard PLY format

### 🚀 Gradio Interface Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install PyTorch CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Gradio interface
python app.py
```

### 🎮 Usage

1. Open `http://127.0.0.1:7870` in your browser
2. Upload a photo from the left panel
3. Click the **"✨ Generate 3D"** button
4. View the 3D model in the right panel
5. Download the PLY file

### ⚙️ Technical Details

- **Resolution**: 1536x1536 (SHARP's original and stable resolution)
- **Inference Mode**: FP16 (half precision)
- **GPU Memory**: ~4-6 GB VRAM
- **Processing Time**: ~4-6 seconds (RTX 3070 Laptop GPU)
- **Model Size**: 2.62 GB (automatically downloaded to `models/` folder)

### 🎯 Optimizations

```python
# FP16 Inference
model = model.half()

# channels_last memory format
model = model.to(device, memory_format=torch.channels_last)

# CuDNN & TF32
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
```

---

## Getting started

We recommend to first create a python environment:

```
conda create -n sharp python=3.13
```

Afterwards, you can install the project using

```
pip install -r requirements.txt
```

To test the installation, run

```
sharp --help
```

## Using the CLI

To run prediction:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians
```

The model checkpoint will be downloaded automatically on first run and cached locally at `~/.cache/torch/hub/checkpoints/`.

Alternatively, you can download the model directly:

```
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
```

To use a manually downloaded checkpoint, specify it with the `-c` flag:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians -c sharp_2572gikvuh.pt
```

The results will be 3D gaussian splats (3DGS) in the output folder. The 3DGS `.ply` files are compatible to various public 3DGS renderers. We follow the OpenCV coordinate convention (x right, y down, z forward). The 3DGS scene center is roughly at (0, 0, +z). When dealing with 3rdparty renderers, please scale and rotate to re-center the scene accordingly.

### Rendering trajectories (CUDA GPU only)

Additionally you can render videos with a camera trajectory. While the gaussians prediction works for all CPU, CUDA, and MPS, rendering videos via the `--render` option currently requires a CUDA GPU. The gsplat renderer takes a while to initialize at the first launch.

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians --render

# Or from the intermediate gaussians:
sharp render -i /path/to/output/gaussians -o /path/to/output/renderings
```

## Evaluation

Please refer to the paper for both quantitative and qualitative evaluations.
Additionally, please check out this [qualitative examples page](https://apple.github.io/ml-sharp/) containing several video comparisons against related work.

## Citation

If you find our work useful, please cite the following paper:

```bibtex
@inproceedings{Sharp2025:arxiv,
  title      = {Sharp Monocular View Synthesis in Less Than a Second},
  author     = {Lars Mescheder and Wei Dong and Shiwei Li and Xuyang Bai and Marcel Santos and Peiyun Hu and Bruno Lecouat and Mingmin Zhen and Ama\"{e}l Delaunoy and Tian Fang and Yanghai Tsin and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2512.10685},
  year       = {2025},
  url        = {https://arxiv.org/abs/2512.10685},
}
```

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details.

## License

Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.

---

## Turkish Description / 🇹🇷 Türkçe Açıklama

### 🎨 Gradio Web Arayüzü

Bu repository'de **SHARP modeli için CUDA optimizasyonlu Gradio web arayüzü** (`app.py`) bulunmaktadır.

### ✨ Özellikler

- **⚡ CUDA & FP16 Optimizasyonu**: RTX GPU'larda maksimum hız
- **🎯 channels_last Memory Format**: Conv2D işlemleri için optimize edilmiş bellek düzeni
- **🔥 TF32 Desteği**: Tensor Core kullanımı ile hızlandırma
- **🧠 CuDNN Benchmark**: Otomatik kernel seçimi
- **🎨 Minimal Arayüz**: Kullanıcı dostu, sade Gradio tasarımı
- **📊 Real-time Durum**: İşlem adımlarının anlık görüntülenmesi
- **💾 Otomatik Model İndirme**: İlk çalıştırmada model otomatik indirilir
- **🧹 Bellek Yönetimi**: VRAM ve geçici dosyaların otomatik temizliği
- **🎬 Centered 3D Viewer**: Modeller otomatik merkezlenir ve ölçeklenir
- **📦 PLY Export**: 3D modeller standart PLY formatında

### 🚀 Gradio Arayüzü Kurulum

```bash
# 1. Virtual environment oluştur
python -m venv .venv
.venv\Scripts\activate

# 2. PyTorch CUDA 12.4 yükle
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Gradio arayüzünü başlat
python app.py
```

### 🎮 Kullanım

1. Tarayıcıda `http://127.0.0.1:7870` adresini açın
2. Sol panelden bir fotoğraf yükleyin
3. **"✨ 3D Oluştur"** butonuna basın
4. Sağ panelde 3D modeli görüntüleyin
5. PLY dosyasını indirin

### ⚙️ Teknik Detaylar

- **Çözünürlük**: 1536x1536 (SHARP'ın orijinal ve stabil çözünürlüğü)
- **Inference Modu**: FP16 (half precision)
- **GPU Memory**: ~4-6 GB VRAM
- **İşlem Süresi**: ~4-6 saniye (RTX 3070 Laptop GPU)
- **Model Boyutu**: 2.62 GB (otomatik `models/` klasörüne indirilir)

### 🎯 Optimizasyonlar

```python
# FP16 Inference
model = model.half()

# channels_last memory format
model = model.to(device, memory_format=torch.channels_last)

# CuDNN & TF32
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
```
