"""Gradio web interface for SHARP model.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import gc
import glob
import logging
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils import color_space as cs_utils
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians

logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger(__name__)

# Suppress verbose logs
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("sharp").setLevel(logging.WARNING)
logging.getLogger("__main__").setLevel(logging.WARNING)

# Suppress CUDA warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
MODEL_PATH = Path(__file__).parent / "models" / "sharp_model.pt"

# Global model cache
_model_cache = {}


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model(device: str):
    """Load the SHARP model with optimizations."""
    if "model" not in _model_cache:
        # Download model if not exists
        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            print(f"İndiriliyor: {DEFAULT_MODEL_URL} → {MODEL_PATH}")
            state_dict = torch.hub.load_state_dict_from_url(
                DEFAULT_MODEL_URL, 
                model_dir=str(MODEL_PATH.parent),
                file_name=MODEL_PATH.name,
                progress=True
            )
            print(f"✅ Model kaydedildi: {MODEL_PATH}")
        else:
            print(f"Model yükleniyor: {MODEL_PATH}")
            state_dict = torch.load(MODEL_PATH, map_location="cpu")
        
        gaussian_predictor = create_predictor(PredictorParams())
        gaussian_predictor.load_state_dict(state_dict)
        gaussian_predictor.eval()
        
        # Apply optimizations based on device
        if device == "cuda":
            # channels_last for faster conv2d
            gaussian_predictor = gaussian_predictor.to(device, memory_format=torch.channels_last)
            
            # Use half precision for faster inference on CUDA
            gaussian_predictor = gaussian_predictor.half()
            
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Set to inference mode
            torch.set_float32_matmul_precision('high')
        else:
            gaussian_predictor.to(device)
        
        _model_cache["model"] = gaussian_predictor
        _model_cache["device"] = device
        _model_cache["use_fp16"] = (device == "cuda")
    
    return _model_cache["model"], _model_cache["device"]


@torch.no_grad()
def predict_image(
    predictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    use_fp16: bool = False,
) -> Gaussians3D:
    """Predict Gaussians from an image with optimizations."""
    internal_shape = (1536, 1536)  # SHARP'ın orijinal ve stabil çözünürlüğü

    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    
    # Use FP16 if enabled
    if use_fp16:
        image_pt = image_pt.half()
    
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)
    
    if use_fp16:
        disparity_factor = disparity_factor.half()

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
        antialias=False  # Antialiasing devre dışı = hız
    )

    # Predict Gaussians in the NDC space.
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)
    
    # Convert back to float32 for postprocessing (Gaussians3D is a NamedTuple)
    if use_fp16:
        gaussians_ndc = Gaussians3D(
            mean_vectors=gaussians_ndc.mean_vectors.float(),
            singular_values=gaussians_ndc.singular_values.float(),
            quaternions=gaussians_ndc.quaternions.float(),
            colors=gaussians_ndc.colors.float(),
            opacities=gaussians_ndc.opacities.float(),
        )
    
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metrics space.
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians


def convert_rgb_to_spherical_harmonics(colors: torch.Tensor) -> torch.Tensor:
    """Convert RGB colors to spherical harmonics representation."""
    return (colors - 0.5) / 0.28209479177387814


def save_standard_ply(gaussians: Gaussians3D, path: Path) -> None:
    """Save Gaussians to a standard PLY file compatible with most 3D viewers."""
    
    def _inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(tensor / (1.0 - tensor))
    
    xyz = gaussians.mean_vectors.flatten(0, 1).clone()
    scale_logits = torch.log(gaussians.singular_values).flatten(0, 1)
    quaternions = gaussians.quaternions.flatten(0, 1)
    colors = convert_rgb_to_spherical_harmonics(
        cs_utils.linearRGB2sRGB(gaussians.colors.flatten(0, 1))
    )
    opacity_logits = _inverse_sigmoid(gaussians.opacities).flatten(0, 1).unsqueeze(-1)
    
    # Merkeze al ve ölçeklendir
    center = xyz.mean(dim=0)
    xyz = xyz - center
    
    # Ölçeklendirme faktörünü hesapla
    max_dist = torch.sqrt((xyz ** 2).sum(dim=1)).max()
    if max_dist > 0:
        scale_factor = 1.5 / max_dist  # [-1.5, 1.5] aralığına sığdır
        xyz = xyz * scale_factor
        # Gaussian scale'leri de ayarla
        scale_logits = scale_logits + torch.log(scale_factor.clone().detach())
    
    attributes = torch.cat(
        (
            xyz,
            colors,
            opacity_logits,
            scale_logits,
            quaternions,
        ),
        dim=1,
    )
    
    # Standard 3DGS PLY format
    dtype_full = [
        (attribute, "f4")
        for attribute in ["x", "y", "z"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    ]
    
    num_gaussians = len(xyz)
    elements = np.empty(num_gaussians, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes.detach().cpu().numpy()))
    vertex_elements = PlyElement.describe(elements, "vertex")
    
    # Only include vertex data for maximum compatibility
    plydata = PlyData([vertex_elements])
    plydata.write(path)


def clear_memory():
    """Clear VRAM, system memory and temporary files (keeps model loaded)."""
    try:
        deleted_files = 0
        freed_space = 0
        
        # Clear temporary PLY files
        temp_dir = tempfile.gettempdir()
        patterns = [
            os.path.join(temp_dir, "tmp*.ply"),
            os.path.join(temp_dir, "*_preview.ply"),
            os.path.join(temp_dir, "*_medium.ply"),
            os.path.join(temp_dir, "*_final.ply")
        ]
        
        for pattern in patterns:
            for filepath in glob.glob(pattern):
                try:
                    file_size = os.path.getsize(filepath)
                    os.remove(filepath)
                    deleted_files += 1
                    freed_space += file_size
                except:
                    pass
        
        # Clear CUDA cache (keeps model in memory)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Python garbage collection
        gc.collect()
        
        freed_mb = freed_space / (1024 * 1024)
        if deleted_files > 0:
            return f"\u2705 Temizlendi: {deleted_files} dosya ({freed_mb:.1f} MB) + VRAM bo\u015falt\u0131ld\u0131"
        else:
            return "\u2705 VRAM ve bellek temizlendi!"
    except Exception as e:
        return f"\u26a0\ufe0f Hata: {str(e)}"


def process_image(image):
    """Process uploaded image and generate 3D Gaussian Splatting file."""
    if image is None:
        yield None, None, "Lütfen bir görsel yükleyin."
        return
    
    try:
        import time
        start_time = time.time()
        
        device = get_device()
        yield None, None, "⚙️ Başlatılıyor..."
        
        # Load model
        predictor, device = load_model(device)
        use_fp16 = _model_cache.get("use_fp16", False)
        yield None, None, f"📦 Model yüklendi (FP16: {use_fp16})"
        
        # Load image and get focal length
        if hasattr(image, 'convert'):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = image
        
        height, width = image_np.shape[:2]
        f_px = max(width, height) * 1.2
        yield None, None, f"🖼️ Görsel hazırlandı ({width}x{height})"
        
        # Predict Gaussians
        yield None, None, "🧠 AI işliyor..."
        gaussians = predict_image(predictor, image_np, f_px, torch.device(device), use_fp16)
        yield None, None, "✨ 3D model oluşturuldu"
        
        # Save to file
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            output_path = Path(tmp_file.name)
        
        yield None, None, "💾 Kaydediliyor..."
        save_standard_ply(gaussians, output_path)
        
        total_time = time.time() - start_time
        num_gaussians = len(gaussians.mean_vectors.flatten(0, 1))
        
        status = f"✅ Tamamlandı • {total_time:.1f}s • {num_gaussians:,} nokta"
        
        # Clear VRAM and memory
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        yield str(output_path), str(output_path), status
        
    except Exception as e:
        # Clear memory on error too
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        yield None, None, f"❌ Hata: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="SHARP - 3D Generator") as demo:
    gr.Markdown("""
    # ✨ SHARP - 3D Generator
    Tek fotoğraftan 3D model oluşturun
    """)
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("## 📷 Giriş")
            input_image = gr.Image(
                label="Fotoğraf Yükle",
                type="pil",
                height=400
            )
            process_btn = gr.Button("✨ 3D Oluştur", variant="primary", size="lg")
            clear_btn = gr.Button("🧹 Bellek Temizle", variant="secondary", size="sm")
            gr.Markdown("**Durum:**")
            output_status = gr.Textbox(
                lines=2,
                show_label=False,
                placeholder="İşlem durumu..."
            )
        
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("## 🎬 Çıktı")
            model_3d = gr.Model3D(
                label="3D Model",
                height=400,
                clear_color=[0.1, 0.1, 0.1, 1.0],
                camera_position=(0, 0, 2.5),
                zoom_speed=1.0
            )
            output_file = gr.File(
                label="PLY Dosyası",
                file_count="single"
            )
    
    # Connect the button
    process_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[model_3d, output_file, output_status]
    )
    
    clear_btn.click(
        fn=clear_memory,
        inputs=[],
        outputs=[output_status]
    )


if __name__ == "__main__":
    device = get_device()
    
    print(f"SHARP - Device: {device.upper()}")
    if device == "cuda":
        import torch
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"URL: http://127.0.0.1:7870\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7870,
        share=False,
        inbrowser=False,
        quiet=False,
        theme=gr.themes.Soft(),
        css="""
            .gradio-container {
                max-width: 1600px !important;
                width: 100% !important;
                margin: auto !important;
            }
            footer {display: none !important;}
            
            /* Yan yana zorla */
            .gradio-row {
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: nowrap !important;
                gap: 1rem !important;
                align-items: center !important;
                justify-content: center !important;
            }
            
            .gradio-column {
                flex: 1 !important;
                min-width: 280px !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
            }
            
            /* 3D Model merkezde */
            .model-3d {
                margin: 0 auto !important;
                display: block !important;
            }
            
            /* Sadece mobilde alt alta */
            @media (max-width: 768px) {
                .gradio-row {
                    flex-direction: column !important;
                }
            }
        """
    )
