import torch

print(f"PyTorch Surumu: {torch.__version__}")
print(f"CUDA Kullanilabilir: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Surumu: {torch.version.cuda}")
    print(f"GPU Sayisi: {torch.cuda.device_count()}")
    print(f"GPU Adi: {torch.cuda.get_device_name(0)}")
    print(f"GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("UYARI: CUDA kullanilamiyor!")
    print("CPU modunda calisacak (yavas)")
