"""
SHARP Otomatik Kurulum ve Baslat Yardimcisi
============================================

Bu script:
1. Ilk kurulum kontrolu yapar (.setup_complete dosyasi)
2. Python surumunu kontrol eder (3.8+)
3. Virtual environment olusturur/aktive eder
4. PyTorch CUDA 12.4 yukler
5. Tum bagimliliklari yukler
6. Model dosyasinin varligini kontrol eder
7. Kurulum tamamlandigini isaretler
8. Uygulamayi baslatir

Kullanim:
    python install.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# Proje kok dizinini belirle
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR

# Proje dizinine gec
os.chdir(PROJECT_ROOT)

# Windows encoding sorunlari icin
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

class Colors:
    """Terminal renkleri"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Başlık yazdır"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_step(step_num, text):
    """Adım yazdır"""
    print(f"{Colors.OKBLUE}[{step_num}] {text}{Colors.ENDC}")

def print_success(text):
    """Basari mesaji"""
    print(f"{Colors.OKGREEN}[OK] {text}{Colors.ENDC}")

def print_error(text):
    """Hata mesaji"""
    print(f"{Colors.FAIL}[HATA] {text}{Colors.ENDC}")

def print_warning(text):
    """Uyari mesaji"""
    print(f"{Colors.WARNING}[UYARI] {text}{Colors.ENDC}")

def run_command(cmd, check=True, capture_output=False, shell=False):
    """Komutu çalıştır"""
    try:
        if capture_output:
            result = subprocess.run(cmd, check=check, capture_output=True, 
                                  text=True, shell=shell)
            return result
        else:
            subprocess.run(cmd, check=check, shell=shell)
            return None
    except subprocess.CalledProcessError as e:
        if capture_output and hasattr(e, 'stderr'):
            print_error(f"Hata: {e.stderr}")
        raise

def check_python_version():
    """Check Python version / Python sürümünü kontrol et"""
    print_step(1, "Checking Python version / Python sürümü kontrol ediliyor...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required, current: {version.major}.{version.minor}")
        print_error(f"Python 3.8+ gerekli, mevcut: {version.major}.{version.minor}")
        sys.exit(1)
    
    # Warning for Python 3.13+
    if version.major == 3 and version.minor >= 13:
        print_warning(f"Python {version.major}.{version.minor}.{version.micro} detected")
        print_warning("PyTorch may have compatibility issues with Python 3.13+")
        print_warning("PyTorch, Python 3.13+ ile uyumluluk sorunları yaşayabilir")
        print_warning("Recommended: Python 3.11 or 3.12 / Önerilen: Python 3.11 veya 3.12")
        print("")
    else:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")

def check_setup_complete():
    """Check if setup is complete - In project directory / Kurulum tamamlanmış mı kontrol et - Proje dizininde"""
    setup_file = PROJECT_ROOT / ".setup_complete"
    return setup_file.exists()

def mark_setup_complete():
    """Mark setup as complete - In project directory / Kurulum tamamlandı işareti koy - Proje dizininde"""
    setup_file = PROJECT_ROOT / ".setup_complete"
    setup_file.write_text("Setup completed successfully")
    print_success(f"Setup completed successfully / Kurulum basariyla tamamlandi: {setup_file}")

def get_venv_paths():
    """Get virtual environment paths - In project directory / Virtual environment yollarını al - Proje dizininde"""
    is_windows = platform.system() == "Windows"
    venv_dir = PROJECT_ROOT / ".venv"
    
    if is_windows:
        python_path = venv_dir / "Scripts" / "python.exe"
        pip_path = venv_dir / "Scripts" / "pip.exe"
        activate_path = venv_dir / "Scripts" / "activate.bat"
    else:
        python_path = venv_dir / "bin" / "python"
        pip_path = venv_dir / "bin" / "pip"
        activate_path = venv_dir / "bin" / "activate"
    
    return python_path, pip_path, activate_path

def create_virtual_environment():
    """Create virtual environment - In project directory / Virtual environment oluştur - Proje dizininde"""
    print_step(2, "Checking virtual environment / Virtual environment kontrol ediliyor...")
    
    venv_dir = PROJECT_ROOT / ".venv"
    python_path, pip_path, _ = get_venv_paths()
    
    if venv_dir.exists() and python_path.exists():
        print_success(f"Virtual environment exists / Virtual environment mevcut: {venv_dir}")
        return python_path, pip_path
    
    print(f"Creating virtual environment / Virtual environment olusturuluyor: {venv_dir}")
    run_command([sys.executable, "-m", "venv", str(venv_dir)])
    
    # Update pip / pip'i güncelle
    print("Updating pip / pip güncelleniyor...")
    run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
    
    print_success(f"Virtual environment created / Virtual environment olusturuldu: {venv_dir}")
    return python_path, pip_path

def install_pytorch(pip_path):
    """Install PyTorch CUDA 12.4 / PyTorch CUDA 12.4 yükle"""
    print_step(3, "Checking PyTorch CUDA 12.4 / PyTorch CUDA 12.4 kontrol ediliyor...")
    
    # PyTorch CUDA yüklü mü kontrol et
    python_path, _, _ = get_venv_paths()
    check_code = """
import sys
try:
    import torch
    if torch.cuda.is_available():
        print('CUDA:OK')
        sys.exit(0)
    else:
        print('CUDA:NO')
        sys.exit(1)
except:
    print('TORCH:NO')
    sys.exit(1)
"""
    
    result = run_command(
        [str(python_path), "-c", check_code],
        check=False,
        capture_output=True
    )
    
    if result and "CUDA:OK" in result.stdout:
        print_success("PyTorch CUDA zaten yüklü ve çalışıyor")
        # CUDA sürümünü göster
        result2 = run_command([
            str(python_path), "-c",
            "import torch; print(f'PyTorch: {torch.__version__}')"
        ], capture_output=True, check=False)
        if result2 and result2.stdout:
            print(f"  {result2.stdout.strip()}")
        return
    
    print("PyTorch CUDA yükleniyor... (İndirme ~2.5GB, 5-10 dakika sürebilir)")
    print("Lütfen bekleyin, işlem iptal edilmemelidir...")
    print("CUDA 12.4 desteği aktif olacak")
    print("[NOT] Ilerleme cubugu goruntulenecek, lutfen bekleyin...\n")
    
    # Önce mevcut torch'u kaldır (CPU sürümü varsa)
    print("Mevcut PyTorch sürümleri kaldırılıyor...")
    run_command([str(pip_path), "uninstall", "-y", "torch", "torchvision", "torchaudio"], check=False)
    
    # CUDA 12.4 sürümünü yükle - ÖNCELİK! (İlerleme çubuğu ile)
    print("PyTorch CUDA 12.4 yükleniyor...\n")
    run_command([
        str(pip_path), "install", "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124",
        "--progress-bar", "on"
    ])
    
    # CUDA kontrolü yap
    result_check = run_command([
        str(python_path), "-c",
        "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
    ], capture_output=True, check=False)
    
    if result_check and result_check.stdout:
        print(result_check.stdout.strip())
    
    print_success("PyTorch CUDA yüklendi")

def install_dependencies(pip_path):
    """Install dependencies - From project directory / Bağımlılıkları yükle - Proje dizininden"""
    print_step(4, "Installing project dependencies / Proje bağımlılıkları yüklen iyor...")
    
    requirements_file = PROJECT_ROOT / "requirements.txt"
    if not requirements_file.exists():
        print_error(f"requirements.txt not found / requirements.txt bulunamadi: {requirements_file}")
        sys.exit(1)
    
    print(f"Installing dependencies / Bagimliliklar yukleniyor: {requirements_file}")
    print("(This may take a few minutes / Bu islem birkac dakika surebilir)")
    print("PyTorch packages will be skipped (CUDA version already installed)")
    print("PyTorch paketleri atlanacak (CUDA surumu zaten yuklendi)\n")
    
    # torch, torchvision, torchaudio'yu requirements'tan skip et
    # Çünkü CUDA sürümleri zaten yüklendi
    import tempfile
    editable_install_line = None
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        with open(requirements_file, 'r', encoding='utf-8') as f:
            for line in f:
                # torch ile ilgili satırları atla
                if any(pkg in line.lower() for pkg in ['torch==', 'torchvision==', 'torchaudio==']):
                    print(f"  Atlaniyor: {line.strip()}")
                    continue
                # -e . (editable install) satırını ayrı tut
                if line.strip().startswith('-e'):
                    editable_install_line = line.strip()
                    print(f"  Editable install ayri yapilacak: {line.strip()}")
                    continue
                tmp.write(line)
        tmp_path = tmp.name
    
    try:
        # Geçici requirements dosyasından yükle
        run_command([str(pip_path), "install", "-r", tmp_path])
        
        # Editable install'ı --no-deps ile yap (torch bağımlılıklarını kurmaz)
        if editable_install_line:
            print("\nEditable install yapiliyor (torch bagimlilik atlanacak)...")
            # -e . yerine -e . --no-deps kullan
            run_command([str(pip_path), "install", "--no-deps", editable_install_line])
    finally:
        # Geçici dosyayı sil
        Path(tmp_path).unlink(missing_ok=True)
    
    print_success("Bagimliliklar yuklendi")

def check_model_file():
    """Model dosyasını kontrol et - Proje dizininde"""
    print_step(5, "Model dosyası kontrol ediliyor...")
    
    model_path = PROJECT_ROOT / "models" / "sharp_model.pt"
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print_success(f"Model dosyası mevcut ({size_mb:.1f} MB): {model_path}")
    else:
        print_warning("Model dosyası bulunamadı")
        print_warning("Model ilk çalıştırmada otomatik indirilecek")
        # models klasörünü oluştur
        model_path.parent.mkdir(exist_ok=True)
        print(f"Models klasoru olusturuldu: {model_path.parent}")

def check_cuda():
    """CUDA kontrolü - Detaylı"""
    print_step(6, "CUDA kontrolü yapılıyor...")
    
    python_path, _, _ = get_venv_paths()
    
    cuda_check = """
import torch
print(f'PyTorch Surumu: {torch.__version__}')
print(f'CUDA Kullanilabilir: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Surumu: {torch.version.cuda}')
    print(f'GPU Sayisi: {torch.cuda.device_count()}')
    print(f'GPU Adi: {torch.cuda.get_device_name(0)}')
else:
    print('UYARI: CUDA kullanılamıyor!')
"""
    
    result = run_command([
        str(python_path), "-c", cuda_check
    ], capture_output=True, check=False)
    
    if result and result.stdout:
        print(result.stdout)
        
        if "True" in result.stdout:
            print_success("CUDA aktif ve hazir!")
        else:
            print_warning("CUDA kullanılamiyor, CPU modu kullanilacak")
            print_warning("NVIDIA GPU ve CUDA 12.4 drivers yuklu oldugundan emin olun")

def verify_environment():
    """Ortam doğrulama - Hızlı kontrol"""
    python_path, pip_path, _ = get_venv_paths()
    
    # Virtual env var mı?
    if not python_path.exists():
        return False, None, None
    
    # PyTorch CUDA yüklü mü?
    check_code = "import torch; exit(0 if torch.cuda.is_available() else 1)"
    result = run_command(
        [str(python_path), "-c", check_code],
        check=False,
        capture_output=True
    )
    
    if result and result.returncode == 0:
        return True, python_path, pip_path
    
    return False, python_path, pip_path

def setup_environment(python_path, pip_path):
    """Tam kurulum - Tüm adımlar"""
    print("\nKurulum baslatiliyor...\n")
    
    # 1. Python sürümü
    check_python_version()
    
    # 2. Virtual environment
    if not python_path or not python_path.exists():
        python_path, pip_path = create_virtual_environment()
    else:
        print_step(2, "Virtual environment kontrol ediliyor...")
        print_success(f"Virtual environment mevcut: {python_path.parent}")
    
    # 3. PyTorch CUDA
    install_pytorch(pip_path)
    
    # 4. Bağımlılıklar
    install_dependencies(pip_path)
    
    # 5. Model kontrolü
    check_model_file()
    
    # 6. CUDA doğrulama
    check_cuda()
    
    # 7. Kurulum işareti
    mark_setup_complete()
    
    print_header("Kurulum Basariyla Tamamlandi!")
    print("Sonraki calistirmada kurulum atlanacak\n")
    
    return python_path

def launch_app(python_path):
    """Uygulamayı başlat"""
    print_header("SHARP Uygulamasi Baslatiliyor")
    
    app_file = PROJECT_ROOT / "app.py"
    print(f"Dizin: {PROJECT_ROOT}")
    print(f"Dosya: {app_file.name}")
    print("Gradio arayuzu aciliyor...")
    print("Durdurmak icin Ctrl+C\n")
    
    # Başlat
    os.execv(str(python_path), [str(python_path), str(app_file)])

def main():
    """Ana fonksiyon - Akıllı kurulum ve başlatma"""
    print_header("SHARP Otomatik Sistem")
    
    print(f"Proje: {PROJECT_ROOT}")
    print(f"Dizin: {os.getcwd()}\n")
    
    # App dosyası kontrolü
    app_file = PROJECT_ROOT / "app.py"
    if not app_file.exists():
        print_error(f"app.py bulunamadi: {app_file}")
        print_error("Proje kok dizininde calistirilmali!")
        sys.exit(1)
    
    # Kurulum durumu kontrolü
    setup_complete = check_setup_complete()
    
    if setup_complete:
        # Kurulum tamamlanmış, sadece basit kontrol
        print_success("Kurulum mevcut, atlanıyor")
        python_path, _, _ = get_venv_paths()
        
        if not python_path.exists():
            print_error("Virtual environment bulunamadi!")
            print_error("Lutfen .setup_complete dosyasini silin ve yeniden calistirin")
            sys.exit(1)
        
        print_success(f"Python: {python_path}\n")
    else:
        # İlk kurulum gerekli
        print_warning("Ilk kurulum baslatiliyor...\n")
        env_ready, python_path, pip_path = verify_environment()
        python_path = setup_environment(python_path, pip_path)
    
    # Uygulamayı başlat
    launch_app(python_path)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nKurulum iptal edildi")
        sys.exit(0)
    except Exception as e:
        print_error(f"Beklenmeyen hata: {str(e)}")
        sys.exit(1)
