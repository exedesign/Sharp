"""Test script - Sadece kurulum testi, uygulama baslatma yok"""
import sys
from pathlib import Path

# install.py'den fonksiyonlari import et
sys.path.insert(0, str(Path(__file__).parent))

from install import (
    print_header, print_success, print_warning, print_error,
    PROJECT_ROOT, check_setup_complete, verify_environment,
    setup_environment, get_venv_paths
)

def test_install():
    """Kurulum sistemini test et"""
    print_header("Kurulum Sistemi Testi")
    
    print(f"Proje: {PROJECT_ROOT}\n")
    
    # 1. Kurulum durumu
    setup_complete = check_setup_complete()
    print(f"Setup Complete: {setup_complete}")
    
    # 2. Ortam kontrolu
    env_ready, python_path, pip_path = verify_environment()
    print(f"Environment Ready: {env_ready}")
    print(f"Python Path: {python_path}\n")
    
    # 3. Karar
    needs_setup = not setup_complete or not env_ready
    
    if needs_setup:
        print_warning("Kurulum gerekli")
        if not setup_complete:
            print("  -> .setup_complete bulunamadi")
        if not env_ready:
            print("  -> Ortam hazir degil")
        
        # Kurulum yap
        python_path = setup_environment(python_path, pip_path)
        print_success(f"\nKurulum tamamlandi: {python_path}")
    else:
        print_success("Sistem hazir!")
        print_success("Kurulum atlanacak")
    
    print("\n" + "="*60)
    print("Test tamamlandi - Uygulama baslatilmadi")
    print("="*60)

if __name__ == "__main__":
    test_install()
