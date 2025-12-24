#!/bin/bash
# SHARP Hızlı Başlatma
# ====================
# Bu script Python install.py'yi çalıştırır

echo ""
echo "===================================================="
echo "     SHARP - Otomatik Kurulum ve Başlatma"
echo "===================================================="
echo ""

# Python kontrolü
if ! command -v python3 &> /dev/null; then
    echo "[HATA] Python3 bulunamadı!"
    echo "Lütfen Python 3.8 veya daha yüksek bir sürüm yükleyin."
    exit 1
fi

# install.py'yi çalıştır
python3 install.py
