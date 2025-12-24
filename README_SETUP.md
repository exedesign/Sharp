# 🚀 SHARP Otomatik Kurulum Sistemi

Projeniz için otomatik kurulum ve başlatma sistemi. İlk kullanımda tüm gereksinimleri kurar, sonraki kullanımlarda direkt başlatır.

## 📋 Gereksinimler

- Python 3.8+
- Windows, Linux veya macOS
- İnternet bağlantısı (ilk kurulum için ~2.5GB indirme)
- (Opsiyonel) NVIDIA GPU + CUDA 12.4

## 🎯 Hızlı Başlangıç

### Windows - En Kolay Yöntem

**Seçenek 1: Tam Otomatik (Önerilen)**
```cmd
start.bat
```
İlk çalıştırmada kurulum yapılır (~10-15 dakika), sonraki kullanımlarda direkt başlar.

**Seçenek 2: Sadece Çalıştır (Kurulum yapıldıysa)**
```cmd
run.bat
```

**Seçenek 3: Manuel**
```cmd
python install.py
```

### Linux/macOS

```bash
chmod +x start.sh
./start.sh
```

## 🔧 Kurulum Detayları

### İlk Çalıştırma (Tek Sefer)

Otomatik kurulum şunları yapar:
1. ✅ Python 3.8+ kontrolü
2. 📦 Virtual environment (`.venv`)
3. 🔥 PyTorch CUDA 12.4 (~2.5GB indirme)
4. 📚 Proje bağımlılıkları
5. 🤖 Model kontrolü
6. 🎮 CUDA testi
7. ✨ `.setup_complete` işareti

**Süre:** 10-15 dakika (internet hızınıza bağlı)

### Sonraki Kullanımlar

- `.setup_complete` kontrolü
- Direkt uygulama başlatma
- **Süre:** ~0 saniye ⚡

## 📂 Dosyalar

```
ml-sharp/
├── install.py         # Ana kurulum scripti (çok platformlu)
├── start.bat          # Windows: Otomatik kurulum + başlat
├── run.bat            # Windows: Sadece başlat (hızlı)
├── start.sh           # Linux/macOS: Otomatik kurulum + başlat
├── .setup_complete    # Kurulum tamamlandı işareti (otomatik)
└── .venv/             # Virtual environment (otomatik)
```

## 💡 Kullanım Senaryoları

### Senaryo 1: İlk Kurulum
```cmd
> start.bat

SHARP Kurulum ve Baslatma Yardimcisi
Ilk kurulum baslatiliyor...

[1] Python sürümü kontrol ediliyor...
[OK] Python 3.13.7

[2] Virtual environment kontrol ediliyor...
Virtual environment oluşturuluyor...
[OK] Virtual environment oluşturuldu

[3] PyTorch CUDA 12.4 kontrol ediliyor...
PyTorch CUDA yükleniyor... (İndirme ~2.5GB, 5-10 dakika sürebilir)
[OK] PyTorch CUDA yüklendi

[4] Proje bağımlılıkları yükleniyor...
[OK] Bağımlılıklar yüklendi

Kurulum Basariyla Tamamlandi!

SHARP Uygulamasi Baslatiliyor
```

### Senaryo 2: Hızlı Başlatma (Kurulum Zaten Var)
```cmd
> run.bat

Virtual environment aktive edildi
Uygulama baslatiliyor...
```

## 🔄 Yeniden Kurulum

Sorun yaşarsanız veya temiz kurulum isterseniz:

```cmd
rmdir /s .venv
del .setup_complete
start.bat
```

Linux/macOS:
```bash
rm -rf .venv .setup_complete
./start.sh
```

## ⚙️ Manuel Kurulum

Gelişmiş kullanıcılar için:

```bash
# 1. Virtual environment
python -m venv .venv

# Windows aktive:
.venv\Scripts\activate

# Linux/macOS aktive:
source .venv/bin/activate

# 2. PyTorch CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Bağımlılıklar
pip install -r requirements.txt

# 4. Başlat
python app.py
```

## 🐛 Sorun Giderme

### "Python bulunamadı"
- Python 3.8+ yükleyin
- PATH'e ekleyin

### "CUDA kullanılamıyor"
- CPU modunda çalışır (daha yavaş)
- NVIDIA GPU + CUDA 12.4 drivers yükleyin

### "Module not found"
- Yeniden kurulum yapın

### PyTorch İndirme Çok Uzun Sürüyor
- Normal, ~2.5GB dosya
- İnternet hızınızı kontrol edin
- Cache kullanılıyor, iptal edip tekrar başlatabilirsiniz

### "Operation cancelled by user"
- PyTorch indirme iptal edilmiş
- `start.bat`'ı tekrar çalıştırın, cache'den devam eder

## 📊 Sistem Gereksinimleri

**Minimum:**
- CPU: 64-bit işlemci
- RAM: 8 GB
- Disk: 10 GB boş alan
- GPU: Yok (CPU modu, yavaş)

**Önerilen:**
- CPU: Intel i7/i9 veya AMD Ryzen 7/9
- RAM: 16 GB+
- Disk: 20 GB+ SSD
- GPU: NVIDIA RTX 3060+ (8GB+ VRAM)

## 🎨 Uygulama Kullanımı

Kurulum sonrası tarayıcı otomatik açılır (`http://localhost:7860`):

1. Resim yükle
2. "Generate 3D Model" tıkla
3. 3D Viewer'da incele
4. PLY formatında indir

## 📝 Sık Sorulan Sorular

**S: İlk kurulum ne kadar sürer?**
C: 10-15 dakika (PyTorch indirme ~2.5GB)

**S: Her seferinde kurulum yapar mı?**
C: Hayır, sadece ilk seferde. `.setup_complete` kontrolü yapar.

**S: GPU olmadan çalışır mı?**
C: Evet, CPU modunda çalışır ama daha yavaş.

**S: Kurulumu nasıl sıfırlarım?**
C: `.setup_complete` ve `.venv` klasörünü silin.

---

**🎉 Kolay Kullanımlar!**
