# 🎯 SHARP Otomatik Kurulum Sistemi

## ✅ Tüm Dosyalar Proje İçinde

Kurulum sistemi **tüm dosyaları proje ana klasöründe** oluşturur:

```
C:\Users\FE\Desktop\Sharp\ml-sharp\
├── .venv\                 # Virtual environment (buradadır!)
├── .setup_complete        # Kurulum tamamlandı işareti (buradadır!)
├── models\                # Model dosyaları (buradadır!)
│   └── sharp_model.pt     # İlk çalıştırmada indirilir
├── install.py             # Kurulum scripti
├── start.bat              # Windows otomatik başlatma
├── run.bat                # Windows hızlı başlatma
├── app.py                 # Ana uygulama
└── ... diğer dosyalar
```

## 🚀 Kullanım

### Windows - Basit Başlatma

**Tam Otomatik (İlk Kurulum + Başlatma):**
```cmd
start.bat
```

**Hızlı Başlatma (Kurulum Varsa):**
```cmd
run.bat
```

## 🔧 Kurulum Detayları

### Otomatik Olarak Oluşturulan Dosyalar

1. **`.venv\`** - Proje kök dizininde
   - Tüm Python paketleri burada
   - İzole ortam, sistem Python'ını etkilemez
   - Konum: `C:\Users\FE\Desktop\Sharp\ml-sharp\.venv`

2. **`.setup_complete`** - Proje kök dizininde
   - Kurulum tamamlandı işareti
   - Bu dosya varsa kurulum atlanır
   - Konum: `C:\Users\FE\Desktop\Sharp\ml-sharp\.setup_complete`

3. **`models\sharp_model.pt`** - Proje kök dizininde
   - Model dosyası (~2GB)
   - İlk çalıştırmada otomatik indirilir
   - Konum: `C:\Users\FE\Desktop\Sharp\ml-sharp\models\sharp_model.pt`

## 📍 Dizin Kontrolü

`install.py` scripti otomatik olarak:
- Proje kök dizinini tespit eder
- Tüm dosyaları proje içinde oluşturur
- Çalışma dizinini proje kök dizinine ayarlar

```python
# Otomatik dizin yönetimi
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR
os.chdir(PROJECT_ROOT)
```

## 🔄 Temizleme

Tüm kurulum dosyalarını silmek için:

```cmd
# Proje kök dizininde
rmdir /s .venv
del .setup_complete
# models klasörünü tutabilirsiniz (tekrar indirmemek için)
```

## ✅ Doğrulama

Kurulum sonrası dosyaları kontrol edin:

```cmd
# Proje dizininde olduğunuzu doğrulayın
cd C:\Users\FE\Desktop\Sharp\ml-sharp

# Dosyaları listeleyin
dir .venv
dir .setup_complete
dir models
```

## 🎯 Önemli Notlar

✅ **Tüm dosyalar proje içinde**
- Sistem dizinlerine yazılmaz
- AppData veya Program Files kullanılmaz
- Tümü `ml-sharp\` klasörü altında

✅ **Taşınabilir**
- Projeyi taşıyabilirsiniz
- `.venv` ve `.setup_complete` de taşınır
- Yeniden kurulum gerekmez

✅ **Temiz Kaldırma**
- Sadece proje klasörünü silin
- Sistem temiz kalır

## 🐛 Sorun Giderme

### "Virtual environment bulunamadı"
```cmd
# Proje dizininde olduğunuzu doğrulayın
cd C:\Users\FE\Desktop\Sharp\ml-sharp
# Yeniden kurun
del .setup_complete
start.bat
```

### Dosyalar farklı yerde mi?
- İmkansız! Script otomatik proje dizinini kullanır
- `install.py` her zaman kendi bulunduğu dizinde çalışır

---

**✨ Tüm kurulum dosyaları projenizin içindedir!**
