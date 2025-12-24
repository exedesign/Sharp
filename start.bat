@echo off
chcp 65001 >nul
REM SHARP Quick Launcher / SHARP Hizli Baslatma
REM ==============================================
REM This batch file checks virtual environment and runs the app
REM Bu batch dosyasi virtual environment kontrolu yapar ve uygulamayi calistirir

echo.
echo ====================================================
echo     SHARP - Quick Start / Hizli Baslatma
echo ====================================================
echo.

REM Check if virtual environment exists / Virtual environment var mi kontrol et
if exist ".venv\Scripts\python.exe" (
    echo [OK] Virtual environment found / Virtual environment bulundu
    
    REM Check venv Python version / venv Python sürümünü kontrol et
    for /f "tokens=2 delims=." %%a in ('.venv\Scripts\python.exe --version 2^>^&1') do set VENVMINOR=%%a
    if "%VENVMINOR%"=="13" (
        echo.
        echo [WARNING] Python 3.13 detected in venv - NOT compatible!
        echo [UYARI] Venv'de Python 3.13 tespit edildi - Uyumlu DEGiL!
        echo.
        echo [AUTO-FIX] Checking for Python 3.12...
        echo [OTO-DUZELT] Python 3.12 kontrol ediliyor...
        
        REM Try to find Python 3.12 / Python 3.12 bulmaya calis
        py -3.12 --version >nul 2>&1
        if errorlevel 1 (
            echo.
            echo [ERROR] Python 3.12 not found! / Python 3.12 bulunamadi!
            echo.
            echo Opening download page in browser...
            echo Tarayicida indirme sayfasi aciliyor...
            start https://www.python.org/downloads/release/python-3120/
            echo.
            echo [INSTRUCTIONS] After installing Python 3.12:
            echo [TALIMATLAR] Python 3.12 kurduktan sonra:
            echo   1. Close this window / Bu pencereyi kapatin
            echo   2. Run start.bat again / start.bat'i tekrar calistirin
            echo.
            pause
            exit /b 1
        )
        
        echo [OK] Python 3.12 found / Python 3.12 bulundu
        echo [AUTO-FIX] Removing old virtual environment...
        echo [OTO-DUZELT] Eski virtual environment siliniyor...
        rmdir /s /q .venv
        del /q .setup_complete 2>nul
        echo.
        echo [AUTO-FIX] Creating new venv with Python 3.12...
        echo [OTO-DUZELT] Python 3.12 ile yeni venv olusturuluyor...
        py -3.12 -m venv .venv
        echo [OK] New virtual environment created / Yeni virtual environment olusturuldu
        echo.
        echo [INFO] Running full setup / Tam kurulum yapiliyor...
        python install.py
        exit /b 0
    )
    
    echo [OK] Starting application / Uygulama baslatiliyor...
    echo.
    
    REM Activate and run / Aktive et ve calistir
    call .venv\Scripts\activate.bat
    .venv\Scripts\python.exe app.py
    
    pause
    exit /b 0
) else (
    echo [INFO] Virtual environment not found / Virtual environment bulunamadi
    echo [INFO] Running setup first / Once kurulum yapiliyor...
    echo.
    
    REM Check Python / Python kontrolu
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found! / Python bulunamadi!
        echo [INFO] Please install Python 3.11 or 3.12
        echo [INFO] Lutfen Python 3.11 veya 3.12 yukleyin
        echo.
        pause
        exit /b 1
    )
    
    REM Run installer / Kurucu calistir
    python install.py
    
    pause
)
