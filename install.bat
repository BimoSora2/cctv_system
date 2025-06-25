@echo off
setlocal enabledelayedexpansion

echo ======================================
echo 🚀 YOLOv8 CCTV System Setup Script
echo ======================================
echo.

:: Check Python version
echo 🐍 Checking Python version...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python not found. Please install Python 3.7+ first.
    goto :end
) else (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo ✅ Found: !PYTHON_VERSION!
)

:: Check if pip is available
echo.
echo 📦 Checking pip availability...
python -m pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ pip not found. Please install pip first.
    goto :end
) else (
    for /f "tokens=*" %%i in ('python -m pip --version') do set PIP_VERSION=%%i
    echo ✅ Found: !PIP_VERSION!
    set PIP_CMD=python -m pip
)

:: Create virtual environment (recommended)
echo.
echo 🔧 Setting up virtual environment (recommended)...
set /p create_venv="Do you want to create a virtual environment? (y/N): "
if /i "!create_venv!"=="y" (
    echo Creating virtual environment 'venv'...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Failed to create virtual environment.
        goto :end
    )
    call venv\Scripts\activate.bat
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Failed to activate virtual environment.
        goto :end
    )
    echo ✅ Virtual environment created and activated
    echo 💡 To activate later: venv\Scripts\activate.bat
) else (
    echo ⚠️ Installing globally (not recommended for production)
)

:: Upgrade pip
echo.
echo ⬆️ Upgrading pip...
%PIP_CMD% install --upgrade pip

:: Install core dependencies
echo.
echo 📦 Installing core dependencies...
echo This may take a few minutes for the first time...

:: Install dependencies
echo.
echo Installing opencv-python...
%PIP_CMD% install opencv-python>=4.5.0
if %ERRORLEVEL% EQU 0 (
    echo ✅ opencv-python installed successfully
) else (
    echo ❌ Failed to install opencv-python
    echo 💡 Try: %PIP_CMD% install opencv-python>=4.5.0 --no-cache-dir
)

echo.
echo Installing numpy...
%PIP_CMD% install numpy>=1.19.0
if %ERRORLEVEL% EQU 0 (
    echo ✅ numpy installed successfully
) else (
    echo ❌ Failed to install numpy
    echo 💡 Try: %PIP_CMD% install numpy>=1.19.0 --no-cache-dir
)

echo.
echo Installing flask...
%PIP_CMD% install flask>=2.0.0
if %ERRORLEVEL% EQU 0 (
    echo ✅ flask installed successfully
) else (
    echo ❌ Failed to install flask
    echo 💡 Try: %PIP_CMD% install flask>=2.0.0 --no-cache-dir
)

echo.
echo Installing flask-socketio...
%PIP_CMD% install flask-socketio>=5.0.0
if %ERRORLEVEL% EQU 0 (
    echo ✅ flask-socketio installed successfully
) else (
    echo ❌ Failed to install flask-socketio
    echo 💡 Try: %PIP_CMD% install flask-socketio>=5.0.0 --no-cache-dir
)

echo.
echo Installing onvif-zeep...
%PIP_CMD% install onvif-zeep>=0.2.12
if %ERRORLEVEL% EQU 0 (
    echo ✅ onvif-zeep installed successfully
) else (
    echo ❌ Failed to install onvif-zeep
    echo 💡 Try: %PIP_CMD% install onvif-zeep>=0.2.12 --no-cache-dir
)

echo.
echo Installing ultralytics...
%PIP_CMD% install ultralytics>=8.0.0
if %ERRORLEVEL% EQU 0 (
    echo ✅ ultralytics installed successfully
) else (
    echo ❌ Failed to install ultralytics
    echo 💡 Try: %PIP_CMD% install ultralytics>=8.0.0 --no-cache-dir
)

echo.
echo Installing torch and torchvision...
%PIP_CMD% install torch>=1.9.0 torchvision>=0.10.0
if %ERRORLEVEL% EQU 0 (
    echo ✅ torch and torchvision installed successfully
) else (
    echo ❌ Failed to install torch and torchvision
    echo 💡 Try: %PIP_CMD% install torch>=1.9.0 torchvision>=0.10.0 --no-cache-dir
)

echo.
echo Installing yt-dlp...
%PIP_CMD% install yt-dlp>=2023.1.6
if %ERRORLEVEL% EQU 0 (
    echo ✅ yt-dlp installed successfully
) else (
    echo ❌ Failed to install yt-dlp
    echo 💡 Try: %PIP_CMD% install yt-dlp>=2023.1.6 --no-cache-dir
)

echo.
echo Installing Pillow...
%PIP_CMD% install Pillow>=8.0.0
if %ERRORLEVEL% EQU 0 (
    echo ✅ Pillow installed successfully
) else (
    echo ❌ Failed to install Pillow
    echo 💡 Try: %PIP_CMD% install Pillow>=8.0.0 --no-cache-dir
)

echo.
echo Installing requests...
%PIP_CMD% install requests>=2.25.0
if %ERRORLEVEL% EQU 0 (
    echo ✅ requests installed successfully
) else (
    echo ❌ Failed to install requests
    echo 💡 Try: %PIP_CMD% install requests>=2.25.0 --no-cache-dir
)

echo.
echo Installing psutil...
%PIP_CMD% install psutil>=5.8.0
if %ERRORLEVEL% EQU 0 (
    echo ✅ psutil installed successfully
) else (
    echo ❌ Failed to install psutil
    echo 💡 Try: %PIP_CMD% install psutil>=5.8.0 --no-cache-dir
)

:: Install PyTorch with CUDA support (optional but recommended for performance)
echo.
echo 🔥 GPU Acceleration Setup...
set /p install_cuda="Do you want to install PyTorch with CUDA support for GPU acceleration? (y/N): "
if /i "!install_cuda!"=="y" (
    echo Installing PyTorch with CUDA support...
    echo 💡 This will download ~2GB of packages
    
    :: Detect NVIDIA GPU
    nvidia-smi >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        :: Try to detect CUDA version
        for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader') do set NVIDIA_DRIVER=%%i
        echo 🔍 Detected NVIDIA driver: !NVIDIA_DRIVER!
        
        :: Rough estimate of CUDA version from driver
        :: This is a simplification - exact mapping depends on driver version
        if !NVIDIA_DRIVER! GEQ 500 (
            echo 🔍 Estimated CUDA 12.x compatibility
            %PIP_CMD% install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        ) else if !NVIDIA_DRIVER! GEQ 450 (
            echo 🔍 Estimated CUDA 11.x compatibility
            %PIP_CMD% install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        ) else (
            echo ⚠️ Older NVIDIA driver detected, installing CPU version
            %PIP_CMD% install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        )
    ) else (
        echo 🚫 NVIDIA GPU not detected, installing CPU version
        %PIP_CMD% install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    )
) else (
    echo Installing CPU-only PyTorch (slower but compatible with all systems)
    %PIP_CMD% install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

:: Download YOLOv8 models
echo.
echo 📥 Downloading YOLOv8 models...
echo 💡 Models will be downloaded automatically on first use, but you can pre-download them:

:: Create models directory if it doesn't exist
if not exist "models\" mkdir models

set /p download_models="Download YOLOv8 models now? (y/N): "
if /i "!download_models!"=="y" (
    echo Downloading yolov8n.pt...
    python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('✅ yolov8n.pt downloaded successfully')" 2>nul
    
    echo Downloading yolov8s.pt...
    python -c "from ultralytics import YOLO; model = YOLO('yolov8s.pt'); print('✅ yolov8s.pt downloaded successfully')" 2>nul
    
    echo Downloading yolov8m.pt...
    python -c "from ultralytics import YOLO; model = YOLO('yolov8m.pt'); print('✅ yolov8m.pt downloaded successfully')" 2>nul
) else (
    echo ⏭️ Models will be downloaded automatically when needed
)

:: Test yt-dlp with YouTube
echo.
echo 📺 Testing yt-dlp with YouTube...
set /p test_ytdlp="Test yt-dlp with a sample YouTube video? (y/N): "
if /i "!test_ytdlp!"=="y" (
    echo Testing yt-dlp functionality...
    python -c "import yt_dlp; ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': False}; with yt_dlp.YoutubeDL(ydl_opts) as ydl: info = ydl.extract_info('https://www.youtube.com/watch?v=dQw4w9WgXcQ', download=False); print('✅ yt-dlp test successful'); print(f'   Title: {info.get(\"title\", \"Unknown\")}'); print(f'   Duration: {info.get(\"duration\", \"Unknown\")} seconds'); print(f'   View count: {info.get(\"view_count\", \"Unknown\")}')" 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ yt-dlp test failed
        echo 💡 This might be due to network issues or YouTube blocking
    )
) else (
    echo ⏭️ yt-dlp test skipped
)

:: Test installation
echo.
echo 🧪 Testing installation...
python -c "
import sys
print('Testing imports...')

try:
    import cv2
    print('✅ OpenCV:', cv2.__version__)
except ImportError as e:
    print('❌ OpenCV failed:', e)

try:
    import numpy as np
    print('✅ NumPy:', np.__version__)
except ImportError as e:
    print('❌ NumPy failed:', e)

try:
    import flask
    print('✅ Flask:', flask.__version__)
except ImportError as e:
    print('❌ Flask failed:', e)

try:
    import flask_socketio
    print('✅ Flask-SocketIO:', flask_socketio.__version__)
except ImportError as e:
    print('❌ Flask-SocketIO failed:', e)

try:
    from ultralytics import YOLO
    print('✅ YOLOv8 (ultralytics): Available')
    
    # Test YOLO model loading
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('🔥 PyTorch device:', device)
    
except ImportError as e:
    print('❌ YOLOv8 failed:', e)
except Exception as e:
    print('⚠️ YOLOv8 warning:', e)
    print('💡 This is normal on first run - models will download automatically')

try:
    from onvif import ONVIFCamera
    print('✅ ONVIF: Available')
except ImportError as e:
    print('❌ ONVIF failed:', e)

try:
    import yt_dlp
    print('✅ yt-dlp:', yt_dlp.version.__version__)
    print('   📺 YouTube Live streaming support enabled')
except ImportError as e:
    print('❌ yt-dlp failed:', e)
    print('💡 Install with: pip install yt-dlp')

try:
    import psutil
    memory = psutil.virtual_memory()
    print('✅ psutil: Available')
    print(f'   💾 System RAM: {memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available')
except ImportError as e:
    print('⚠️ psutil not available (optional for system monitoring)')

print('')
print('🎯 System Requirements Check:')
import platform
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'Architecture: {platform.machine()}')

# Check webcam availability
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print('📹 Webcam: Available')
        else:
            print('📹 Webcam: Detected but cannot read frames')
        cap.release()
    else:
        print('📹 Webcam: Not detected')
except:
    print('📹 Webcam: Could not test')

# Network connectivity test
try:
    import requests
    response = requests.get('https://www.google.com', timeout=5)
    if response.status_code == 200:
        print('🌐 Internet: Connected')
    else:
        print('🌐 Internet: Limited connectivity')
except:
    print('🌐 Internet: No connection or timeout')
"

echo.
echo ======================================
echo 🎉 Installation Complete!
echo ======================================
echo.
echo 📋 Next Steps:
echo 1. Start your CCTV system:
echo    python app.py
echo.
echo 2. Open your browser to:
echo    http://localhost:4000
echo.
echo 3. Connect your camera using:
echo    • RTSP URL (IP cameras)
echo    • USB Webcam (auto-detected)
echo    • YouTube Live streams
echo    • Video files
echo.
echo 💡 Features Available:
echo • 🎥 RTSP/IP Camera support with ONVIF PTZ control
echo • 📹 USB/Built-in webcam auto-detection
echo • 📺 YouTube Live streaming (requires yt-dlp)
echo • 🎮 Twitch stream support
echo • 📁 Video file playback with timing control
echo • 🤖 YOLOv8 AI person detection
echo • 🎯 Auto person tracking (PTZ cameras)
echo • ⚡ Ultra low-latency streaming
echo • 🔴 Live stream timing preservation
echo.
echo 🔧 Model Performance Tips:
echo • yolov8n.pt: Fastest detection (~30 FPS)
echo • yolov8s.pt: Balanced speed/accuracy (~20 FPS)
echo • yolov8m.pt: Better accuracy (~15 FPS)
echo • yolov8l.pt: High accuracy (~10 FPS)
echo • GPU acceleration will be used automatically if available
echo.
echo 📺 YouTube Live Usage:
echo • Use format: https://www.youtube.com/watch?v=VIDEO_ID
echo • Or channel live: https://www.youtube.com/c/CHANNEL/live
echo • yt-dlp will extract direct stream URLs automatically
echo • System preserves original live timing
echo.
echo 🔧 Troubleshooting:
echo • If imports fail, try: pip install --no-cache-dir ^<package^>
echo • For CUDA issues, install appropriate PyTorch version
echo • For camera connection issues, check firewall and network
echo • For YouTube issues, ensure yt-dlp is updated: pip install --upgrade yt-dlp
echo • If yt-dlp fails, YouTube may be blocking requests - try different streams
echo.
echo 📚 Documentation:
echo • YOLOv8: https://docs.ultralytics.com/
echo • OpenCV: https://docs.opencv.org/
echo • ONVIF: https://www.onvif.org/
echo • yt-dlp: https://github.com/yt-dlp/yt-dlp
echo.
echo 🚀 Ready to start your AI-powered CCTV system!
echo ======================================

:end
endlocal
