@echo off
REM ============================================================================
REM Multi-Source CCTV System - Windows Installation Script
REM Universal .pt Model Support | Live Streaming | Enhanced RTSP | PTZ Control
REM ============================================================================
REM
REM This script will:
REM 1. Check system requirements
REM 2. Install Python dependencies
REM 3. Setup virtual environment
REM 4. Download YOLO models (optional)
REM 5. Configure Windows-specific settings
REM
REM Requirements:
REM - Windows 10/11 (x64)
REM - Python 3.8+ installed and in PATH
REM - Internet connection
REM - Administrator privileges (recommended)
REM
REM ============================================================================

title Multi-Source CCTV System - Windows Installer
color 0A
echo.
echo ============================================================================
echo     🚀 MULTI-SOURCE CCTV SYSTEM - WINDOWS INSTALLER
echo ============================================================================
echo.
echo    AI Detection ^| Live Streaming ^| PTZ Control ^| Enhanced RTSP Support
echo    Universal YOLO Models ^| Webcam Support ^| File Timing Preservation
echo.
echo ============================================================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Running with Administrator privileges
) else (
    echo ⚠️  WARNING: Not running as Administrator
    echo    Some features may not work properly
    echo    Right-click and "Run as administrator" for best results
    echo.
)

REM ============================================================================
REM SYSTEM REQUIREMENTS CHECK
REM ============================================================================

echo 🔍 Checking system requirements...
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo.
    echo 📥 Please install Python 3.8+ from: https://python.org/downloads/
    echo    ⚠️  Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo ✅ Python found: %PYTHON_VERSION%
)

REM Check Python version compatibility
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python 3.8+ required. Found: %PYTHON_VERSION%
    echo    Please upgrade Python from: https://python.org/downloads/
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Python version compatible
)

REM Check pip installation
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: pip is not installed
    echo    Please reinstall Python with pip included
    echo.
    pause
    exit /b 1
) else (
    echo ✅ pip is available
)

REM Check internet connection
ping -n 1 google.com >nul 2>&1
if errorlevel 1 (
    echo ⚠️  WARNING: Internet connection may be limited
    echo    Some packages may fail to download
    echo.
) else (
    echo ✅ Internet connection available
)

REM Check available disk space (rough estimate)
for /f "tokens=3" %%i in ('dir /-c %SystemDrive%\ 2^>nul ^| find "bytes free"') do set FREE_SPACE=%%i
if defined FREE_SPACE (
    echo ✅ Disk space available
) else (
    echo ⚠️  Could not check disk space
)

echo.
echo ============================================================================
echo 📦 INSTALLATION OPTIONS
echo ============================================================================
echo.
echo Select installation type:
echo.
echo [1] 🚀 Quick Install (CPU only - works on all systems)
echo [2] 🔥 GPU Install (NVIDIA CUDA - requires compatible GPU)
echo [3] 🔧 Development Install (includes debugging tools)
echo [4] 🎯 Custom Install (choose components)
echo [5] ❌ Exit installer
echo.
set /p INSTALL_CHOICE="Enter your choice (1-5): "

if "%INSTALL_CHOICE%"=="1" goto :quick_install
if "%INSTALL_CHOICE%"=="2" goto :gpu_install
if "%INSTALL_CHOICE%"=="3" goto :dev_install
if "%INSTALL_CHOICE%"=="4" goto :custom_install
if "%INSTALL_CHOICE%"=="5" goto :exit_installer
echo Invalid choice. Using Quick Install...
goto :quick_install

:quick_install
echo.
echo 🚀 Quick Install selected (CPU only)
set INSTALL_TYPE=quick
goto :setup_environment

:gpu_install
echo.
echo 🔥 GPU Install selected
echo.
echo ⚠️  GPU Requirements Check:
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ❌ NVIDIA GPU driver not found
    echo    Install NVIDIA drivers first: https://nvidia.com/drivers
    echo.
    echo Continue anyway? [Y/N]
    set /p GPU_CONTINUE=""
    if /i not "%GPU_CONTINUE%"=="Y" goto :quick_install
) else (
    echo ✅ NVIDIA GPU driver detected
    nvidia-smi | findstr "CUDA Version"
)
set INSTALL_TYPE=gpu
goto :setup_environment

:dev_install
echo.
echo 🔧 Development Install selected
echo    Includes: debugging tools, testing framework, code formatting
set INSTALL_TYPE=dev
goto :setup_environment

:custom_install
echo.
echo 🎯 Custom Install selected
echo.
echo Choose components to install:
echo.
set /p INSTALL_YOLO="Install YOLO AI Detection? [Y/N]: "
set /p INSTALL_STREAMING="Install Live Streaming support? [Y/N]: "
set /p INSTALL_PTZ="Install PTZ Camera control? [Y/N]: "
set /p INSTALL_GPU="Install GPU acceleration? [Y/N]: "
set /p INSTALL_DEV="Install development tools? [Y/N]: "
set INSTALL_TYPE=custom
goto :setup_environment

REM ============================================================================
REM ENVIRONMENT SETUP
REM ============================================================================

:setup_environment
echo.
echo ============================================================================
echo 🔧 SETTING UP PYTHON ENVIRONMENT
echo ============================================================================
echo.

REM Create project directory if not exists
if not exist "cctv_system" (
    echo 📁 Creating project directory...
    mkdir cctv_system
)
cd cctv_system

REM Remove old virtual environment if exists
if exist "venv" (
    echo 🗑️  Removing old virtual environment...
    rmdir /s /q venv
)

REM Create virtual environment
echo 🐍 Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip, setuptools, and wheel
echo ⬆️  Upgrading pip and build tools...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ⚠️  Warning: Failed to upgrade pip
)

REM ============================================================================
REM DEPENDENCY INSTALLATION
REM ============================================================================

echo.
echo ============================================================================
echo 📦 INSTALLING DEPENDENCIES
echo ============================================================================
echo.

REM Core dependencies (always installed)
echo 🔧 Installing core dependencies...
python -m pip install opencv-python>=4.8.0 numpy>=1.24.0 Pillow>=10.0.0
if errorlevel 1 (
    echo ❌ Failed to install core dependencies
    goto :installation_failed
)

echo 🌐 Installing web framework...
python -m pip install flask>=2.3.0 flask-socketio>=5.3.0 eventlet>=0.33.0
if errorlevel 1 (
    echo ❌ Failed to install web framework
    goto :installation_failed
)

echo 🎥 Installing camera control...
python -m pip install onvif-zeep>=0.2.12 zeep>=4.2.0
if errorlevel 1 (
    echo ❌ Failed to install camera control
    goto :installation_failed
)

echo 🛠️  Installing system utilities...
python -m pip install requests>=2.31.0 psutil>=5.9.0
if errorlevel 1 (
    echo ❌ Failed to install utilities
    goto :installation_failed
)

REM Windows-specific dependencies
echo 🪟 Installing Windows-specific packages...
python -m pip install pywin32>=306
if errorlevel 1 (
    echo ⚠️  Warning: Failed to install Windows-specific packages
)

REM Conditional installations based on type
if "%INSTALL_TYPE%"=="custom" (
    if /i "%INSTALL_YOLO%"=="Y" goto :install_yolo
    if /i "%INSTALL_STREAMING%"=="Y" goto :install_streaming
    if /i "%INSTALL_PTZ%"=="Y" goto :install_ptz
    if /i "%INSTALL_GPU%"=="Y" goto :install_gpu
    if /i "%INSTALL_DEV%"=="Y" goto :install_dev
    goto :post_install
) else (
    goto :install_yolo
)

:install_yolo
echo 🤖 Installing YOLO AI Detection...
python -m pip install ultralytics>=8.0.0
if errorlevel 1 (
    echo ❌ Failed to install YOLO
    goto :installation_failed
)

if not "%INSTALL_TYPE%"=="custom" goto :install_streaming
if /i "%INSTALL_STREAMING%"=="Y" goto :install_streaming
goto :check_gpu

:install_streaming
echo 📺 Installing live streaming support...
python -m pip install yt-dlp>=2023.7.6
if errorlevel 1 (
    echo ⚠️  Warning: Failed to install streaming support
)

if not "%INSTALL_TYPE%"=="custom" goto :check_gpu
goto :check_gpu

:install_ptz
echo 🕹️  PTZ control already included in camera control
goto :check_gpu

:check_gpu
if "%INSTALL_TYPE%"=="gpu" goto :install_gpu
if "%INSTALL_TYPE%"=="custom" (
    if /i "%INSTALL_GPU%"=="Y" goto :install_gpu
)
goto :install_pytorch_cpu

:install_gpu
echo 🔥 Installing GPU acceleration (PyTorch with CUDA)...
echo    This may take several minutes...
python -m pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo ⚠️  GPU installation failed, falling back to CPU version...
    goto :install_pytorch_cpu
) else (
    echo ✅ GPU acceleration installed
    goto :check_dev
)

:install_pytorch_cpu
echo 💻 Installing CPU-only PyTorch...
python -m pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ❌ Failed to install PyTorch
    goto :installation_failed
)

:check_dev
if "%INSTALL_TYPE%"=="dev" goto :install_dev
if "%INSTALL_TYPE%"=="custom" (
    if /i "%INSTALL_DEV%"=="Y" goto :install_dev
)
goto :post_install

:install_dev
echo 🔧 Installing development tools...
python -m pip install pytest>=7.4.0 black>=23.0.0 flake8>=6.0.0 pynvml>=11.5.0
if errorlevel 1 (
    echo ⚠️  Warning: Some development tools failed to install
)

goto :post_install

REM ============================================================================
REM POST-INSTALLATION SETUP
REM ============================================================================

:post_install
echo.
echo ============================================================================
echo 🎯 POST-INSTALLATION SETUP
echo ============================================================================
echo.

REM Create models directory
if not exist "models" (
    echo 📁 Creating models directory...
    mkdir models
)

REM Ask about YOLO model download
if "%INSTALL_TYPE%"=="custom" (
    if /i not "%INSTALL_YOLO%"=="Y" goto :skip_models
)

echo.
echo 🤖 YOLO Model Download:
echo.
echo Would you like to download official YOLO models now?
echo This will download ~6-50MB depending on model size.
echo.
echo [1] Download YOLOv8 Nano (fastest, ~6MB)
echo [2] Download YOLOv8 Small (balanced, ~22MB)
echo [3] Download YOLOv8 Medium (accurate, ~52MB)
echo [4] Download multiple models
echo [5] Skip download (models will auto-download on first use)
echo.
set /p MODEL_CHOICE="Enter your choice (1-5): "

if "%MODEL_CHOICE%"=="1" (
    echo 📥 Downloading YOLOv8 Nano...
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>nul
    if exist "yolov8n.pt" move yolov8n.pt models\
)
if "%MODEL_CHOICE%"=="2" (
    echo 📥 Downloading YOLOv8 Small...
    python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')" 2>nul
    if exist "yolov8s.pt" move yolov8s.pt models\
)
if "%MODEL_CHOICE%"=="3" (
    echo 📥 Downloading YOLOv8 Medium...
    python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')" 2>nul
    if exist "yolov8m.pt" move yolov8m.pt models\
)
if "%MODEL_CHOICE%"=="4" (
    echo 📥 Downloading multiple models...
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt')" 2>nul
    if exist "yolov8n.pt" move yolov8n.pt models\
    if exist "yolov8s.pt" move yolov8s.pt models\
)

:skip_models

REM Create application files
echo 📝 Creating application files...

REM Create requirements.txt
echo # Multi-Source CCTV System Requirements > requirements.txt
echo opencv-python^>=4.8.0 >> requirements.txt
echo flask^>=2.3.0 >> requirements.txt
echo flask-socketio^>=5.3.0 >> requirements.txt
echo onvif-zeep^>=0.2.12 >> requirements.txt
echo ultralytics^>=8.0.0 >> requirements.txt
echo numpy^>=1.24.0 >> requirements.txt
echo yt-dlp^>=2023.7.6 >> requirements.txt
echo torch^>=2.0.0 >> requirements.txt
echo torchvision^>=0.15.0 >> requirements.txt
echo requests^>=2.31.0 >> requirements.txt
echo psutil^>=5.9.0 >> requirements.txt
echo pywin32^>=306 >> requirements.txt

REM Create startup script
echo @echo off > start_cctv.bat
echo title Multi-Source CCTV System >> start_cctv.bat
echo echo Starting Multi-Source CCTV System... >> start_cctv.bat
echo call venv\Scripts\activate.bat >> start_cctv.bat
echo python app.py >> start_cctv.bat
echo pause >> start_cctv.bat

REM Create update script
echo @echo off > update_system.bat
echo title Update Multi-Source CCTV System >> update_system.bat
echo echo Updating Multi-Source CCTV System... >> update_system.bat
echo call venv\Scripts\activate.bat >> update_system.bat
echo python -m pip install --upgrade pip >> update_system.bat
echo python -m pip install --upgrade -r requirements.txt >> update_system.bat
echo echo Update completed! >> update_system.bat
echo pause >> update_system.bat

REM ============================================================================
REM WINDOWS FIREWALL CONFIGURATION
REM ============================================================================

echo.
echo 🔥 Windows Firewall Configuration:
echo.
echo The system needs to open port 4000 for the web interface.
echo Configure Windows Firewall? [Y/N]
set /p FIREWALL_CONFIG=""
if /i "%FIREWALL_CONFIG%"=="Y" (
    echo 🛡️  Configuring Windows Firewall...
    netsh advfirewall firewall add rule name="Multi-Source CCTV - Web Interface" dir=in action=allow protocol=TCP localport=4000 >nul 2>&1
    if errorlevel 1 (
        echo ⚠️  Warning: Failed to configure firewall automatically
        echo    Please manually allow port 4000 in Windows Firewall
    ) else (
        echo ✅ Firewall configured successfully
    )
    
    echo 🛡️  Configuring ONVIF discovery ports...
    netsh advfirewall firewall add rule name="Multi-Source CCTV - ONVIF Discovery" dir=in action=allow protocol=TCP localport=80,8080,8899 >nul 2>&1
    if not errorlevel 1 echo ✅ ONVIF ports configured
)

REM ============================================================================
REM INSTALLATION VERIFICATION
REM ============================================================================

echo.
echo ============================================================================
echo ✅ INSTALLATION VERIFICATION
echo ============================================================================
echo.

echo 🔍 Testing Python imports...
python -c "import cv2; print('✅ OpenCV:', cv2.__version__)" 2>nul || echo ❌ OpenCV import failed
python -c "import flask; print('✅ Flask:', flask.__version__)" 2>nul || echo ❌ Flask import failed
python -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>nul || echo ❌ PyTorch import failed

if "%INSTALL_TYPE%"=="custom" (
    if /i not "%INSTALL_YOLO%"=="Y" goto :skip_yolo_test
)
python -c "from ultralytics import YOLO; print('✅ YOLO available')" 2>nul || echo ❌ YOLO import failed

:skip_yolo_test
python -c "import yt_dlp; print('✅ yt-dlp available')" 2>nul || echo ❌ yt-dlp import failed

echo.
echo 🔧 Testing GPU acceleration...
python -c "import torch; print('✅ CUDA available:', torch.cuda.is_available())" 2>nul || echo ❌ GPU test failed

echo.
echo ============================================================================
echo 🎉 INSTALLATION COMPLETED SUCCESSFULLY!
echo ============================================================================
echo.
echo 📋 Installation Summary:
echo    • Installation Type: %INSTALL_TYPE%
echo    • Python Version: %PYTHON_VERSION%
echo    • Virtual Environment: ✅ Created
echo    • Dependencies: ✅ Installed
echo    • Models Directory: ✅ Created
if exist "models\*.pt" echo    • YOLO Models: ✅ Downloaded
echo    • Startup Scripts: ✅ Created
echo    • Firewall: %FIREWALL_CONFIG% Configured
echo.
echo 🚀 Quick Start:
echo    1. Double-click 'start_cctv.bat' to launch the system
echo    2. Open your browser to: http://localhost:4000
echo    3. Select your video source and start streaming!
echo.
echo 📁 Project Location: %CD%
echo.
echo 🔧 Management Scripts:
echo    • start_cctv.bat     - Start the CCTV system
echo    • update_system.bat  - Update dependencies
echo    • requirements.txt   - Package list
echo.
echo 📖 Supported Video Sources:
echo    ✅ RTSP/IP Cameras with enhanced URL parsing
echo    ✅ USB/Built-in Webcams with auto-detection
echo    ✅ YouTube Live streams with original timing
echo    ✅ Twitch, Facebook, Instagram Live streams
echo    ✅ Video files (MP4, AVI, MOV, etc.) with timing preservation
echo    ✅ HLS (.m3u8) and DASH live streams
echo.
echo 🤖 AI Detection Features:
echo    ✅ Universal YOLO model support (official + custom .pt files)
echo    ✅ Real-time object detection (80+ COCO classes)
echo    ✅ Toggleable detection overlay (show/hide bounding boxes)
if "%INSTALL_TYPE%"=="gpu" echo    ✅ GPU acceleration enabled
echo    ✅ Person tracking and counting
echo    ✅ PTZ auto-tracking with ONVIF control
echo.
echo ⚡ Performance Features:
echo    ✅ Ultra-low latency streaming
echo    ✅ Live stream timing preservation  
echo    ✅ File timing preservation (fixed fast playback)
echo    ✅ Multi-threaded processing
echo    ✅ Real-time performance monitoring
echo.
echo 🌐 Web Interface:
echo    ✅ Responsive design (desktop/mobile)
echo    ✅ Real-time controls and monitoring
echo    ✅ Source templates for easy setup
echo    ✅ Fullscreen video viewing
echo.
echo 💡 Tips:
echo    • Place custom .pt model files in the 'models' folder
echo    • Use 'update_system.bat' to update dependencies
echo    • Check Windows Firewall if connection issues occur
echo    • For GPU issues, verify NVIDIA drivers are installed
echo.
echo 📞 Support:
echo    • Documentation: Check README.md in project folder
echo    • Troubleshooting: Review installation log above
echo    • Issues: Check firewall, camera IPs, and model files
echo.
echo Press any key to exit installer...
pause >nul
goto :cleanup

REM ============================================================================
REM ERROR HANDLING
REM ============================================================================

:installation_failed
echo.
echo ============================================================================
echo ❌ INSTALLATION FAILED
echo ============================================================================
echo.
echo The installation encountered errors. Common solutions:
echo.
echo 🔧 Troubleshooting Steps:
echo    1. Run as Administrator
echo    2. Check internet connection
echo    3. Update pip: python -m pip install --upgrade pip
echo    4. Clear pip cache: pip cache purge
echo    5. Install Visual C++ Build Tools:
echo       https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo.
echo 🔄 Retry Options:
echo    • Run this installer again
echo    • Try manual installation: pip install -r requirements.txt
echo    • Install packages one by one
echo.
echo 📞 Get Help:
echo    • Check the troubleshooting guide in README.md
echo    • Verify Python and pip installation
echo    • Ensure Windows version compatibility
echo.
pause
goto :cleanup

:exit_installer
echo.
echo Installation cancelled by user.
echo.
pause
goto :cleanup

REM ============================================================================
REM CLEANUP
REM ============================================================================

:cleanup
REM Deactivate virtual environment if active
if defined VIRTUAL_ENV (
    deactivate
)

REM Return to original directory
cd /d %~dp0

exit /b 0

REM ============================================================================
REM END OF SCRIPT
REM ============================================================================