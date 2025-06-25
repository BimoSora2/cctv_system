@echo off
setlocal enabledelayedexpansion

echo ====================================
echo 🎥 Multi-Source CCTV System Launcher
echo ====================================
echo.

:: Check if Python is installed
echo 🐍 Checking Python installation...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python not found! Please install Python 3.7+ first.
    echo   📥 Download from: https://www.python.org/downloads/
    goto :exit_error
) else (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo ✅ Found: !PYTHON_VERSION!
)

:: Check for virtual environment
echo.
echo 🔍 Checking for virtual environment...
if exist "venv\" (
    echo ✅ Virtual environment found.
    
    :: Activate virtual environment
    echo 🔄 Activating virtual environment...
    call venv\Scripts\activate.bat
    
    if !ERRORLEVEL! NEQ 0 (
        echo ⚠️ Failed to activate virtual environment
        echo   💡 Try running: python -m venv venv
        echo   💡 Then restart this script
    ) else (
        echo ✅ Virtual environment activated
    )
) else (
    echo ⚠️ Virtual environment not found
    echo   💡 Create it first: python -m venv venv
    
    choice /C YN /M "Would you like to create a virtual environment now"
    if !ERRORLEVEL! EQU 1 (
        echo 🔄 Creating virtual environment...
        python -m venv venv
        if !ERRORLEVEL! NEQ 0 (
            echo ❌ Failed to create virtual environment
            goto :exit_error
        ) else (
            echo ✅ Virtual environment created
            echo 🔄 Activating virtual environment...
            call venv\Scripts\activate.bat
            if !ERRORLEVEL! NEQ 0 (
                echo ⚠️ Failed to activate virtual environment
            ) else (
                echo ✅ Virtual environment activated
            )
        )
    )
)

:: Quick dependency check
echo.
echo 🧪 Checking basic dependencies...
python -c "import cv2; print('✅ OpenCV available')" 2>nul || echo ⚠️ OpenCV not available - install with: pip install opencv-python
python -c "import flask; print('✅ Flask available')" 2>nul || echo ⚠️ Flask not available - install with: pip install flask
python -c "import numpy; print('✅ NumPy available')" 2>nul || echo ⚠️ NumPy not available - install with: pip install numpy

:: Check for app.py
echo.
echo 🔍 Checking for app.py...
if not exist "app.py" (
    echo ❌ app.py not found in current directory!
    echo   💡 Make sure to run this script from the CCTV system directory
    goto :exit_error
) else (
    echo ✅ Found app.py
)

:: Check for index.html (front-end)
echo 🔍 Checking for index.html...
if not exist "index.html" (
    echo ⚠️ index.html not found in current directory!
    echo   💡 The web interface may not work properly
) else (
    echo ✅ Found index.html
)

:: Check for YOLO models
echo.
echo 🔍 Checking for YOLO model files...
set "MODEL_FOUND=0"
if exist "*.pt" set "MODEL_FOUND=1"
if exist "models\*.pt" set "MODEL_FOUND=1"
if exist "weights\*.pt" set "MODEL_FOUND=1"
if exist "yolo\*.pt" set "MODEL_FOUND=1"

if !MODEL_FOUND! EQU 1 (
    echo ✅ YOLO model files found
) else (
    echo ⚠️ No YOLO model files found
    echo   💡 The system will use motion detection instead of AI detection
    echo   💡 To enable AI detection, download a model like yolov8n.pt
)

echo.
echo 🚀 Starting Multi-Source CCTV System...
echo.
echo 📋 System will start with:
echo   • Web interface available at: http://localhost:4000
echo   • Video stream at: http://localhost:4000/video_feed
echo   • Supports: RTSP/IP Cameras, Webcams, YouTube Streams, Video Files
echo   • Press Ctrl+C to stop the system gracefully
echo.
echo ⏳ Starting server... (This may take a few seconds)
echo ====================================

:: Start the Python application
python app.py

:: Handle exit
:exit
echo.
echo 👋 Thank you for using Multi-Source CCTV System!
goto :end

:exit_error
echo.
echo ❌ Could not start CCTV system
echo 👋 Please fix the issues and try again

:end
endlocal
