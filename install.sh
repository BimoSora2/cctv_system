#!/bin/bash

echo "======================================"
echo "🚀 YOLOv8 CCTV System Setup Script"
echo "======================================"
echo ""

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>/dev/null || python --version 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✅ Found: $python_version"
else
    echo "❌ Python not found. Please install Python 3.7+ first."
    exit 1
fi

# Check if pip is available
echo ""
echo "📦 Checking pip availability..."
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    echo "❌ pip not found. Please install pip first."
    exit 1
fi
echo "✅ Found: $PIP_CMD"

# Create virtual environment (recommended)
echo ""
echo "🔧 Setting up virtual environment (recommended)..."
read -p "Do you want to create a virtual environment? (y/N): " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
    source venv/bin/activate  # Linux/Mac
    # For Windows, use: venv\Scripts\activate
    echo "✅ Virtual environment created and activated"
    echo "💡 To activate later: source venv/bin/activate"
else
    echo "⚠️  Installing globally (not recommended for production)"
fi

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
$PIP_CMD install --upgrade pip

# Install core dependencies
echo ""
echo "📦 Installing core dependencies..."
echo "This may take a few minutes for the first time..."

# Install in order of importance
dependencies=(
    "opencv-python>=4.5.0"
    "numpy>=1.19.0"
    "flask>=2.0.0"
    "flask-socketio>=5.0.0"
    "onvif-zeep>=0.2.12"
    "ultralytics>=8.0.0"
    "torch>=1.9.0"
    "torchvision>=0.10.0"
    "yt-dlp>=2023.1.6"
    "Pillow>=8.0.0"
    "requests>=2.25.0"
    "psutil>=5.8.0"
)

for dep in "${dependencies[@]}"; do
    echo ""
    echo "Installing $dep..."
    $PIP_CMD install "$dep"
    if [ $? -eq 0 ]; then
        echo "✅ $dep installed successfully"
    else
        echo "❌ Failed to install $dep"
        echo "💡 Try: $PIP_CMD install $dep --no-cache-dir"
    fi
done

# Install PyTorch with CUDA support (optional but recommended for performance)
echo ""
echo "🔥 GPU Acceleration Setup..."
read -p "Do you want to install PyTorch with CUDA support for GPU acceleration? (y/N): " install_cuda
if [[ $install_cuda =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch with CUDA support..."
    echo "💡 This will download ~2GB of packages"
    
    # Detect CUDA version (basic check)
    if command -v nvidia-smi &> /dev/null; then
        cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
        echo "🔍 Detected CUDA version: $cuda_version"
        
        # Install appropriate PyTorch version
        if [[ $cuda_version == "11."* ]]; then
            $PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        elif [[ $cuda_version == "12."* ]]; then
            $PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        else
            echo "⚠️  Unknown CUDA version, installing CPU version"
            $PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        echo "🚫 NVIDIA GPU not detected, installing CPU version"
        $PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "Installing CPU-only PyTorch (slower but compatible with all systems)"
    $PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Download YOLOv8 models
echo ""
echo "📥 Downloading YOLOv8 models..."
echo "💡 Models will be downloaded automatically on first use, but you can pre-download them:"

models=(
    "yolov8n.pt"
    "yolov8s.pt"
    "yolov8m.pt"
)

read -p "Download YOLOv8 models now? (y/N): " download_models
if [[ $download_models =~ ^[Yy]$ ]]; then
    for model in "${models[@]}"; do
        if [ ! -f "$model" ]; then
            echo "Downloading $model..."
            python3 -c "
from ultralytics import YOLO
try:
    model = YOLO('$model')
    print('✅ $model downloaded successfully')
except Exception as e:
    print('❌ Failed to download $model:', e)
"
        else
            echo "✅ $model already exists"
        fi
    done
else
    echo "⏭️  Models will be downloaded automatically when needed"
fi

# Test yt-dlp with YouTube
echo ""
echo "📺 Testing yt-dlp with YouTube..."
read -p "Test yt-dlp with a sample YouTube video? (y/N): " test_ytdlp
if [[ $test_ytdlp =~ ^[Yy]$ ]]; then
    echo "Testing yt-dlp functionality..."
    python3 -c "
import yt_dlp

try:
    # Test with a known stable video (Rick Roll - always available)
    test_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(test_url, download=False)
        if info:
            print('✅ yt-dlp test successful')
            print(f'   Title: {info.get(\"title\", \"Unknown\")}')
            print(f'   Duration: {info.get(\"duration\", \"Unknown\")} seconds')
            print(f'   View count: {info.get(\"view_count\", \"Unknown\")}')
        else:
            print('❌ yt-dlp test failed: No info extracted')
            
except Exception as e:
    print(f'❌ yt-dlp test failed: {e}')
    print('💡 This might be due to network issues or YouTube blocking')
"
else
    echo "⏭️  yt-dlp test skipped"
fi

# Test installation
echo ""
echo "🧪 Testing installation..."
python3 -c "
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
    
    # Test model initialization
    model = YOLO('yolov8n.pt')
    print('✅ YOLOv8 model test: Success')
    
except ImportError as e:
    print('❌ YOLOv8 failed:', e)
except Exception as e:
    print('⚠️  YOLOv8 warning:', e)
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
    print('⚠️  psutil not available (optional for system monitoring)')

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

echo ""
echo "======================================"
echo "🎉 Installation Complete!"
echo "======================================"
echo ""
echo "📋 Next Steps:"
echo "1. Start your CCTV system:"
echo "   python3 app.py"
echo ""
echo "2. Open your browser to:"
echo "   http://localhost:4000"
echo ""
echo "3. Connect your camera using:"
echo "   • RTSP URL (IP cameras)"
echo "   • USB Webcam (auto-detected)"
echo "   • YouTube Live streams"
echo "   • Video files"
echo ""
echo "💡 Features Available:"
echo "• 🎥 RTSP/IP Camera support with ONVIF PTZ control"
echo "• 📹 USB/Built-in webcam auto-detection"
echo "• 📺 YouTube Live streaming (requires yt-dlp)"
echo "• 🎮 Twitch stream support"
echo "• 📁 Video file playback with timing control"
echo "• 🤖 YOLOv8 AI person detection"
echo "• 🎯 Auto person tracking (PTZ cameras)"
echo "• ⚡ Ultra low-latency streaming"
echo "• 🔴 Live stream timing preservation"
echo ""
echo "🔧 Model Performance Tips:"
echo "• yolov8n.pt: Fastest detection (~30 FPS)"
echo "• yolov8s.pt: Balanced speed/accuracy (~20 FPS)"
echo "• yolov8m.pt: Better accuracy (~15 FPS)"
echo "• yolov8l.pt: High accuracy (~10 FPS)"
echo "• GPU acceleration will be used automatically if available"
echo ""
echo "📺 YouTube Live Usage:"
echo "• Use format: https://www.youtube.com/watch?v=VIDEO_ID"
echo "• Or channel live: https://www.youtube.com/c/CHANNEL/live"
echo "• yt-dlp will extract direct stream URLs automatically"
echo "• System preserves original live timing"
echo ""
echo "🔧 Troubleshooting:"
echo "• If imports fail, try: pip install --no-cache-dir <package>"
echo "• For CUDA issues, install appropriate PyTorch version"
echo "• For camera connection issues, check firewall and network"
echo "• For YouTube issues, ensure yt-dlp is updated: pip install --upgrade yt-dlp"
echo "• If yt-dlp fails, YouTube may be blocking requests - try different streams"
echo ""
echo "📚 Documentation:"
echo "• YOLOv8: https://docs.ultralytics.com/"
echo "• OpenCV: https://docs.opencv.org/"
echo "• ONVIF: https://www.onvif.org/"
echo "• yt-dlp: https://github.com/yt-dlp/yt-dlp"
echo ""
echo "🚀 Ready to start your AI-powered CCTV system!"
echo "======================================"