#!/bin/bash

# Function to handle graceful shutdown
cleanup() {
    echo ""
    echo "🛑 KeyboardInterrupt detected (Ctrl+C pressed)"
    echo "🔄 Gracefully shutting down CCTV system..."
    echo "📹 Stopping camera feeds..."
    echo "🌐 Closing web server..."
    echo "🔒 Deactivating virtual environment..."
    echo "✅ CCTV system stopped successfully"
    echo "👋 Thank you for using CCTV System!"
    exit 0
}

# Set up signal trap for Ctrl+C (SIGINT)
trap cleanup SIGINT

echo "🎥 Starting Multi-Source CCTV System..."
echo "💡 Press Ctrl+C to stop the system gracefully"
cd ~/cctv_system

# Pastikan virtual environment ada dan aktif
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment 'venv' not found!"
    echo "   Please run: python3 -m venv venv"
    exit 1
fi

echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Verify activation
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Failed to activate virtual environment"
    echo "   Try: source venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Quick dependency check (without heavy imports)
echo "🧪 Checking basic dependencies..."
python -c "import cv2; print('✅ OpenCV available')" 2>/dev/null || echo "⚠️  OpenCV not available - install with: pip install opencv-python"
python -c "import flask; print('✅ Flask available')" 2>/dev/null || echo "⚠️  Flask not available - install with: pip install flask"
python -c "import numpy; print('✅ NumPy available')" 2>/dev/null || echo "⚠️  NumPy not available - install with: pip install numpy"

# Check Python executable
echo "🐍 Using Python: $(which python)"

# Quick YOLO availability check (without importing)
echo "🤖 YOLO Detection:"
if python -c "import importlib.util; exit(0 if importlib.util.find_spec('ultralytics') else 1)" 2>/dev/null; then
    echo "   ✅ ultralytics package available"
    echo "   🤖 AI Detection will be handled by app.py"
else
    echo "   ⚠️  ultralytics not installed"
    echo "   💡 Install with: pip install ultralytics"
    echo "   🔄 Motion detection will be used as fallback"
fi

# Check for .pt model files (simple check)
echo "📦 Model Detection:"
model_count=0
for dir in "." "models" "weights" "yolo"; do
    if [ -d "$dir" ]; then
        pt_files=$(find "$dir" -maxdepth 1 -name "*.pt" -type f 2>/dev/null | wc -l)
        if [ "$pt_files" -gt 0 ]; then
            model_count=$((model_count + pt_files))
        fi
    fi
done

if [ "$model_count" -gt 0 ]; then
    echo "   ✅ Found $model_count .pt model file(s)"
    echo "   🚀 AI object detection available"
else
    echo "   📁 No .pt model files found"
    echo "   💡 Place any .pt file in ./models/ ./weights/ or current directory"
    echo "   🔄 Motion detection will be used as fallback"
fi

echo "   🎯 Full model analysis will be done by app.py"

# Cek app.py (wajib)
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found!"
    echo "   Please copy app.py to ~/cctv_system/"
    exit 1
fi

# Cek templates (wajib)
if [ ! -f "index.html" ]; then
    echo "❌ index.html not found!"
    echo "   Please copy index.html to ~/cctv_system/"
    exit 1
fi

echo "✅ Core files found. Starting server..."
echo ""
echo "🌐 MULTI-SOURCE CCTV SYSTEM READY:"
echo "   📡 Web interface: http://localhost:4000"
echo "   📺 Video sources: RTSP/IP, Webcam, Live Streams, Files"
echo "   🤖 AI Detection: Universal .pt model support"
echo "   🔲 Detection overlay: Toggleable bounding boxes"
echo "   ⏱️  Timing preservation: Original speed for all sources"
echo "   📹 Enhanced RTSP: All standard URL formats with auto IP extraction"
echo ""
echo "🎮 CONTROLS:"
echo "   ⏹️  Stop: Ctrl+C (graceful shutdown) or ./stop_cctv.sh"
echo "   🔄 Restart: ./start_cctv.sh"
echo "   📝 Logs: Displayed below in real-time"
echo ""
echo "💡 FEATURES:"
echo "   🔍 Smart model detection: Official YOLO + Custom .pt files"
echo "   📡 Enhanced RTSP: Auto-credential extraction, IP detection"
echo "   🔴 Live streams: YouTube, Twitch, HLS with original timing"
echo "   📁 File support: All formats with proper timing preservation"
echo "   🔲 Detection overlay: Hide/show bounding boxes independently"
echo "   🕹️  PTZ control: ONVIF auto-discovery for supported cameras"
echo ""
echo "🚀 QUICK START:"
echo "   1. Open http://localhost:4000 in your browser"
echo "   2. Select video source type (RTSP, Webcam, Stream, File)"
echo "   3. Configure source settings"
echo "   4. Click 'Connect to Source'"
echo "   5. Enjoy AI-powered video monitoring!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the Python application
python app.py
