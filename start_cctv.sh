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

echo "🎥 Starting CCTV System..."
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

# Test import with timeout protection
echo "🧪 Testing YOLO import..."
timeout 10 python -c "from ultralytics import YOLO; print('✅ YOLO import successful')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ YOLO dependencies verified"
elif [ $? -eq 124 ]; then
    echo "⚠️  YOLO import timeout (10s) - may have CUDA issues"
    echo "   System will continue but YOLO may use CPU fallback"
else
    echo "⚠️  YOLO import failed - object detection may be limited"
    echo "   System will continue with motion detection only"
fi

# Check Python executable
echo "🐍 Using Python: $(which python)"

# Function to check YOLO model files
check_yolo_models() {
    local models_dir=""
    local found_models=()
    
    # Array of YOLO model files to check
    local yolo_models=(
        "yolov8n.pt"    # Nano - fastest
        "yolov8s.pt"    # Small - balanced
        "yolov8m.pt"    # Medium - good accuracy
        "yolov8l.pt"    # Large - high accuracy
        "yolov8x.pt"    # Extra Large - highest accuracy
        "yolo11n.pt"    # YOLO11 Nano
        "yolo11s.pt"    # YOLO11 Small
        "yolo11m.pt"    # YOLO11 Medium
        "yolo11l.pt"    # YOLO11 Large
        "yolo11x.pt"    # YOLO11 Extra Large
    )
    
    # Check each model file
    for model in "${yolo_models[@]}"; do
        if [ -f "$models_dir/$model" ]; then
            found_models+=("$model")
        fi
    done
    
    # Return found models
    echo "${found_models[@]}"
}

# Modern YOLOv8 Detection - Only check for model files (.pt)
echo "🔍 Checking YOLO model files..."
found_yolo_models=($(check_yolo_models))

if [ ${#found_yolo_models[@]} -eq 0 ]; then
    # No local YOLO models found - but that's OK!
    echo "📥 No local YOLO model files found"
    echo "   ✅ YOLOv8 will auto-download yolov8n.pt (6MB) on first use"
    echo "   🌐 Internet connection required for initial download"
    echo "   ⚡ After download, models are cached locally for offline use"
    echo ""
    echo "📦 Available YOLOv8 models for auto-download:"
    echo "   🏃 yolov8n.pt (~6MB)  - Nano (fastest, good for real-time)"
    echo "   ⚖️  yolov8s.pt (~22MB) - Small (balanced speed/accuracy)"
    echo "   🎯 yolov8m.pt (~52MB) - Medium (better accuracy)"
    echo "   🔍 yolov8l.pt (~88MB) - Large (high accuracy)"
    echo "   🚀 yolov8x.pt (~136MB)- Extra Large (highest accuracy)"
    echo ""
    echo "✅ Object detection ENABLED - auto-download mode"
    echo ""
else
    # Local YOLO models found
    echo "✅ Local YOLO model files found - Object detection ENABLED"
    echo "   🚀 Using local models (no internet required)"
    echo "   📦 Found models: ${found_yolo_models[*]}"
    echo ""
    
    # Show detailed model info with file sizes
    for model in "${found_yolo_models[@]}"; do
        if [ -f "models/$model" ]; then
            size=$(du -h "models/$model" 2>/dev/null | cut -f1 || echo "?")
            case $model in
                "yolov8n.pt") echo "      🏃 $model ($size) - Nano (fastest, real-time)" ;;
                "yolov8s.pt") echo "      ⚖️  $model ($size) - Small (balanced)" ;;
                "yolov8m.pt") echo "      🎯 $model ($size) - Medium (good accuracy)" ;;
                "yolov8l.pt") echo "      🔍 $model ($size) - Large (high accuracy)" ;;
                "yolov8x.pt") echo "      🚀 $model ($size) - Extra Large (best accuracy)" ;;
                "yolo11n.pt") echo "      🏃 $model ($size) - YOLO11 Nano (latest)" ;;
                "yolo11s.pt") echo "      ⚖️  $model ($size) - YOLO11 Small (latest)" ;;
                "yolo11m.pt") echo "      🎯 $model ($size) - YOLO11 Medium (latest)" ;;
                "yolo11l.pt") echo "      🔍 $model ($size) - YOLO11 Large (latest)" ;;
                "yolo11x.pt") echo "      🚀 $model ($size) - YOLO11 Extra Large (latest)" ;;
                *) echo "      📦 $model ($size) - Custom model" ;;
            esac
        fi
    done
    echo ""
    echo "💡 Tip: System will automatically select the best available model"
    echo ""
fi

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
echo "🌐 Web interface: http://localhost:4000"
echo "⏹️  Stop with: Ctrl+C (graceful shutdown) or ./stop_cctv.sh"
echo "📝 Logs will be displayed below..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the Python application
python app.py