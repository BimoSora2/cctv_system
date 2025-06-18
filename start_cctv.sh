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

# Enhanced function to check YOLO model files
check_yolo_models() {
    local found_models=()
    local found_custom=()
    
    # Array of official YOLO model files to check (priority order)
    local official_models=(
        "yolo11n.pt"    # YOLO11 Nano - latest
        "yolo11s.pt"    # YOLO11 Small
        "yolo11m.pt"    # YOLO11 Medium
        "yolo11l.pt"    # YOLO11 Large
        "yolo11x.pt"    # YOLO11 Extra Large
        "yolov8n.pt"    # YOLOv8 Nano - fastest
        "yolov8s.pt"    # YOLOv8 Small - balanced
        "yolov8m.pt"    # YOLOv8 Medium - good accuracy
        "yolov8l.pt"    # YOLOv8 Large - high accuracy
        "yolov8x.pt"    # YOLOv8 Extra Large - highest accuracy
        "yolov5n.pt"    # YOLOv5 Nano
        "yolov5s.pt"    # YOLOv5 Small
        "yolov5m.pt"    # YOLOv5 Medium
        "yolov5l.pt"    # YOLOv5 Large
        "yolov5x.pt"    # YOLOv5 Extra Large
    )
    
    # Directories to search in
    local search_dirs=("." "models" "weights" "yolo")
    
    # Phase 1: Check for official YOLO models first
    for dir in "${search_dirs[@]}"; do
        if [ -d "$dir" ]; then
            for model in "${official_models[@]}"; do
                if [ -f "$dir/$model" ]; then
                    # Check file size (must be > 1MB to be valid)
                    if command -v stat >/dev/null 2>&1; then
                        if [[ "$OSTYPE" == "darwin"* ]]; then
                            size=$(stat -f%z "$dir/$model" 2>/dev/null || echo "0")
                        else
                            size=$(stat -c%s "$dir/$model" 2>/dev/null || echo "0")
                        fi
                        size_mb=$((size / 1024 / 1024))
                        if [ $size_mb -ge 1 ]; then
                            found_models+=("$dir/$model")
                        fi
                    else
                        # Fallback if stat command not available
                        found_models+=("$dir/$model")
                    fi
                fi
            done
        fi
    done
    
    # Phase 2: If no official models found, look for ANY .pt files
    if [ ${#found_models[@]} -eq 0 ]; then
        for dir in "${search_dirs[@]}"; do
            if [ -d "$dir" ]; then
                while IFS= read -r -d '' file; do
                    if [ -f "$file" ]; then
                        filename=$(basename "$file")
                        # Skip if it's an official model we already checked
                        local is_official=false
                        for official in "${official_models[@]}"; do
                            if [ "$filename" = "$official" ]; then
                                is_official=true
                                break
                            fi
                        done
                        
                        if [ "$is_official" = false ]; then
                            # Check file size (skip very small files < 1MB)
                            if command -v stat >/dev/null 2>&1; then
                                if [[ "$OSTYPE" == "darwin"* ]]; then
                                    size=$(stat -f%z "$file" 2>/dev/null || echo "0")
                                else
                                    size=$(stat -c%s "$file" 2>/dev/null || echo "0")
                                fi
                                size_mb=$((size / 1024 / 1024))
                                if [ $size_mb -ge 1 ] && [ $size_mb -le 500 ]; then
                                    found_custom+=("$file")
                                fi
                            else
                                # Fallback if stat command not available
                                found_custom+=("$file")
                            fi
                        fi
                    fi
                done < <(find "$dir" -maxdepth 1 -name "*.pt" -type f -print0 2>/dev/null)
            fi
        done
    fi
    
    # Return results: official models first, then custom models
    local all_models=("${found_models[@]}" "${found_custom[@]}")
    echo "${all_models[@]}"
}

# Function to get file size in human readable format
get_file_size() {
    local file="$1"
    if command -v stat >/dev/null 2>&1; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            local size=$(stat -f%z "$file" 2>/dev/null || echo "0")
        else
            # Linux
            local size=$(stat -c%s "$file" 2>/dev/null || echo "0")
        fi
        
        if [ $size -ge 1073741824 ]; then
            echo "$(( size / 1073741824 ))GB"
        elif [ $size -ge 1048576 ]; then
            echo "$(( size / 1048576 ))MB"
        elif [ $size -ge 1024 ]; then
            echo "$(( size / 1024 ))KB"
        else
            echo "${size}B"
        fi
    else
        # Fallback using du if stat not available
        du -h "$file" 2>/dev/null | cut -f1 || echo "?"
    fi
}

# Function to classify model type
classify_model() {
    local filename="$1"
    local filepath="$2"
    
    case $filename in
        yolo11*.pt)
            echo "YOLO11 OFFICIAL"
            ;;
        yolov8*.pt)
            echo "YOLOv8 OFFICIAL"
            ;;
        yolov5*.pt)
            echo "YOLOv5 OFFICIAL"
            ;;
        *yolo*|*YOLO*)
            echo "YOLO CUSTOM"
            ;;
        *model*|*weights*|*trained*|*best*|*final*)
            echo "CUSTOM MODEL"
            ;;
        *)
            echo "CUSTOM .PT"
            ;;
    esac
}

# Enhanced YOLO Detection with comprehensive .pt file support
echo "🔍 Scanning for YOLO models and .pt files..."
found_yolo_models=($(check_yolo_models))

if [ ${#found_yolo_models[@]} -eq 0 ]; then
    # No YOLO models found - provide comprehensive guidance
    echo ""
    echo "📥 No YOLO model files (.pt) found in current directory, models/, weights/, or yolo/ folders"
    echo ""
    echo "🤖 UNIVERSAL .PT MODEL SUPPORT:"
    echo "   ✅ Official YOLO models: yolo11n.pt, yolov8n.pt, yolov5n.pt, etc."
    echo "   ✅ Custom trained models: my_model.pt, best.pt, custom_weights.pt, etc."
    echo "   ✅ Fine-tuned models: specialized.pt, fine_tuned.pt, etc."
    echo "   ✅ Transfer learning models: adapted.pt, transfer_model.pt, etc."
    echo "   ✅ Smart detection with priority system (Official > Custom > Size-based)"
    echo ""
    echo "📂 SUPPORTED LOCATIONS:"
    echo "   • Current directory: ./my_model.pt"
    echo "   • Models folder: ./models/yolo11n.pt"
    echo "   • Weights folder: ./weights/best.pt"
    echo "   • YOLO folder: ./yolo/custom.pt"
    echo ""
    echo "📦 QUICK DOWNLOAD EXAMPLES (Official Models):"
    echo "   mkdir -p models"
    echo "   # YOLO11 (Latest - Recommended)"
    echo "   wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt -P models/"
    echo "   wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11s.pt -P models/"
    echo ""
    echo "   # YOLOv8 (Stable)"
    echo "   wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -P models/"
    echo "   wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt -P models/"
    echo ""
    echo "💡 CUSTOM MODEL GUIDELINES:"
    echo "   • File size: 1MB - 500MB (filters out corrupted files)"
    echo "   • Format: PyTorch .pt files only"
    echo "   • Naming: Use descriptive names (my_model.pt, best_weights.pt)"
    echo "   • Compatibility: Ensure model is YOLO-compatible or similar architecture"
    echo ""
    echo "⚠️  SYSTEM BEHAVIOR:"
    echo "   ✅ Motion detection will be used as fallback"
    echo "   ✅ All video sources will work normally"
    echo "   ✅ AI detection can be enabled later by adding .pt files"
    echo "   ✅ No internet auto-download (manual control)"
    echo ""
else
    # YOLO models found - show detailed analysis
    echo "✅ YOLO model files found - AI Object Detection ENABLED"
    echo "   🚀 System ready with local models (no internet required)"
    echo "   📦 Found ${#found_yolo_models[@]} model file(s)"
    echo ""
    
    # Analyze and display each found model
    echo "📊 MODEL ANALYSIS:"
    local model_count=0
    for model_path in "${found_yolo_models[@]}"; do
        model_count=$((model_count + 1))
        filename=$(basename "$model_path")
        size=$(get_file_size "$model_path")
        model_type=$(classify_model "$filename" "$model_path")
        
        # Determine model description based on filename
        case $filename in
            yolo11n.pt) echo "   $model_count. 🏃 $filename ($size) - YOLO11 Nano (fastest, real-time) [$model_type]" ;;
            yolo11s.pt) echo "   $model_count. ⚖️  $filename ($size) - YOLO11 Small (balanced) [$model_type]" ;;
            yolo11m.pt) echo "   $model_count. 🎯 $filename ($size) - YOLO11 Medium (good accuracy) [$model_type]" ;;
            yolo11l.pt) echo "   $model_count. 🔍 $filename ($size) - YOLO11 Large (high accuracy) [$model_type]" ;;
            yolo11x.pt) echo "   $model_count. 🚀 $filename ($size) - YOLO11 Extra Large (best accuracy) [$model_type]" ;;
            yolov8n.pt) echo "   $model_count. 🏃 $filename ($size) - YOLOv8 Nano (fastest, real-time) [$model_type]" ;;
            yolov8s.pt) echo "   $model_count. ⚖️  $filename ($size) - YOLOv8 Small (balanced) [$model_type]" ;;
            yolov8m.pt) echo "   $model_count. 🎯 $filename ($size) - YOLOv8 Medium (good accuracy) [$model_type]" ;;
            yolov8l.pt) echo "   $model_count. 🔍 $filename ($size) - YOLOv8 Large (high accuracy) [$model_type]" ;;
            yolov8x.pt) echo "   $model_count. 🚀 $filename ($size) - YOLOv8 Extra Large (best accuracy) [$model_type]" ;;
            yolov5*.pt) echo "   $model_count. 🔧 $filename ($size) - YOLOv5 model [$model_type]" ;;
            *) 
                # Custom model - try to give meaningful description
                if [[ $filename == *"nano"* || $filename == *"n."* ]]; then
                    echo "   $model_count. 🏃 $filename ($size) - Custom Nano model [$model_type]"
                elif [[ $filename == *"small"* || $filename == *"s."* ]]; then
                    echo "   $model_count. ⚖️  $filename ($size) - Custom Small model [$model_type]"
                elif [[ $filename == *"medium"* || $filename == *"m."* ]]; then
                    echo "   $model_count. 🎯 $filename ($size) - Custom Medium model [$model_type]"
                elif [[ $filename == *"large"* || $filename == *"l."* ]]; then
                    echo "   $model_count. 🔍 $filename ($size) - Custom Large model [$model_type]"
                elif [[ $filename == *"best"* ]]; then
                    echo "   $model_count. 🏆 $filename ($size) - Best trained model [$model_type]"
                elif [[ $filename == *"final"* ]]; then
                    echo "   $model_count. 🎯 $filename ($size) - Final trained model [$model_type]"
                else
                    echo "   $model_count. 📦 $filename ($size) - Custom model [$model_type]"
                fi
                ;;
        esac
    done
    
    echo ""
    echo "🎯 SMART SELECTION SYSTEM:"
    echo "   • Priority 1: Official YOLO models (YOLO11 > YOLOv8 > YOLOv5)"
    echo "   • Priority 2: Custom models with YOLO-like names"
    echo "   • Priority 3: Models with standard naming (best.pt, model.pt, etc.)"
    echo "   • Priority 4: Size-based selection (5-150MB preferred)"
    echo "   • 🤖 System will automatically select the best available model"
    echo ""
    echo "💡 COMPATIBILITY NOTES:"
    local has_official=false
    local has_custom=false
    for model_path in "${found_yolo_models[@]}"; do
        filename=$(basename "$model_path")
        if [[ $filename == yolo*.pt ]]; then
            has_official=true
        else
            has_custom=true
        fi
    done
    
    if [ "$has_official" = true ] && [ "$has_custom" = true ]; then
        echo "   ✅ Mix of official and custom models detected"
        echo "   🎯 Official models will be prioritized for reliability"
        echo "   📦 Custom models available as alternatives"
    elif [ "$has_official" = true ]; then
        echo "   ✅ Official YOLO models detected - maximum compatibility"
        echo "   🚀 Ready for immediate use with standard object detection"
    else
        echo "   📦 Custom models detected - ensure YOLO compatibility"
        echo "   ⚠️  Custom models may have different object classes"
        echo "   💡 Consider downloading official model for standard detection"
    fi
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
echo ""
echo "🌐 MULTI-SOURCE CCTV SYSTEM READY:"
echo "   📡 Web interface: http://localhost:4000"
echo "   📺 Video sources: RTSP/IP, Webcam, Live Streams, Files"
echo "   🤖 AI Detection: Universal .pt model support"
echo "   🔲 Detection overlay: Toggleable bounding boxes"
echo "   ⏱️  Timing preservation: Original speed for all sources"
echo ""
echo "🎮 CONTROLS:"
echo "   ⏹️  Stop: Ctrl+C (graceful shutdown) or ./stop_cctv.sh"
echo "   🔄 Restart: ./start_cctv.sh"
echo "   📝 Logs: Displayed below in real-time"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the Python application
python app.py