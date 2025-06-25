# 🚀 Multi-Source CCTV System with Universal AI Detection

## 📖 Overview

A comprehensive, real-time video surveillance system that supports multiple video sources with AI-powered object detection. Built with Python Flask, OpenCV, and Universal YOLO model support, featuring ultra-low latency streaming, enhanced RTSP parsing, PTZ camera control, and toggleable detection overlays.

## ✨ Key Features

### 🤖 Universal AI Detection System
- **Official YOLO Models**: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
- **YOLOv8 Series**: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt  
- **YOLOv5 Series**: yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt
- **Custom .pt Models**: Supports ANY PyTorch .pt file (my_model.pt, best.pt, custom_weights.pt, etc.)
- **Smart Model Detection**: Auto-finds and prioritizes best available models with intelligent ranking
- **Cross-Platform Discovery**: Works on Windows, Linux, macOS with multiple search locations
- **Size Validation**: 1MB-500MB filtering to exclude corrupted/invalid files
- **Pattern Matching**: Prioritizes YOLO-like named files and model-related patterns
- **GPU Acceleration**: CUDA support when available with automatic fallback to CPU
- **Real-time Object Detection**: 80+ COCO classes with person tracking and counting
- **🔲 Toggleable Detection Overlay**: Show/hide bounding boxes while keeping detection active

### 📡 Enhanced Multi-Source Video Support
- **🎥 RTSP/IP Cameras** with revolutionary URL parsing
  - **Universal Format Support**: All standard RTSP URL formats
  - **Smart Credential Extraction**: Auto-parses username/password from URLs
  - **Intelligent IP Detection**: Extracts camera IP from ANY URL format for ONVIF PTZ
  - **Real-time Error Notifications**: Enhanced RTSP connection error alerts
  - **Brand-Specific Templates**: Pre-configured for Hikvision, Dahua, Axis, etc.
  - **Fallback Authentication**: Uses separate credential fields if URL doesn't contain them
- **📹 USB/Built-in Webcams** with intelligent auto-detection
  - **Device Information Display**: Shows resolution, FPS, and device names
  - **Real-time Scanning**: Refresh webcam list without restart
  - **Cross-Platform Support**: Works on Windows (DirectShow), Linux (V4L2), macOS
- **📺 Live Streaming URLs** with original timing preservation
  - **YouTube Live**: Real-time live stream processing with yt-dlp integration
  - **Twitch Streams**: Native Twitch live stream support
  - **HLS Live Streams**: .m3u8 live streams with proper timing
  - **Facebook Live**: Facebook live stream integration
  - **Instagram Live**: Instagram live stream support
  - **DASH Adaptive Streams**: .mpd adaptive streaming
- **📁 Video Files** with fixed timing preservation
  - **Auto-Detection**: Distinguishes between file URLs and live stream URLs
  - **Original Speed Preservation**: Maintains natural playbook speed (FIXED fast playback)
  - **File Format Support**: MP4, AVI, MOV, MKV, FLV, WMV, WebM, M4V, 3GP, OGV, TS, MTS, M2TS
  - **Loop Functionality**: Continuous video looping for testing
  - **Network Files**: HTTP/HTTPS file URLs with proper buffering
- **🌐 HTTP/HTTPS Direct Streams** with adaptive handling

### ⏱️ Revolutionary Timing Preservation System
- **🔴 Live Stream Mode**: Preserves original FPS and timing for live streams
  - **Auto-Detection**: Identifies live streams from URL patterns and content
  - **Minimal Buffering**: Ultra-low latency for real-time streaming
  - **Timing Drift Detection**: Monitors and corrects timing inconsistencies
  - **Frame Timestamp Buffer**: Maintains timing accuracy over extended periods
- **📁 File Mode**: Maintains natural playback speed for video files
  - **Original FPS Detection**: Reads and preserves source frame rate
  - **Speed Correction**: Fixes fast playback issues common in video processing
  - **Buffer Management**: Optimized buffering strategies per file type
- **🎯 Smart Source Detection**: Automatically categorizes URLs as live streams or files
- **⚙️ Adaptive Timing**: Different strategies for each source type (RTSP, webcam, stream, file)
- **🔧 User Controls**: Override auto-detection with manual timing preferences

### 🔒 Revolutionary RTSP Support
- **🧠 Smart URL Parsing**: Handles ALL standard RTSP URL formats
  - `rtsp://192.168.1.4:554/live/ch00_0` (No authentication)
  - `rtsp://admin:@192.168.1.4:554/live/ch00_0` (Username only)
  - `rtsp://admin:admin@192.168.1.4:554/live/ch00_0` (Full authentication)
  - `rtsp://user:pass@192.168.1.4:8080/path/stream` (Custom ports and paths)
- **🔍 Auto-Credential Extraction**: Intelligent parsing of embedded credentials
- **🌐 Universal IP Detection**: Extracts camera IP from ANY URL format for ONVIF PTZ
  - Works with HTTP, HTTPS, RTSP, or plain IP formats
  - Real-time IP validation (0-255 octet range checking)
  - Auto-fills camera IP field as you type URLs
- **⚠️ Real-time Error Notifications**: Enhanced RTSP connection error alerts
- **📋 Brand Templates**: Pre-configured URL formats for popular camera brands
  - Hikvision: `rtsp://admin:admin@IP:554/Streaming/Channels/101`
  - Dahua: `rtsp://admin:admin@IP:554/cam/realmonitor?channel=1&subtype=0`
  - Axis: `rtsp://admin:admin@IP:554/axis-media/media.amp`
- **🔄 Fallback Authentication**: Uses separate credential fields when URL parsing fails

### 🕹️ Advanced PTZ Camera Control (RTSP/ONVIF Only)
- **🔍 Smart ONVIF Discovery**: Multi-port, multi-authentication discovery system
  - **Port Scanning**: Tests ports 80, 8080, 8899, 8000, 554, 9999
  - **Authentication Matrix**: Tests no auth, admin/admin, admin/(blank), user-provided
  - **Timeout Management**: Configurable timeouts per connection attempt
  - **Device Information**: Retrieves manufacturer, model, and capabilities
- **🎮 Full PTZ Control**: Pan, Tilt, Zoom, and Home position
- **🎯 AI-Powered Person Tracking**: Automatic person following with YOLO/Motion detection
- **⚙️ Adjustable Parameters**: 
  - Tracking sensitivity (pixel threshold for movement)
  - PTZ movement speed multiplier
  - Tracking cooldown period
- **🏠 Home Position**: Quick return to preset position
- **🔄 Continuous Movement**: Smooth PTZ operations with proper stop commands

### ⚡ Ultra-Low Latency Architecture
- **🧵 Threaded Design**: Separated capture, detection, and streaming threads
- **📊 Minimal Buffering**: Source-specific buffer optimization
  - RTSP: 1-frame buffer for real-time performance
  - Live Streams: Minimal buffering for low latency
  - Files: Adaptive buffering based on file type and size
- **🎯 Frame Queue Management**: Intelligent frame dropping to maintain real-time performance
- **📈 Performance Monitoring**: Real-time FPS, latency, and resource tracking
- **🔧 Dynamic Quality**: Adaptive quality based on performance requirements

### 🌐 Modern Responsive Web Interface
- **📱 Mobile-Optimized**: Works perfectly on smartphones and tablets
- **🎛️ Real-time Controls**: Live adjustment of all parameters without restart
- **📋 Source Templates**: Pre-configured settings for popular cameras and services
- **🖥️ Fullscreen Support**: Immersive video viewing with escape key support
- **📊 Performance Dashboard**: Live system monitoring and statistics
- **🎨 Modern UI**: Clean, intuitive interface with source-specific feature visibility
- **🔄 Tab-Based Organization**: Separate controls for each source type
- **⚡ WebSocket Updates**: Real-time status updates without page refresh

### 📺 OBS Studio Integration
- **🔗 Stream URL Generation**: Automatic OBS-compatible stream URL creation
- **📋 One-Click Copy**: Copy stream URL directly to clipboard
- **📖 Setup Instructions**: Built-in guide for OBS configuration
- **🎥 Media Source Compatible**: Works as OBS Media Source input
- **⚙️ Network Buffering**: Optimized for OBS streaming requirements

## 🛠️ Installation

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended for multiple cameras
- **Storage**: 2GB free space (for models and dependencies)
- **Network**: Internet connection for model downloads and streaming
- **GPU** (Optional): NVIDIA GPU with CUDA for AI acceleration

### 🐧 Linux Installation (Ubuntu/Debian/CentOS/Arch)

#### Automated Installation Script
```bash
# Download and run automated installer
wget https://raw.githubusercontent.com/your-repo/multi-source-cctv/main/install.sh
chmod +x install.sh
./install.sh

# The installer will:
# ✅ Detect your Linux distribution
# ✅ Install system dependencies
# ✅ Create virtual environment
# ✅ Install Python packages
# ✅ Configure permissions
# ✅ Download YOLO models (optional)
```

#### Manual Installation
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and system dependencies
sudo apt install python3 python3-pip python3-venv python3-opencv -y
sudo apt install libopencv-dev libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 -y
sudo apt install ffmpeg libavcodec-extra -y

# Create virtual environment
python3 -m venv cctv_env
source cctv_env/bin/activate

# Install Python packages
pip install --upgrade pip
pip install opencv-python>=4.8.0 flask>=2.3.0 flask-socketio>=5.3.0
pip install onvif-zeep>=0.2.12 numpy>=1.24.0 ultralytics>=8.0.0
pip install yt-dlp>=2023.7.6 psutil>=5.9.0 requests>=2.31.0

# Optional: GPU acceleration (NVIDIA CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Start the application
python app.py
```

### 🪟 Windows Installation

#### Method 1: Automated Installer (Recommended)
```cmd
Download install.bat from repository
Right-click install.bat → "Run as administrator"
Choose installation type:
  🚀 Quick Install (CPU-only)
  🔥 GPU Install (NVIDIA CUDA)
  🔧 Development Install
  🎯 Custom Install

The installer automatically:
✅ Checks system requirements
✅ Creates virtual environment  
✅ Installs all dependencies
✅ Configures Windows Firewall
✅ Downloads YOLO models
✅ Creates startup scripts
✅ Configures DirectShow webcam support
```

#### Method 2: Manual Installation
```cmd
# Download Python 3.8+ from python.org
# ✅ Check "Add Python to PATH" during installation

# Open Command Prompt as Administrator
python -m pip install --upgrade pip

# Create virtual environment
python -m venv cctv_env
cctv_env\Scripts\activate

# Install core dependencies
pip install opencv-python>=4.8.0 flask>=2.3.0 flask-socketio>=5.3.0
pip install onvif-zeep>=0.2.12 numpy>=1.24.0 ultralytics>=8.0.0
pip install yt-dlp>=2023.7.6 pywin32>=306 eventlet>=0.33.0

# For GPU acceleration (NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Configure Windows Firewall
netsh advfirewall firewall add rule name="Multi-Source CCTV" dir=in action=allow protocol=TCP localport=4000
```

### 🚀 Quick Start Guide

#### Step 1: Launch Application
```bash
# Linux
source cctv_env/bin/activate
python app.py

# Windows  
cctv_env\Scripts\activate
python app.py

# Or use startup scripts (created by installer)
./start_cctv.sh    # Linux
start_cctv.bat     # Windows

# Access web interface
Open browser: http://localhost:4000
```

#### Step 2: Test with Webcam (Easiest)
```
1. Click "📹 Webcam" tab
2. Click "🔄 Refresh Webcam List"  
3. Select webcam from detected list
4. Click "🚀 Connect to Source"
5. Toggle "🤖 AI Detection" ON
6. You should see video with person detection!
```

#### Step 3: Connect IP Camera (RTSP)
```
1. Click "📹 RTSP/IP" tab
2. Use template or enter RTSP URL:
   • Basic: rtsp://192.168.1.100:554/stream1
   • With Auth: rtsp://admin:password@192.168.1.100:554/stream1
   • System auto-extracts IP for PTZ control
3. Click "🚀 Connect to Source"
4. Enable "🤖 AI Detection" and "🎯 Auto Tracking"
```

#### Step 4: Try Live Streaming
```
1. Click "📺 Live Stream" tab
2. Use template or enter URL:
   • YouTube Live: https://www.youtube.com/watch?v=LIVE_VIDEO_ID
   • Twitch: https://www.twitch.tv/CHANNEL_NAME
   • HLS: https://example.com/live/stream.m3u8
3. Enable "⏱️ Preserve Original Timing"
4. Click "🚀 Connect to Source"
```

#### Step 5: Load Video File
```
1. Click "📁 File/URL" tab
2. Enter file path or URL:
   • Local: C:\Videos\sample.mp4
   • Network: http://example.com/video.mp4
3. Enable "⏱️ Preserve Original Speed" (recommended)
4. Choose buffer size and loop options
5. Click "🚀 Connect to Source"
```

## 🤖 Universal AI Model Support

### Official YOLO Models (Auto-Download)
The system automatically detects and downloads official YOLO models:

**YOLO11 Series (Latest):**
- `yolo11n.pt` - Nano (6MB, fastest)
- `yolo11s.pt` - Small (22MB, balanced)  
- `yolo11m.pt` - Medium (50MB, accurate)
- `yolo11l.pt` - Large (100MB, very accurate)
- `yolo11x.pt` - Extra Large (166MB, most accurate)

**YOLOv8 Series (Popular):**
- `yolov8n.pt` - Nano (6MB, fastest)
- `yolov8s.pt` - Small (22MB, balanced)
- `yolov8m.pt` - Medium (50MB, accurate)  
- `yolov8l.pt` - Large (100MB, very accurate)
- `yolov8x.pt` - Extra Large (166MB, most accurate)

**YOLOv5 Series (Stable):**
- `yolov5n.pt` - Nano (4MB, fastest)
- `yolov5s.pt` - Small (14MB, balanced)
- `yolov5m.pt` - Medium (42MB, accurate)
- `yolov5l.pt` - Large (93MB, very accurate)  
- `yolov5x.pt` - Extra Large (166MB, most accurate)

### Custom Model Support
The system supports ANY PyTorch .pt model file:

**Custom Model Examples:**
- `my_model.pt` - Your custom trained model
- `best.pt` - Best checkpoint from training
- `custom_weights.pt` - Fine-tuned weights
- `security_model.pt` - Security-specific model
- `person_detector.pt` - Person-only detection model

### Smart Model Discovery System

**Priority Ranking:**
1. **Official YOLO models** (highest priority)
2. **YOLO-pattern files** (contains 'yolo', 'ultralytics', 'detection')
3. **Model-pattern files** (contains 'model', 'weights', 'trained', 'best')
4. **Size-appropriate files** (5-150MB range)
5. **Small models** (1-5MB, likely nano variants)
6. **Large models** (150-500MB, custom trained)
7. **Any other .pt files**

**Search Locations:**
```
Current Working Directory:
├── ./
├── ./models/
├── ./weights/
├── ./yolo/
├── ./ultralytics/

Script Directory:
├── [script_path]/
├── [script_path]/models/
├── [script_path]/weights/

User Cache:
├── ~/.cache/ultralytics/
├── ~/.ultralytics/

System Locations:
├── /usr/local/share/ultralytics/     (Linux)
├── /opt/ultralytics/                 (Linux)
├── C:/ProgramData/ultralytics/       (Windows)
├── %LOCALAPPDATA%/ultralytics/       (Windows)
```

**Size and Pattern Filtering:**
- ✅ **Size Range**: 1MB - 500MB (excludes corrupted files)
- ✅ **Pattern Matching**: Prioritizes YOLO-like file names
- ✅ **Validation**: Checks file integrity before loading
- ✅ **Automatic Selection**: Chooses best available model

### Model Performance Guide

**For Real-time Performance (Recommended):**
- Use **Nano models** (yolo11n.pt, yolov8n.pt) for fastest detection
- Input Size: 320×320 pixels
- Confidence: 30-40%

**For Balanced Performance:**
- Use **Small models** (yolo11s.pt, yolov8s.pt) for good accuracy
- Input Size: 416×416 pixels  
- Confidence: 40-50%

**For Maximum Accuracy:**
- Use **Medium/Large models** (yolo11m.pt, yolov8m.pt) for best detection
- Input Size: 640×640 pixels
- Confidence: 50-60%

### GPU Acceleration Support

**NVIDIA CUDA:**
```bash
# Check CUDA availability
nvidia-smi

# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Performance Comparison:**
- **CPU (Intel i7)**: ~5-10 FPS with nano models
- **GPU (RTX 3060)**: ~25-30 FPS with nano models  
- **GPU (RTX 4080)**: ~50+ FPS with medium models

## 🔧 Advanced Configuration

### RTSP Camera Setup Examples

**Basic RTSP (No Authentication):**
```
rtsp://192.168.1.100:554/stream1
rtsp://192.168.1.100:554/live/ch00_0
rtsp://192.168.1.100:8554/stream
```

**RTSP with Authentication in URL:**
```
rtsp://admin:password@192.168.1.100:554/stream1
rtsp://user:pass123@192.168.1.100:8080/live/main
rtsp://admin:@192.168.1.100:554/stream (username only)
```

**Brand-Specific RTSP URLs:**
```python
# Hikvision IP Cameras
rtsp://admin:admin@192.168.1.100:554/Streaming/Channels/101
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/102

# Dahua IP Cameras  
rtsp://admin:admin@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1

# Axis IP Cameras
rtsp://admin:admin@192.168.1.100:554/axis-media/media.amp
rtsp://root:password@192.168.1.100:554/axis-media/media.amp?videocodec=h264

# Generic IP Cameras
rtsp://admin:admin@192.168.1.100:554/live0.264
rtsp://admin:admin@192.168.1.100:554/stream/0
```

### Live Stream URL Examples

**YouTube Live Streams:**
```
https://www.youtube.com/watch?v=jfKfPfyJRdk
https://www.youtube.com/watch?v=LIVE_VIDEO_ID
```

**Twitch Live Streams:**
```
https://www.twitch.tv/nasa
https://www.twitch.tv/CHANNEL_NAME
```

**HLS Live Streams (.m3u8):**
```
https://cph-p2p-msl.akamaized.net/hls/live/2000341/test/master.m3u8
https://example.com/live/stream.m3u8
```

**Facebook Live:**
```
https://www.facebook.com/username/live
https://www.facebook.com/watch/live/?v=VIDEO_ID
```

**Instagram Live:**
```
https://www.instagram.com/username/live
```

**DASH Streams (.mpd):**
```
https://dash.akamaized.net/akamai/streamroot/cenc/srcs/drmfree_video_only.mpd
```

### File Source Examples

**Local Video Files:**
```python
# Windows
C:\Videos\sample.mp4
D:\CCTV_Recordings\camera1_20231201.avi

# Linux/macOS  
/home/user/videos/sample.mp4
/media/recordings/camera1.mov
```

**Network Video Files:**
```python
# HTTP/HTTPS URLs
http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4
https://sample-videos.com/zip/10/mp4/SampleVideo_720x480_1mb.mp4

# FTP URLs (if accessible)
ftp://demo.server.com/path/video.mp4
```

**Supported Video Formats:**
- **Container Formats**: MP4, AVI, MOV, MKV, FLV, WMV, WebM, M4V, 3GP, OGV
- **Streaming Formats**: TS, MTS, M2TS, MPEGTS
- **Codecs**: H.264, H.265/HEVC, VP8, VP9, MPEG-4, MJPEG

## 📊 Detection Features & Capabilities

### Object Detection Classes (80+ COCO Objects)
The system can detect and track these object classes:

**Person & Vehicles:**
- Person, Bicycle, Car, Motorcycle, Airplane, Bus, Train, Truck, Boat

**Traffic & Outdoor:**
- Traffic Light, Fire Hydrant, Stop Sign, Parking Meter, Bench

**Animals:**
- Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe

**Sports & Recreation:**
- Frisbee, Skis, Snowboard, Sports Ball, Kite, Baseball Bat, Baseball Glove, Skateboard, Surfboard, Tennis Racket

**Food & Kitchen:**
- Bottle, Wine Glass, Cup, Fork, Knife, Spoon, Bowl, Banana, Apple, Sandwich, Orange, Broccoli, Carrot, Hot Dog, Pizza, Donut, Cake

**Furniture & Electronics:**
- Chair, Couch, Potted Plant, Bed, Dining Table, Toilet, TV, Laptop, Mouse, Remote, Keyboard, Cell Phone, Microwave, Oven, Toaster, Sink, Refrigerator

**Personal Items:**
- Backpack, Umbrella, Handbag, Tie, Suitcase, Book, Clock, Vase, Scissors, Teddy Bear, Hair Drier, Toothbrush

### Detection Visualization

**Golden Bounding Boxes:**
- High-contrast golden color (#00D7FF in BGR) for maximum visibility
- Thick 3-pixel borders for clear object boundaries
- Rounded corners for modern appearance

**Smart Labels:**
- Object class name with confidence percentage
- Background color matching bounding box
- White text for optimal contrast
- Auto-positioning to avoid frame edges

**Toggleable Overlay System:**
- **Show Mode**: Display all bounding boxes and labels
- **Hide Mode**: Clean video view while detection continues in background  
- **Independent Control**: Toggle visibility without affecting detection performance
- **Real-time Toggle**: Change mode without disconnecting source

### Performance Optimization

**Detection Settings:**
- **Confidence Threshold**: 10-90% (default: 40%)
- **Input Resolution**: 320×320, 416×416, 640×640 pixels
- **Detection FPS**: Independent from stream FPS for optimal performance
- **IoU Threshold**: 0.3 (automatically configured)

**Threading Architecture:**
- **Capture Thread**: Dedicated video frame acquisition
- **Detection Thread**: AI processing and object detection  
- **Streaming Thread**: Web interface video delivery
- **PTZ Thread**: Camera movement control (RTSP only)

## 🕹️ PTZ Control & Auto-Tracking

### ONVIF Discovery Process

**Multi-Port Discovery:**
The system intelligently scans multiple ports to find ONVIF services:
- **Port 80**: Standard HTTP (most common)
- **Port 8080**: Alternative HTTP
- **Port 8899**: NVR/Recorder ONVIF  
- **Port 8000**: Hikvision style
- **Port 554**: RTSP port (some cameras)
- **Port 9999**: Dahua style

**Authentication Matrix:**
For each port, tries multiple authentication methods:
1. **No Authentication**: Anonymous access
2. **admin/(blank)**: Admin with no password
3. **admin/admin**: Common default credentials
4. **User Provided**: Credentials from RTSP URL or form fields

**Discovery Results:**
- **Full PTZ**: Pan, Tilt, Zoom, Home, and Auto-tracking available
- **Limited PTZ**: Basic movement without advanced features
- **Basic ONVIF**: Device info only, no movement control
- **Not Available**: No ONVIF support detected

### PTZ Movement Controls

**Manual Control:**
- **Pan**: Left/Right movement (⬅️ ➡️)
- **Tilt**: Up/Down movement (⬆️ ⬇️)  
- **Zoom**: In/Out (🔍+ 🔍-)
- **Home**: Return to preset position (🏠)

**Auto-Tracking Features:**
- **AI-Powered**: Uses YOLO person detection for tracking
- **Motion Fallback**: Uses motion detection if YOLO unavailable
- **Tracking Sensitivity**: 20-200 pixel threshold (default: 80px)
- **Movement Speed**: 0.1x to 2.0x multiplier (default: 1.0x)
- **Cooldown Period**: 1-second minimum between movements
- **Frame Center Targeting**: Keeps person centered in view

**Smart Tracking Logic:**
- **Person Priority**: Tracks closest person to last known position
- **Smooth Movement**: Gradual PTZ adjustments to avoid jarring motion
- **Boundary Respect**: Stays within camera's mechanical limits
- **Auto-Disable**: Stops tracking if no person detected for 10 seconds

### PTZ Performance Optimization

**Speed Settings:**
- **Slow**: 0.1-0.3x for precise positioning
- **Normal**: 0.5-1.0x for balanced tracking
- **Fast**: 1.5-2.0x for quick movements

**Network Optimization:**
- **Timeout Controls**: 3-second connection timeout
- **Command Queuing**: Prevents command overlap
- **Status Monitoring**: Real-time PTZ position feedback

## ⚡ Performance & Optimization

### Ultra-Low Latency Architecture

**Threaded Design:**
```python
# Main Threads
📹 Frame Capture Thread    # Video source acquisition
🤖 Detection Thread       # AI processing (separate)  
📺 Streaming Thread       # Web interface delivery
🕹️ PTZ Control Thread     # Camera movement (RTSP only)
📊 Performance Monitor    # Resource usage tracking
```

**Buffer Management:**
- **RTSP Sources**: 1-frame buffer for real-time performance
- **Live Streams**: Minimal buffering with timing preservation
- **File Sources**: Adaptive buffering (small/medium/large options)
- **Webcams**: 1-frame buffer with high refresh rate

**Quality Optimization:**
- **Auto Mode**: Dynamically adjusts based on performance
- **High Quality**: 30 FPS, 60-80% JPEG quality, 640×640 detection
- **Balanced Mode**: 25 FPS, 60% JPEG quality, 320×320 detection  
- **Performance Mode**: 15 FPS, 40% JPEG quality, 320×320 detection

### Hardware Acceleration

**GPU Support (NVIDIA CUDA):**
```bash
# Check GPU availability
nvidia-smi

# Expected performance improvements
CPU (i7-10700K):     ~8 FPS  (nano model)
GPU (RTX 3060):     ~25 FPS  (nano model)  
GPU (RTX 4080):     ~45 FPS  (medium model)
```

**CPU Optimization:**
- **Multi-threading**: Utilizes all available CPU cores
- **Model Selection**: Auto-selects appropriate model size for CPU
- **Frame Skipping**: Intelligent frame dropping during high load
- **Memory Management**: Efficient garbage collection and cleanup

### Real-time Performance Monitoring

**Performance Metrics:**
- **Stream FPS**: Current video streaming frame rate
- **Detection FPS**: AI processing frame rate  
- **Latency**: End-to-end processing delay
- **Memory Usage**: RAM consumption monitoring
- **GPU Utilization**: CUDA memory and processing usage
- **Network Bandwidth**: Stream quality and data usage

**Adaptive Performance:**
- **Auto-scaling**: Reduces quality under high load
- **Dynamic FPS**: Adjusts target frame rate based on capability
- **Quality Fallback**: Temporarily reduces detection resolution
- **Resource Limiting**: Prevents system overload

## 🌐 Web Interface Features

### Tab-Based Source Organization

**🎥 RTSP/IP Camera Tab:**
- Enhanced URL templates for popular brands
- Real-time IP extraction from URLs
- ONVIF PTZ control section (visible only when connected)
- Auto-tracking toggle (PTZ cameras only)
- RTSP connection error notifications
- Camera credential management

**📹 Webcam Tab:**
- Auto-detection of available cameras
- Device information display (name, resolution, FPS)
- Refresh functionality without restart
- No PTZ controls (webcam-specific optimization)

**📺 Live Stream Tab:**
- Stream URL templates (YouTube, Twitch, HLS, etc.)
- Live timing preservation controls
- Buffer size optimization  
- Connection timeout settings
- Live stream indicators and status

**📁 File/URL Tab:**
- Auto-detection of file vs live stream URLs
- Original timing preservation controls  
- File looping options
- Buffer size management
- Format compatibility indicators

### Universal Controls (All Source Types)

**🤖 AI Detection Panel:**
- YOLO model toggle (ON/OFF)
- Detection overlay toggle (show/hide bounding boxes)
- Confidence threshold slider (10-90%)
- Input size selection (320×320, 416×416, 640×640)
- Model information display
- Custom model support indicator

**⚡ Performance Panel:**
- Target FPS control (10-30 FPS)
- Quality mode selection (Auto, High, Balanced, Performance)
- Real-time performance statistics
- Resource usage monitoring
- Latency measurement

**📊 System Status Panel:**
- Connection status indicator
- Source type display with timing mode
- OBS streaming URL generation
- Live performance metrics
- Error message display

### Advanced UI Features

**🖥️ Video Display:**
- **Fullscreen Mode**: Immersive viewing with escape key support
- **Fit Modes**: Cover (fill) vs Contain (fit) display options  
- **Live Indicators**: Real-time 🔴LIVE and ⏱️ORIG timing badges
- **Detection Overlay**: Toggleable bounding boxes and labels
- **Quality Controls**: Right-click context menu for settings

**📱 Mobile Responsiveness:**
- **Responsive Layout**: Adapts to smartphone and tablet screens
- **Touch Controls**: Optimized for touch interaction
- **Swipe Navigation**: Easy switching between source types
- **Portrait Mode**: Vertical layout optimization

**⚡ Real-time Updates:**
- **WebSocket Communication**: Live status updates without refresh
- **Performance Metrics**: Real-time FPS and quality monitoring  
- **Error Notifications**: Instant connection problem alerts
- **Status Indicators**: Live connection and detection status

## 📺 OBS Studio Integration

### Automatic Stream URL Generation

When you connect to any video source, the system automatically generates an OBS-compatible stream URL:

**Stream URL Format:**
```
http://localhost:4000/video_feed
http://YOUR_SERVER_IP:4000/video_feed
```

**Features:**
- **📋 One-Click Copy**: Copy URL directly to clipboard
- **📖 Built-in Instructions**: Step-by-step OBS setup guide
- **⚙️ Optimized Settings**: Pre-configured for OBS Media Source
- **🔄 Universal Compatibility**: Works with any connected video source
- **🌐 Network Access**: Access from other devices on network

### OBS Setup Instructions

**Step-by-Step Setup:**
1. **Open OBS Studio**
2. **Add Source**: Click + in Sources panel → Media Source
3. **Create New**: Give source a descriptive name
4. **Configure Source**:
   - ✅ **Uncheck "Local File"**
   - ✅ **Check "Network Buffering"** 
   - 📋 **Paste URL** into "Input" field
   - ⚡ **Set Buffer Size**: 2-5 seconds recommended
5. **Click OK** to add the source
6. **Resize/Position** as needed in OBS

**Recommended OBS Settings:**
- **Input Buffer**: 2-5 seconds for smooth playback
- **Restart When Activated**: Enabled for reliability
- **Network Buffering**: Always enabled for network sources
- **Hardware Acceleration**: Use if available

### Advanced OBS Integration

**Multiple Camera Setup:**
```
Camera 1: http://SERVER_IP:4000/video_feed (when Camera 1 connected)
Camera 2: http://SERVER_IP:4001/video_feed (second instance)
Camera 3: http://SERVER_IP:4002/video_feed (third instance)
```

**Stream Quality Optimization:**
- **High Quality**: 80-90% JPEG quality for crisp OBS recording
- **Balanced**: 60-70% JPEG quality for streaming
- **Performance**: 40-50% JPEG quality for multiple cameras

**Network Considerations:**
- **Local Network**: Use server IP address for best performance
- **Remote Access**: Configure router port forwarding if needed
- **Bandwidth**: Each stream uses ~2-10 Mbps depending on quality

## 🔍 Troubleshooting & FAQ

### Common Installation Issues

**❌ "Python not found" Error**
```bash
# Linux
sudo apt install python3 python3-pip python3-venv

# Windows
Download Python from https://python.org/downloads/
✅ Check "Add Python to PATH" during installation
```

**❌ "Microsoft Visual C++ Build Tools Required" (Windows)**
```cmd
Download and install:
https://visualstudio.microsoft.com/visual-cpp-build-tools/
Or install Visual Studio Community with C++ tools
```

**❌ OpenCV Installation Failed**
```bash
# Try headless version
pip install opencv-python-headless>=4.8.0

# Clear pip cache and retry
pip cache purge
pip install --no-cache-dir opencv-python

# Linux: Install system packages first
sudo apt install libopencv-dev python3-opencv
```

**❌ PyTorch CUDA Issues**
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU-only fallback
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Connection Issues

**❌ RTSP Connection Failed**
```bash
# Test RTSP URL with VLC first
vlc rtsp://192.168.1.100:554/stream1

# Common solutions:
✅ Check camera IP and port
✅ Verify username/password  
✅ Try different RTSP stream paths
✅ Test network connectivity: ping CAMERA_IP
✅ Check firewall settings
✅ Use brand-specific URL templates

# Popular RTSP URL formats:
Hikvision: rtsp://admin:pass@IP:554/Streaming/Channels/101
Dahua: rtsp://admin:pass@IP:554/cam/realmonitor?channel=1&subtype=0
```

**❌ Webcam Not Detected**
```bash
# Linux: Check permissions
sudo usermod -a -G video $USER
# Logout and login again

# Check available cameras
ls /dev/video*

# Windows: Check privacy settings
Settings → Privacy → Camera → Allow apps to access camera
```

**❌ ONVIF PTZ Discovery Failed**
```bash
# Check firewall (allow ONVIF ports)
# Linux
sudo ufw allow 80,8080,8899,8000,554,9999/tcp

# Windows  
netsh advfirewall firewall add rule name="ONVIF" dir=in action=allow protocol=TCP localport=80,8080,8899

# Test manual connection on different ports
# Try: 80, 8080, 8899, 8000, 554, 9999
```

**❌ YouTube/Twitch Stream Issues**
```bash
# Update yt-dlp
pip install --upgrade yt-dlp

# Clear cache
yt-dlp --rm-cache-dir

# Test URL manually
yt-dlp --get-url "https://www.youtube.com/watch?v=VIDEO_ID"

# For region-restricted content, use VPN
```

### Performance Issues

**❌ Low FPS / High Latency**
```bash
✅ Reduce YOLO input size: Settings → 320×320
✅ Lower confidence threshold: Settings → 30%  
✅ Disable detection overlay temporarily
✅ Use smaller YOLO model (nano instead of large)
✅ Enable GPU acceleration if available
✅ Reduce target FPS: Settings → 15 FPS
```

**❌ High CPU/Memory Usage**
```bash
# Monitor resources
htop                    # Linux
Task Manager           # Windows

# Optimizations
✅ Use GPU instead of CPU for AI detection
✅ Reduce buffer sizes
✅ Lower video quality/resolution  
✅ Close other applications
✅ Use nano YOLO models
✅ Restart application periodically
```

**❌ Video Quality Issues**
```bash
# Improve quality
✅ Increase quality settings: High mode
✅ Check source resolution and bitrate
✅ Ensure adequate network bandwidth
✅ Use larger YOLO input size: 640×640
✅ Verify camera/stream settings
```

### AI Detection Issues

**❌ YOLO Model Not Loading**
```bash
# Check models directory
ls models/
# Should contain .pt files

# Download manually
mkdir models
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -P models/

# Test loading
python -c "from ultralytics import YOLO; YOLO('models/yolov8n.pt')"
```

**❌ No Objects Detected**
```bash
✅ Lower confidence threshold: 20-30%
✅ Check lighting conditions  
✅ Try different model size: yolov8s.pt or yolov8m.pt
✅ Verify objects are clearly visible
✅ Check camera angle and distance
✅ Test with different video source
```

**❌ Too Many False Detections**
```bash
✅ Increase confidence threshold: 60-70%
✅ Use larger input size: 640×640
✅ Check for reflections or shadows
✅ Adjust camera positioning
✅ Use newer model: yolo11n.pt vs yolov5n.pt
```

### Network Issues

**❌ Can't Access Web Interface**
```bash
# Check if service is running
netstat -tlnp | grep 4000    # Linux  
netstat -an | findstr 4000   # Windows

# Test local access
curl http://localhost:4000

# Check firewall
# Linux
sudo ufw allow 4000

# Windows
netsh advfirewall firewall add rule name="CCTV" dir=in action=allow protocol=TCP localport=4000

# Access from another device
http://YOUR_SERVER_IP:4000
```

**❌ CORS Errors**
```bash
✅ Try different browser (Chrome, Firefox, Edge)
✅ Clear browser cache: Ctrl+Shift+Delete
✅ Use IP address instead of localhost
✅ Check browser console for errors (F12)
✅ Disable browser extensions temporarily
```

### Quick Health Check

**Test All Components:**
```bash
# Import test
python -c "
import cv2, flask, torch, ultralytics, yt_dlp
print('✅ All imports successful')
print(f'OpenCV: {cv2.__version__}')
print(f'PyTorch: {torch.__version__}')  
print(f'CUDA: {torch.cuda.is_available()}')
"

# Camera test
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print('✅ Camera test:', 'OK' if ret else 'FAILED')
cap.release()
"

# YOLO test
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('✅ YOLO model loaded successfully')
"
```

## 📁 Project Structure

```
multi-source-cctv/
├── 📄 app.py                    # Main Flask application server
├── 🌐 index.html               # Single-page web interface
├── 📋 requirements.txt         # Python dependencies
├── 📖 README.md                # This comprehensive documentation
├── 🪟 install.bat              # Windows automated installer
├── 🐧 install.sh               # Linux installation script
├── 🚀 start_cctv.bat          # Windows startup script
├── 🔄 update_system.bat       # Windows update script
├── 📁 models/                  # AI model directory
│   ├── yolov8n.pt             # Nano model (auto-downloaded)
│   ├── yolov8s.pt             # Small model (optional)
│   ├── yolo11n.pt             # Latest nano model (optional)
│   └── custom_model.pt        # Custom trained models
├── 📁 cctv_env/               # Python virtual environment
├── 📁 logs/                   # Application logs (auto-created)
├── 📁 config/                 # Configuration files (auto-created)
│   ├── cameras.json           # Saved camera configurations
│   ├── settings.json          # User preferences
│   └── templates.json         # RTSP URL templates
└── 📁 static/                 # Web assets (optional)
```

### Auto-Generated Files

**After Installation:**
- `cctv_env/` - Python virtual environment
- `models/` - YOLO models directory  
- `start_cctv.bat/sh` - Platform-specific startup scripts
- `update_system.bat/sh` - Dependency update scripts

**During Runtime:**
- `logs/` - Application and error logs
- `config/` - User settings and camera configurations

## 🚢 Deployment Options

### 🏠 Local Development
```bash
# Quick development setup
git clone https://github.com/your-repo/multi-source-cctv.git
cd multi-source-cctv
./install.sh  # or install.bat on Windows
python app.py
```

### 🐳 Docker Deployment
```bash
# Build and run with Docker
docker build -t multisource-cctv .
docker run -p 4000:4000 -v ./models:/app/models multisource-cctv

# With GPU support
docker run --gpus all -p 4000:4000 -v ./models:/app/models multisource-cctv
```

### ☁️ Cloud Deployment
```bash
# AWS EC2 (recommended: t3.medium or larger)
# Google Cloud Platform
# Azure Container Instances
# DigitalOcean Droplets

# See full deployment guide in documentation
```

### 🖥️ Production Server
```bash
# Ubuntu Server with systemd service
# Nginx reverse proxy
# SSL/TLS with Let's Encrypt
# Automated backups and monitoring

# See production deployment section for details
```

## 🛡️ Security & Privacy

### Security Features
- **🔒 HTTPS Support**: SSL/TLS encryption for web interface
- **🔐 Authentication**: Camera credential management
- **🛡️ Firewall Integration**: Automated firewall configuration  
- **🔒 Input Validation**: Sanitized user inputs and URL parsing
- **📊 Audit Logging**: Comprehensive activity logging

### Privacy Considerations
- **🏠 Local Processing**: All AI detection runs locally (no cloud)
- **📹 Video Privacy**: Video streams stay on your network
- **🔒 Credential Storage**: Encrypted camera password storage
- **🚫 No Telemetry**: No usage data sent to external services
- **📊 Optional Logging**: User-controlled logging levels

## 📊 API Documentation

### REST API Endpoints

**Connection Management:**
- `POST /connect_source` - Connect to video source
- `POST /disconnect` - Disconnect current source  
- `GET /scan_webcams` - Discover available webcams

**AI Detection Controls:**
- `POST /toggle_yolo` - Enable/disable YOLO detection
- `POST /toggle_detection_overlay` - Show/hide detection boxes
- `POST /update_settings` - Update AI parameters

**PTZ Control (RTSP/ONVIF only):**
- `POST /ptz_move` - Move PTZ camera
- `POST /toggle_tracking` - Enable/disable auto-tracking

**Streaming & Status:**
- `GET /video_feed` - Real-time video stream (MJPEG)
- `GET /status` - System status and performance metrics
- `GET /health` - Health check endpoint

### WebSocket Events

**Real-time Updates:**
- `status_update` - Live system status
- `performance_metrics` - FPS and resource usage
- `detection_results` - Object detection data
- `connection_status` - Source connection state

## 🔮 Roadmap & Future Features

### Planned Features (v2.0)
- **📱 Mobile App**: Native iOS/Android applications
- **☁️ Cloud Storage**: Optional cloud backup integration
- **🎯 Custom Training**: Easy custom model training interface
- **📊 Analytics Dashboard**: Advanced detection analytics
- **🔔 Alert System**: Email/SMS notifications for detections
- **🎮 Game Controller**: Physical PTZ control via gamepad

### Community Requests
- **🌐 Multi-Language Support**: Internationalization
- **🎨 Theme Customization**: Dark/light themes and color options
- **🔌 Plugin System**: Extensible architecture for third-party add-ons
- **📈 Advanced Reporting**: Detection statistics and reports
- **🏢 Multi-User Support**: User accounts and permissions

### Technology Upgrades
- **🚀 YOLO11+**: Latest YOLO model support as released
- **⚡ TensorRT**: NVIDIA TensorRT optimization
- **🧠 Edge TPU**: Google Coral Edge TPU support
- **📱 WebRTC**: Ultra-low latency streaming protocol
- **🔄 Auto-Updates**: Automatic dependency and model updates

## 📧 Support & Community

### 🆘 Getting Help

**Before Seeking Help:**
1. ✅ Check the [Troubleshooting](#-troubleshooting--faq) section
2. ✅ Search [existing issues](https://github.com/your-repo/multi-source-cctv/issues)
3. ✅ Try with a minimal configuration
4. ✅ Test with different video sources

### 📋 Bug Reports

**Create detailed bug reports with:**
- **System Info**: OS, Python version, GPU details
- **Installation Method**: Windows installer, Linux script, manual, Docker
- **Error Messages**: Complete error text and log files
- **Steps to Reproduce**: Exact sequence to trigger the issue
- **Expected vs Actual**: What should happen vs what actually happens

### ✨ Feature Requests

**Suggest new features with:**
- **Problem Description**: What problem does this solve?
- **Proposed Solution**: How should it work?
- **Use Cases**: Who would benefit?
- **Examples**: Similar features in other software

### 💬 Community Resources

- **GitHub Discussions**: Q&A, ideas, and general discussion
- **GitHub Issues**: Bug reports and feature requests
- **Community Wiki**: User-contributed guides and tips
- **Video Tutorials**: Setup and configuration guides

### 🏢 Professional Support

For organizations requiring professional assistance:
- **Installation Services**: Professional setup and configuration
- **Custom Development**: Feature development and customization
- **Training & Documentation**: Staff training and custom documentation
- **Performance Optimization**: System tuning and optimization
- **Security Auditing**: Security assessment and hardening

## 🎖️ Acknowledgments

### Core Technologies
- **[Ultralytics](https://ultralytics.com/)**: Revolutionary YOLO implementation and pre-trained models
- **[OpenCV](https://opencv.org/)**: Comprehensive computer vision library for video processing
- **[Flask](https://flask.palletsprojects.com/)**: Lightweight web framework for the interface
- **[PyTorch](https://pytorch.org/)**: Deep learning framework powering AI inference
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)**: Advanced video extraction for live streaming platforms

### Standards & Protocols
- **[ONVIF](https://www.onvif.org/)**: Open Network Video Interface Forum for PTZ control
- **[RTSP Protocol](https://tools.ietf.org/html/rfc2326)**: Real Time Streaming Protocol specification
- **[WebSocket](https://tools.ietf.org/html/rfc6455)**: Real-time bidirectional communication
- **[MJPEG](https://tools.ietf.org/html/rfc2435)**: Motion JPEG video streaming standard

### Community Contributors
- **🐛 Bug Hunters**: Users who identified and reported critical issues
- **✨ Feature Contributors**: Developers who added valuable functionality
- **📖 Documentation Writers**: Contributors who improved guides and tutorials
- **🧪 Beta Testers**: Users who tested across different platforms and configurations
- **🌍 International Community**: Translators and international user support

### Special Recognition
- **Security Researchers**: Responsible disclosure of security vulnerabilities
- **Performance Engineers**: Contributors who optimized system performance
- **Accessibility Advocates**: Making the interface more inclusive and usable
- **Educational Users**: Teachers and students using the system for learning

### Research & Development
- **YOLO Research Papers**: Object detection algorithm development
- **Computer Vision Community**: Academic research in real-time video processing
- **Open Source Movement**: Collaborative development and knowledge sharing
- **Edge Computing Research**: Bringing AI inference to resource-constrained devices

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

**✅ Permissions:**
- Commercial use
- Modification and distribution
- Private use

**📄 Conditions:**
- License and copyright notice must be included

**❌ Limitations:**
- No liability or warranty

### Third-Party Licenses

**Important:** This project uses several libraries with different licenses:
- **Ultralytics YOLO**: AGPL-3.0 (commercial license available)
- **OpenCV**: Apache 2.0
- **PyTorch**: BSD-3-Clause
- **Flask**: BSD-3-Clause

For commercial use, please review third-party licenses, especially YOLO's AGPL-3.0 terms.

---

**🚀 Built with ❤️ for real-time AI-powered video surveillance**


![506238882_4196252233936638_4386220351926044013_n](https://github.com/user-attachments/assets/8a9f3c07-7bc6-476c-841e-f7566ac34588)
![FireShot Capture 079 - Multi-Source CCTV System - YOLOv8 AI with Fixed File Timing - localhost](https://github.com/user-attachments/assets/abe77f94-e9a3-4517-926a-1e76f2b597f2)

OBS Support and etc
URL Example: http://localhost:4000/video_feed
![Cuplikan layar pada 2025-06-25 12-48-58](https://github.com/user-attachments/assets/fb5d9748-581c-4ed4-905c-66d90a438548)

---

**⭐ If this project helps you, please give it a star on GitHub!**

[![GitHub stars](https://img.shields.io/github/stars/your-repo/multi-source-cctv.svg?style=social&label=Star)](https://github.com/your-repo/multi-source-cctv)
[![GitHub forks](https://img.shields.io/github/forks/your-repo/multi-source-cctv.svg?style=social&label=Fork)](https://github.com/your-repo/multi-source-cctv/fork)
[![GitHub issues](https://img.shields.io/github/issues/your-repo/multi-source-cctv.svg)](https://github.com/your-repo/multi-source-cctv/issues)
[![GitHub license](https://img.shields.io/github/license/your-repo/multi-source-cctv.svg)](https://github.com/your-repo/multi-source-cctv/blob/main/LICENSE)
