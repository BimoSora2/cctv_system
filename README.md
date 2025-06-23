# 🚀 Multi-Source CCTV System with AI Detection

## 📖 Overview

A comprehensive, real-time video surveillance system that supports multiple video sources with AI-powered object detection. Built with Python Flask, OpenCV, and YOLO, featuring ultra-low latency streaming, PTZ camera control, and toggleable detection overlays.

## ✨ Key Features

### 📡 Multi-Source Video Support
- **RTSP/IP Cameras** with enhanced URL parsing
  - Supports all standard RTSP formats
  - Auto-credential extraction from URLs
  - Smart IP detection for ONVIF PTZ control
  - Brand-specific templates (Hikvision, Dahua, Axis)
- **USB/Built-in Webcams** with auto-detection
- **Live Streaming URLs** (YouTube Live, Twitch, HLS, Facebook Live, Instagram Live)
- **Video Files** (MP4, AVI, MOV, etc.) with proper timing preservation
- **HTTP/HTTPS Direct Streams**
- **DASH Adaptive Streams**

### 🤖 Universal AI Detection System
- **Official YOLO Models**: yolo11n.pt, yolov8n.pt, yolov5n.pt, etc.
- **Custom .pt Models**: Supports ANY PyTorch .pt file
- **Smart Model Detection**: Auto-finds and prioritizes best available models
- **Cross-Platform Discovery**: Works on Windows, Linux, macOS
- **GPU Acceleration**: CUDA support when available
- **Real-time Object Detection**: Person tracking and counting
- **Toggleable Detection Overlay**: Show/hide bounding boxes while keeping detection active

### ⏱️ Advanced Timing Preservation
- **Live Stream Mode**: Preserves original FPS and timing for live streams
- **File Mode**: Maintains natural playback speed for video files
- **Auto-Detection**: Automatically identifies live streams vs files
- **Fixed Timing Issues**: Resolves fast playback problems
- **Source-Specific Optimization**: Different strategies for each source type

### 🔒 Enhanced RTSP Support
- **Smart URL Parsing**: Handles all standard RTSP URL formats
  - `rtsp://192.168.1.4:554/live/ch00_0` (No auth)
  - `rtsp://admin:@192.168.1.4:554/live/ch00_0` (User only)
  - `rtsp://admin:admin@192.168.1.4:554/live/ch00_0` (Full auth)
- **Auto-Credential Extraction**: Parses username/password from URLs
- **Fallback Authentication**: Uses separate credential fields if URL doesn't contain them
- **IP Auto-Detection**: Extracts camera IP for ONVIF PTZ control
- **Error Notifications**: Real-time RTSP connection error alerts

### 🕹️ PTZ Camera Control
- **Smart ONVIF Discovery**: Multi-port, multi-authentication discovery
- **Pan/Tilt/Zoom Control**: Full PTZ functionality
- **Auto Person Tracking**: AI-powered person following
- **Adjustable Sensitivity**: Customizable tracking parameters
- **Speed Control**: Variable PTZ movement speeds

### ⚡ Ultra-Low Latency Streaming
- **Threaded Architecture**: Separated capture and streaming threads
- **Minimal Buffering**: Optimized for real-time performance
- **Source-Specific Optimization**: Different strategies per source type
- **Performance Monitoring**: Real-time FPS and latency tracking

### 🌐 Modern Web Interface
- **Responsive Design**: Works on desktop and mobile
- **Real-time Controls**: Live adjustment of all parameters
- **Source Templates**: Pre-configured settings for popular services
- **Fullscreen Support**: Immersive video viewing
- **Performance Dashboard**: Live system monitoring

## 🛠️ Installation

### Prerequisites
```bash
pip install opencv-python flask flask-socketio onvif-zeep numpy ultralytics yt-dlp
```

### Optional Dependencies
- **yt-dlp**: For YouTube/Twitch stream extraction
- **PyTorch**: For GPU acceleration (CUDA support)

### Quick Start
1. Clone the repository
2. Install dependencies
3. Place YOLO models in `./models/` folder (optional - will auto-download)
4. Run the application:
```bash
python app.py
```
5. Open http://localhost:4000 in your browser

## 📦 AI Model Support

### Official YOLO Models (Auto-Download)
- **YOLO11**: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
- **YOLOv8**: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
- **YOLOv5**: yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt

### Custom Models
- **Any .pt File**: my_model.pt, best.pt, custom_weights.pt, etc.
- **Smart Priority System**: Official models > Custom models > Size-based selection
- **Size Filtering**: 1MB-500MB (excludes corrupted files)
- **Pattern Matching**: Prioritizes YOLO-like named files

### Model Discovery Locations
```
./models/
./weights/
./
~/.cache/ultralytics/
/usr/local/share/ultralytics/ (Linux)
C:/ProgramData/ultralytics/ (Windows)
```

## 🔧 Configuration

### RTSP Camera Setup
```python
# Basic RTSP
rtsp://192.168.1.100:554/stream1

# With Authentication in URL
rtsp://admin:password@192.168.1.100:554/stream1

# Brand-Specific Examples
# Hikvision
rtsp://admin:admin@192.168.1.100:554/Streaming/Channels/101

# Dahua
rtsp://admin:admin@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0

# Axis
rtsp://admin:admin@192.168.1.100:554/axis-media/media.amp
```

### Live Stream URLs
```python
# YouTube Live
https://www.youtube.com/watch?v=VIDEO_ID

# Twitch
https://www.twitch.tv/CHANNEL_NAME

# HLS Live Stream
https://example.com/live/stream.m3u8

# Facebook Live
https://www.facebook.com/USERNAME/live
```

### File Sources
```python
# Local Files
/path/to/video.mp4

# Network Files
http://example.com/video.mp4

# With Loop
loop_video: true
```

## 🎛️ API Endpoints

### Connection Management
- `POST /connect_source` - Connect to video source
- `POST /disconnect` - Disconnect current source
- `GET /scan_webcams` - Discover available webcams

### AI Controls
- `POST /toggle_yolo` - Enable/disable YOLO detection
- `POST /toggle_detection_overlay` - Show/hide detection boxes
- `POST /update_settings` - Update AI parameters

### PTZ Control
- `POST /ptz_move` - Move PTZ camera
- `POST /toggle_tracking` - Enable/disable auto-tracking

### Streaming
- `GET /video_feed` - Real-time video stream
- `GET /status` - System status and performance

## 🔍 Detection Features

### Object Detection
- **80+ Object Classes**: Standard COCO dataset objects
- **Person Detection**: Specialized person tracking and counting
- **Confidence Threshold**: Adjustable detection sensitivity
- **Input Size**: Configurable for speed vs accuracy balance

### Visual Feedback
- **Golden Bounding Boxes**: High-contrast detection visualization
- **Object Labels**: Class names with confidence scores
- **Detection Overlay Toggle**: Hide boxes while keeping detection active
- **Real-time Counts**: Live person counting display

## 📊 Performance Optimization

### Speed Settings
- **Target FPS**: 10-30 FPS range
- **Quality Modes**: High, Balanced, Performance, Auto
- **Input Sizes**: 320×320 (Fast), 416×416 (Balanced), 640×640 (Accurate)
- **Buffer Management**: Minimal latency configuration

### Hardware Acceleration
- **GPU Support**: CUDA acceleration when available
- **CPU Fallback**: Optimized CPU processing
- **Threading**: Multi-threaded capture and processing

## 🔒 Security Features

### Authentication
- **RTSP Credentials**: Username/password support
- **URL Embedded Auth**: Parse credentials from RTSP URLs
- **ONVIF Discovery**: Multiple authentication methods
- **Fallback Options**: Graceful degradation for auth failures

### Network Security
- **CORS Support**: Cross-origin resource sharing
- **Timeout Controls**: Connection and read timeouts
- **Error Handling**: Graceful failure management

## 🐛 Troubleshooting

### Common Issues

#### YOLO Model Not Found
```bash
# Create models directory
mkdir models

# Download official model
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -P models/

# Or place any .pt file in models/ folder
```

#### RTSP Connection Failed
- Check URL format and credentials
- Verify camera IP and port
- Test with VLC or other RTSP client
- Check firewall settings

#### Performance Issues
- Reduce input size (320×320)
- Lower target FPS
- Disable YOLO if not needed
- Use CPU instead of GPU for small models

#### Webcam Not Detected
- Check USB connections
- Verify camera drivers
- Try different camera indices
- Grant camera permissions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLO implementation
- **OpenCV**: Computer vision library
- **Flask**: Web framework
- **ONVIF**: PTZ camera control standard

## 📧 Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed description
4. Include system information and error logs

---

**⚡ Built for real-time performance with professional-grade features**

![506238882_4196252233936638_4386220351926044013_n](https://github.com/user-attachments/assets/8a9f3c07-7bc6-476c-841e-f7566ac34588)
![FireShot Capture 079 - Multi-Source CCTV System - YOLOv8 AI with Fixed File Timing - localhost](https://github.com/user-attachments/assets/abe77f94-e9a3-4517-926a-1e76f2b597f2)
