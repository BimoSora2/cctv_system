#!/usr/bin/env python3
"""
Enhanced Multi-Source CCTV System Backend with Ultra Low-Latency Streaming
Universal .pt Model Detection System with Smart Priority
Supports: RTSP/ONVIF, Webcams, Live Streams, Files - All with YOLOv8 AI Detection
Enhanced with Live Stream Original Timing Preservation & Fixed File Timing
Added Toggleable Detection Overlay Feature
Requires: opencv-python, flask, flask-socketio, onvif-zeep, numpy, ultralytics, yt-dlp
"""

import cv2
import numpy as np
import threading
import time
import json
import gc
import queue
import signal
import os
import sys
import platform
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from onvif import ONVIFCamera
import base64
import logging

# Enhanced KeyboardInterrupt handling
def signal_handler(signum, frame):
    print("\n")
    print("🛑 KeyboardInterrupt detected during startup!")
    print("🔄 Gracefully shutting down Multi-Source CCTV system...")
    print("🚀 If you want to restart, run: python app.py")
    print("✅ Startup interrupted cleanly")
    print("👋 Thank you for using Multi-Source CCTV System!")
    sys.exit(0)

# Set up signal handler BEFORE any heavy imports
signal.signal(signal.SIGINT, signal_handler)

# Enhanced YOLOv8 imports with better timeout protection and fallback
print("🔄 Loading YOLOv8 dependencies...")
YOLO_AVAILABLE = False
try:
    # First, try a quick import test
    import importlib.util
    
    # Check if ultralytics is installed
    spec = importlib.util.find_spec("ultralytics")
    if spec is None:
        print("⚠️  ultralytics not installed. Install with: pip install ultralytics")
        YOLO_AVAILABLE = False
    else:
        # Try importing with a more robust approach
        print("📦 Found ultralytics package, attempting import...")
        
        def safe_ultralytics_import():
            """Safely import ultralytics with error handling"""
            try:
                from ultralytics import YOLO
                # JANGAN test model creation di sini - ini yang menyebabkan download!
                # test_model = YOLO()  # ❌ INI YANG MENYEBABKAN AUTO-DOWNLOAD!
                return True, YOLO
            except Exception as e:
                print(f"⚠️  Ultralytics import failed: {e}")
                return False, None
        
        # Use threading for timeout control
        import threading
        result_container = {'success': False, 'YOLO': None}
        
        def import_thread():
            result_container['success'], result_container['YOLO'] = safe_ultralytics_import()
        
        thread = threading.Thread(target=import_thread, daemon=True)
        thread.start()
        thread.join(timeout=15)  # 15 second timeout
        
        if thread.is_alive():
            print("⚠️  YOLOv8 import timeout - continuing without YOLO")
            YOLO_AVAILABLE = False
        elif result_container['success']:
            YOLO = result_container['YOLO']
            YOLO_AVAILABLE = True
            print("✅ YOLOv8 (ultralytics) loaded successfully")
        else:
            print("⚠️  YOLOv8 import failed - continuing without YOLO")
            YOLO_AVAILABLE = False
            
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Warning: ultralytics not installed. Install with: pip install ultralytics")
except Exception as e:
    YOLO_AVAILABLE = False
    print(f"⚠️  YOLOv8 import error: {e}")

# Optional yt-dlp for enhanced streaming support
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
    print("✅ yt-dlp loaded for enhanced streaming support")
except ImportError:
    YT_DLP_AVAILABLE = False
    print("💡 Optional: Install yt-dlp for YouTube/Twitch support: pip install yt-dlp")

# Setup logging with debug level untuk troubleshooting
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'multi_source_cctv_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class MultiSourceCCTV:
    def __init__(self):
        print("🚀 Initializing Multi-Source CCTV System...")
        
        # Source Configuration
        self.source_type = "none"  # none, rtsp, webcam, stream, file
        self.source_info = {}
        self.source_display_name = "None"
        
        # RTSP/ONVIF Configuration
        self.rtsp_url = ""
        self.camera_ip = ""
        self.camera_port = 80
        self.username = ""
        self.password = ""
        
        # Webcam Configuration
        self.webcam_index = -1
        self.available_webcams = []
        
        # Stream Configuration
        self.stream_url = ""
        self.stream_type = "auto"
        
        # File Configuration
        self.file_path = ""
        self.loop_video = False
        
        # System State
        self.camera = None
        self.onvif_cam = None
        self.ptz_service = None
        self.media_service = None
        self.media_profile = None
        self.cap = None
        self.is_running = False
        self.motion_tracking = False
        self.last_person_position = None
        self.ptz_moving = False
        self.person_count = 0
        self.auth_method_used = ""
        
        # ULTRA LOW LATENCY SETTINGS
        self.target_fps = 25
        self.max_fps = 30
        self.stream_fps = 25
        self.detection_fps = 5
        
        # STREAMING OPTIMIZATIONS
        self.frame_buffer_size = 1
        self.jpeg_quality = 60
        self.stream_quality = 60
        self.detection_quality = 40
        
        # LIVE STREAM TIMING SETTINGS - NEW
        self.preserve_live_timing = True  # Flag untuk mempertahankan timing asli live stream
        self.live_stream_fps = None  # FPS asli dari live stream
        self.is_live_stream = False  # Flag untuk mendeteksi apakah sumber adalah live stream
        self.stream_start_time = None  # Waktu mulai stream untuk sinkronisasi
        self.frame_timestamps = []  # Buffer timestamp frame untuk live stream
        self.max_timestamp_buffer = 10  # Maximum timestamp buffer size
        self.timing_sensitivity = 1.0  # Sensitivity untuk timing preservation
        
        # ADAPTIVE TIMING untuk berbagai jenis stream
        self.adaptive_timing = {
            'live': True,    # Gunakan timing asli untuk live stream
            'file': True,    # Gunakan timing asli untuk file (FIXED)
            'rtsp': True,    # Gunakan timing asli untuk RTSP
            'webcam': False  # Gunakan timing yang dioptimasi untuk webcam
        }
        
        # THREADING SEPARATION
        self.streaming_thread = None
        self.detection_thread = None
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()
        self.stream_frame_queue = queue.Queue(maxsize=2)
        
        # FRAME TIMING
        self.last_stream_time = 0
        self.last_detection_time = 0
        self.stream_interval = 1.0 / self.stream_fps
        self.detection_interval = 1.0 / self.detection_fps
        
        # DETECTION SEPARATION
        self.detection_enabled = False
        self.detection_frame = None
        self.detection_results = []
        self.detection_lock = threading.Lock()
        
        # DETECTION OVERLAY SETTINGS
        self.show_detection_overlay = True  # Flag untuk show/hide bounding boxes
        
        # Threading optimizations
        self.thread_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="CCTV")
        
        # Enhanced YOLOv8 Configuration
        self.yolo_model = None
        self.yolo_model_path = None
        self.yolo_model_name = None
        self.yolo_enabled = False
        self.yolo_confidence_threshold = 0.4
        self.yolo_iou_threshold = 0.3
        self.yolo_input_size = 320
        self.yolo_device = 'cpu'
        
        # YOLOv8 Dynamic Settings Lock
        self.yolo_settings_lock = threading.Lock()
        self.yolo_model_needs_reload = False
        
        # Enhanced YOLOv8 Setup with better error handling
        print("🤖 Setting up YOLOv8...")
        self.yolo_ready = self.setup_yolov8_enhanced()
        
        # Tracking parameters
        self.tracking_sensitivity = 80
        self.pan_speed = 0.2
        self.tilt_speed = 0.2
        self.ptz_speed_multiplier = 1.0
        self.tracking_cooldown = 1.0
        self.last_tracking_time = 0
        
        # Background Subtractor
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            varThreshold=16,
            history=100
        )
        
        # Performance monitoring
        self.performance_stats = {
            'stream_fps': 0,
            'detection_fps': 0,
            'frames_streamed': 0,
            'frames_detected': 0,
            'last_stats_time': time.time()
        }
        
        # Connection parameters for low latency
        self.rtsp_transport = "tcp"
        self.connection_timeout = 3000
        self.read_timeout = 1000
        
        print("✅ Multi-Source CCTV System initialized successfully!")
    
    def find_yolo_models_enhanced(self):
        """Enhanced YOLO model detection with support for any .pt file names"""
        possible_locations = []
        
        # Current working directory
        cwd = Path.cwd()
        logger.info(f"🔍 Current working directory: {cwd}")
        possible_locations.extend([
            cwd,
            cwd / "models",
            cwd / "weights",
            cwd / "yolo",
            cwd / "ultralytics",
        ])
        
        # Script directory
        script_dir = Path(__file__).parent
        logger.info(f"🔍 Script directory: {script_dir}")
        possible_locations.extend([
            script_dir,
            script_dir / "models",
            script_dir / "weights",
            script_dir / "yolo",
            script_dir / "ultralytics",
        ])
        
        # Home directory ultralytics cache
        home_dir = Path.home()
        possible_locations.extend([
            home_dir / ".cache" / "ultralytics",
            home_dir / ".ultralytics",
        ])
        
        # System-wide locations
        if platform.system() == "Linux":
            possible_locations.extend([
                Path("/usr/local/share/ultralytics"),
                Path("/opt/ultralytics"),
            ])
        elif platform.system() == "Windows":
            possible_locations.extend([
                Path("C:/ProgramData/ultralytics"),
                Path(os.environ.get("LOCALAPPDATA", "")) / "ultralytics",
            ])
        
        # Model names to search for (prioritized - official YOLO models)
        official_model_names = [
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"
        ]
        
        found_models = []
        
        # Debug: Print semua lokasi yang akan dicari
        logger.info(f"🔍 Searching in {len(possible_locations)} locations:")
        for location in possible_locations:
            logger.info(f"   📂 {location} {'✅' if location.exists() else '❌'}")
        
        for location in possible_locations:
            if not location.exists():
                continue
                
            # Debug: List isi folder jika ada
            try:
                files_in_dir = list(location.glob("*.pt"))
                if files_in_dir:
                    logger.info(f"📁 Found .pt files in {location}:")
                    for file in files_in_dir:
                        logger.info(f"   • {file.name}")
            except Exception as e:
                logger.debug(f"Cannot list files in {location}: {e}")
                
            # PHASE 1: Search for official YOLO model names first
            for model_name in official_model_names:
                model_path = location / model_name
                if model_path.exists() and model_path.is_file():
                    try:
                        size_mb = model_path.stat().st_size / (1024 * 1024)
                        found_models.append({
                            'path': str(model_path),
                            'name': model_name,
                            'size_mb': size_mb,
                            'location': str(location),
                            'is_official': True,
                            'priority': 1  # Highest priority for official models
                        })
                        logger.info(f"✅ Found official model: {model_name} at {model_path}")
                    except Exception as e:
                        logger.debug(f"Error checking model {model_path}: {e}")
                        continue
        
        # PHASE 2: If no official models found, search for ANY .pt files
        if not found_models:
            logger.info("🔍 No official YOLO models found. Searching for any .pt files...")
            
            for location in possible_locations:
                if not location.exists():
                    continue
                    
                try:
                    # Find all .pt files in this location
                    pt_files = list(location.glob("*.pt"))
                    
                    for pt_file in pt_files:
                        if pt_file.is_file():
                            try:
                                size_mb = pt_file.stat().st_size / (1024 * 1024)
                                file_name = pt_file.name
                                
                                # Skip very small files (likely not YOLO models)
                                if size_mb < 1.0:
                                    logger.debug(f"⏭️ Skipping small file: {file_name} ({size_mb:.2f}MB)")
                                    continue
                                
                                # Skip very large files (likely not standard YOLO models)
                                if size_mb > 500.0:
                                    logger.debug(f"⏭️ Skipping large file: {file_name} ({size_mb:.2f}MB)")
                                    continue
                                
                                # Assign priority based on naming patterns and size
                                priority = self._calculate_pt_file_priority(file_name, size_mb)
                                
                                found_models.append({
                                    'path': str(pt_file),
                                    'name': file_name,
                                    'size_mb': size_mb,
                                    'location': str(location),
                                    'is_official': False,
                                    'priority': priority
                                })
                                logger.info(f"📦 Found .pt file: {file_name} ({size_mb:.1f}MB) - Priority: {priority}")
                                
                            except Exception as e:
                                logger.debug(f"Error checking .pt file {pt_file}: {e}")
                                continue
                                
                except Exception as e:
                    logger.debug(f"Error scanning .pt files in {location}: {e}")
                    continue
        
        # Sort found models by priority and size
        if found_models:
            found_models.sort(key=lambda x: (x['priority'], -x['size_mb']))
            logger.info(f"📊 Found {len(found_models)} potential YOLO model(s)")
            
            # Log top candidates
            for i, model in enumerate(found_models[:5]):  # Show top 5 candidates
                status = "Official" if model['is_official'] else "Custom"
                logger.info(f"   {i+1}. {model['name']} ({model['size_mb']:.1f}MB) - {status}")
        
        return found_models
    
    def _calculate_pt_file_priority(self, file_name, size_mb):
        """Calculate priority for custom .pt files based on name patterns and size"""
        file_lower = file_name.lower()
        
        # Priority 2: Files with YOLO-like naming patterns
        yolo_patterns = ['yolo', 'yolov', 'ultralytics', 'detection', 'object']
        if any(pattern in file_lower for pattern in yolo_patterns):
            return 2
        
        # Priority 3: Files with model-like naming patterns
        model_patterns = ['model', 'net', 'weights', 'trained', 'best', 'final']
        if any(pattern in file_lower for pattern in model_patterns):
            return 3
        
        # Priority 4: Files with reasonable YOLO model sizes (5-150MB)
        if 5.0 <= size_mb <= 150.0:
            return 4
        
        # Priority 5: Small models (1-5MB) - might be nano models
        if 1.0 <= size_mb < 5.0:
            return 5
        
        # Priority 6: Larger models (150-500MB) - might be custom trained
        if 150.0 < size_mb <= 500.0:
            return 6
        
        # Priority 7: Any other .pt file
        return 7
    
    def safe_cuda_check_enhanced(self):
        """Enhanced CUDA availability check with better timeout handling"""
        try:
            # Quick torch availability check
            import importlib.util
            torch_spec = importlib.util.find_spec("torch")
            if torch_spec is None:
                print("💻 PyTorch not available, using CPU")
                return 'cpu'
            
            def cuda_check_thread():
                try:
                    import torch
                    return torch.cuda.is_available(), torch
                except Exception as e:
                    print(f"⚠️  PyTorch import error: {e}")
                    return False, None
            
            result_container = {'available': False, 'torch': None}
            
            def check_thread():
                result_container['available'], result_container['torch'] = cuda_check_thread()
            
            thread = threading.Thread(target=check_thread, daemon=True)
            thread.start()
            thread.join(timeout=5)  # 5 second timeout
            
            if thread.is_alive():
                print("⚠️  CUDA check timeout, defaulting to CPU")
                return 'cpu'
            
            if result_container['available'] and result_container['torch']:
                torch = result_container['torch']
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    print(f"🚀 CUDA available - GPU Memory: {gpu_memory:.1f}GB")
                    return 'cuda'
                except Exception as e:
                    print(f"⚠️  CUDA detection error: {e}, using CPU")
                    return 'cpu'
            else:
                print("💻 CUDA not available, using CPU")
                return 'cpu'
                
        except Exception as e:
            print(f"⚠️  CUDA check error: {e}, using CPU")
            return 'cpu'
    
    def setup_yolov8_enhanced(self):
        """Enhanced YOLOv8 initialization with support for any .pt file"""
        if not YOLO_AVAILABLE:
            logger.warning("YOLOv8 not available. Using motion detection only.")
            return False
            
        try:
            logger.info("🤖 Setting up YOLOv8 with enhanced .pt file detection...")
            
            # Find available models dengan sistem prioritas
            logger.info("🔍 Searching for YOLO models (official and custom .pt files)...")
            found_models = self.find_yolo_models_enhanced()
            
            # Debug: Print semua model yang ditemukan
            logger.info(f"📂 Found {len(found_models)} potential model(s):")
            for i, model in enumerate(found_models[:10]):  # Show top 10
                status = "OFFICIAL" if model.get('is_official', False) else "CUSTOM"
                priority = model.get('priority', 'N/A')
                logger.info(f"   {i+1}. {model['name']} ({model['size_mb']:.1f}MB) - {status} [Priority: {priority}]")
            
            model_path = None
            selected_model = None
            
            if found_models:
                # Models are already sorted by priority, so take the first one
                selected_model = found_models[0]
                model_path = selected_model['path']
                self.yolo_model_name = selected_model['name']
                
                # Enhanced logging based on model type
                if selected_model.get('is_official', False):
                    logger.info(f"🎯 Selected OFFICIAL model: {self.yolo_model_name} ({selected_model['size_mb']:.1f}MB)")
                else:
                    logger.info(f"📦 Selected CUSTOM .pt file: {self.yolo_model_name} ({selected_model['size_mb']:.1f}MB)")
                    logger.info(f"   ⚠️ Note: Custom model may have different classes or performance")
                
                logger.info(f"📍 Model path: {model_path}")
            
            if not model_path:
                # Enhanced guidance for users
                logger.warning("📁 No YOLO models or .pt files found")
                logger.warning("💡 To enable AI detection, you can:")
                logger.warning("   1. Download official YOLO models:")
                logger.warning("      • mkdir models")
                logger.warning("      • wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt -P models/")
                logger.warning("      • wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -P models/")
                logger.warning("   2. Or place any .pt model file in:")
                logger.warning("      • ./models/ folder")
                logger.warning("      • Current directory")
                logger.warning("      • ./weights/ folder")
                logger.warning("   3. Supported: Any PyTorch .pt file (YOLO, custom models, etc.)")
                logger.warning("   4. Restart the application")
                logger.warning("🔄 Continuing with motion detection only...")
                return False
            
            # Enhanced model loading with better error handling
            logger.info(f"🚀 Loading model: {model_path}")
            
            def load_model():
                try:
                    model = YOLO(model_path)
                    return model, None
                except Exception as e:
                    return None, str(e)
            
            result_container = {'model': None, 'error': None}
            
            def load_thread():
                result_container['model'], result_container['error'] = load_model()
            
            load_thread_obj = threading.Thread(target=load_thread, daemon=True)
            load_thread_obj.start()
            load_thread_obj.join(timeout=30)  # 30 second timeout for model loading
            
            if load_thread_obj.is_alive():
                logger.error("⚠️ Model loading timeout - YOLOv8 initialization failed")
                return False
            
            if result_container['model'] is None:
                logger.error(f"⚠️ Model loading failed: {result_container['error']}")
                
                # Additional guidance for custom models
                if selected_model and not selected_model.get('is_official', False):
                    logger.error("💡 Custom .pt file failed to load. This might happen if:")
                    logger.error("   • File is corrupted or incomplete")
                    logger.error("   • Model was trained with incompatible PyTorch version")
                    logger.error("   • Model is not a YOLO-compatible format")
                    logger.error("   • Try downloading an official YOLO model instead")
                
                return False
            else:
                self.yolo_model = result_container['model']
            
            self.yolo_model_path = str(model_path)
            
            # Enhanced device selection
            self.yolo_device = self.safe_cuda_check_enhanced()
            
            # Safe device assignment
            try:
                self.yolo_model.to(self.yolo_device)
                logger.info(f"✅ Model assigned to device: {self.yolo_device}")
            except Exception as e:
                logger.warning(f"⚠️ Device assignment failed: {e}, using CPU")
                self.yolo_device = 'cpu'
                self.yolo_model.to('cpu')
            
            # Enhanced model warm-up with timeout
            logger.info("🔥 Warming up model...")
            try:
                def warmup():
                    dummy_image = np.zeros((self.yolo_input_size, self.yolo_input_size, 3), dtype=np.uint8)
                    return self.yolo_model.predict(
                        dummy_image, 
                        verbose=False, 
                        device=self.yolo_device, 
                        imgsz=self.yolo_input_size
                    )
                
                warmup_result = {'success': False}
                
                def warmup_thread():
                    try:
                        warmup()
                        warmup_result['success'] = True
                    except Exception as e:
                        logger.warning(f"Warmup failed: {e}")
                
                warmup_thread_obj = threading.Thread(target=warmup_thread, daemon=True)
                warmup_thread_obj.start()
                warmup_thread_obj.join(timeout=15)  # 15 second timeout
                
                if not warmup_result['success']:
                    logger.warning("⚠️ Model warmup failed or timed out, but continuing...")
                else:
                    logger.info("🔥 Model warmed up successfully")
                    
            except Exception as e:
                logger.warning(f"⚠️ Model warmup error: {e}, but continuing...")
            
            # Enhanced success logging
            model_type = "Official YOLO" if selected_model.get('is_official', False) else "Custom .pt"
            logger.info(f"✅ YOLOv8 initialized successfully!")
            logger.info(f"   📁 Model: {self.yolo_model_name} ({model_type})")
            logger.info(f"   🖥️ Device: {self.yolo_device}")
            logger.info(f"   📏 Input size: {self.yolo_input_size}x{self.yolo_input_size}")
            logger.info(f"   📊 Size: {selected_model['size_mb']:.1f}MB")
            
            if not selected_model.get('is_official', False):
                logger.info(f"   ℹ️ Note: Custom model detected - object classes may vary")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ YOLOv8 initialization failed: {e}")
            return False
    
    def detect_stream_type_and_timing(self, url, stream_type):
        """Deteksi jenis stream dan tentukan strategi timing yang tepat"""
        is_live = False
        preserve_timing = False
        
        # Deteksi live stream berdasarkan URL dan tipe
        live_indicators = [
            'live', 'stream', '.m3u8', 'youtube.com/watch', 'twitch.tv',
            'facebook.com/watch', 'instagram.com/live', 'tiktok.com/live'
        ]
        
        if any(indicator in url.lower() for indicator in live_indicators):
            is_live = True
            preserve_timing = True
            logger.info(f"🔴 Detected LIVE stream: {url}")
            logger.info("⏱️  Will preserve original live stream timing")
        
        # Stream type specific detection
        if stream_type in ['youtube', 'twitch', 'hls', 'dash']:
            is_live = True
            preserve_timing = True
        elif stream_type in ['http'] and any(live_word in url.lower() for live_word in ['live', 'stream']):
            is_live = True
            preserve_timing = True
        
        return is_live, preserve_timing
    
    def detect_url_type_and_timing(self, url):
        """Enhanced detection untuk membedakan file URL vs live stream URL"""
        is_live = False
        preserve_timing = True  # Default preserve timing untuk semua
        
        # Deteksi berdasarkan URL pattern
        live_indicators = [
            'live', 'stream', '.m3u8', '.mpd',
            'youtube.com/watch', 'twitch.tv', 'facebook.com/watch', 
            'instagram.com/live', 'tiktok.com/live'
        ]
        
        # Deteksi file extension vs live stream
        file_extensions = [
            '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm',
            '.m4v', '.3gp', '.ogv', '.ts', '.mts', '.m2ts'
        ]
        
        url_lower = url.lower()
        
        # Cek apakah ini file extension
        is_file = any(url_lower.endswith(ext) for ext in file_extensions)
        
        # Cek apakah ini live stream indicator
        is_live_indicator = any(indicator in url_lower for indicator in live_indicators)
        
        if is_live_indicator and not is_file:
            is_live = True
            preserve_timing = True
            logger.info(f"🔴 Detected LIVE STREAM URL: {url}")
        elif is_file:
            is_live = False
            preserve_timing = True  # File juga perlu preserve timing asli
            logger.info(f"📁 Detected FILE URL: {url}")
        else:
            # Default: treat as file with original timing
            is_live = False
            preserve_timing = True
            logger.info(f"📄 Default FILE mode with original timing: {url}")
        
        return is_live, preserve_timing
    
    def scan_webcams(self):
        """Enhanced webcam detection with device information"""
        webcams = []
        max_webcams = 10  # Check up to 10 webcam indices
        
        logger.info("🔍 Scanning for available webcams...")
        
        for index in range(max_webcams):
            try:
                # Test webcam with minimal timeout
                cap = cv2.VideoCapture(index)
                
                if cap.isOpened():
                    # Try to read a frame to verify it's actually working
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Get webcam properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        # Generate device name
                        device_name = self.get_webcam_name(index)
                        
                        webcam_info = {
                            'index': index,
                            'name': device_name,
                            'resolution': f"{width}x{height}",
                            'fps': fps,
                            'info': f"{width}x{height} @ {fps:.1f}fps"
                        }
                        
                        webcams.append(webcam_info)
                        logger.info(f"   ✅ Found: {device_name} (Index: {index}) - {width}x{height}")
                
                cap.release()
                
            except Exception as e:
                logger.debug(f"   ❌ Index {index}: {e}")
                continue
        
        self.available_webcams = webcams
        logger.info(f"📹 Webcam scan complete: {len(webcams)} webcam(s) found")
        
        return webcams
    
    def get_webcam_name(self, index):
        """Get human-readable webcam name based on OS"""
        try:
            system = platform.system()
            
            if system == "Windows":
                # Try to get device name from Windows registry or WMI
                return f"Webcam {index}"
            
            elif system == "Linux":
                # Try to read from /sys/class/video4linux/
                device_path = f"/sys/class/video4linux/video{index}/name"
                if os.path.exists(device_path):
                    with open(device_path, 'r') as f:
                        return f.read().strip()
                else:
                    return f"Video Device {index}"
            
            elif system == "Darwin":  # macOS
                # macOS webcam detection
                return f"Camera {index}"
            
            else:
                return f"Webcam {index}"
                
        except Exception:
            return f"Webcam {index}"
    
    def get_stream_url(self, url, stream_type):
        """Enhanced stream URL processing with yt-dlp support"""
        try:
            if stream_type == "auto":
                # Auto-detect stream type
                if "youtube.com" in url or "youtu.be" in url:
                    stream_type = "youtube"
                elif "twitch.tv" in url:
                    stream_type = "twitch"
                elif url.endswith(('.m3u8', '.m3u')):
                    stream_type = "hls"
                elif url.endswith('.mpd'):
                    stream_type = "dash"
                else:
                    stream_type = "http"
            
            if stream_type in ["youtube", "twitch"] and YT_DLP_AVAILABLE:
                # Use yt-dlp to extract direct stream URL
                logger.info(f"🔍 Extracting stream URL from {stream_type.title()}...")
                
                ydl_opts = {
                    'format': 'best[ext=mp4]/best',
                    'quiet': True,
                    'no_warnings': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    if 'url' in info:
                        direct_url = info['url']
                        logger.info(f"✅ Extracted direct URL for {stream_type.title()}")
                        return direct_url, stream_type
                    else:
                        logger.warning(f"⚠️ Could not extract URL from {url}")
                        return url, stream_type
            
            return url, stream_type
            
        except Exception as e:
            logger.warning(f"⚠️ Stream URL processing error: {e}")
            return url, stream_type
    
    def connect_source(self, source_config):
        """Enhanced connection method for multiple source types"""
        try:
            if self.cap:
                self.cap.release()
            
            source_type = source_config.get('source_type', 'rtsp')
            self.source_type = source_type
            self.source_info = source_config.copy()
            
            logger.info(f"🔗 Connecting to {source_type.upper()} source...")
            
            success = False
            message = ""
            
            if source_type == 'rtsp':
                success, message = self.connect_rtsp(source_config)
                self.source_display_name = "RTSP/IP Camera"
                
            elif source_type == 'webcam':
                success, message = self.connect_webcam(source_config)
                webcam_info = source_config.get('webcam_info', {})
                self.source_display_name = webcam_info.get('name', f"Webcam {source_config.get('webcam_index', 0)}")
                
            elif source_type == 'stream':
                success, message = self.connect_stream(source_config)
                self.source_display_name = f"Live Stream ({source_config.get('stream_type', 'auto').upper()})"
                
            elif source_type == 'file':
                success, message = self.connect_file(source_config)
                file_path = source_config.get('file_url', '')
                self.source_display_name = f"File: {os.path.basename(file_path)}"
                
            else:
                return False, f"Unknown source type: {source_type}"
            
            if success:
                # Reset performance counters
                self.performance_stats = {
                    'stream_fps': 0,
                    'detection_fps': 0,
                    'frames_streamed': 0,
                    'frames_detected': 0,
                    'last_stats_time': time.time()
                }
                
                # Reset background subtractor
                self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                    detectShadows=False, varThreshold=16, history=100
                )
                
                self.last_person_position = None
                self.ptz_moving = False
                self.person_count = 0
                
                logger.info(f"✅ Successfully connected to {source_type.upper()} source")
                return True, message
            else:
                logger.error(f"❌ Failed to connect to {source_type.upper()} source: {message}")
                return False, message
                
        except Exception as e:
            logger.error(f"❌ Connection error: {str(e)}")
            return False, f"Connection error: {str(e)}"
    
    def connect_rtsp(self, config):
        """Connect to RTSP stream with ONVIF discovery"""
        self.rtsp_url = config['rtsp_url']
        self.camera_ip = config.get('camera_ip', '')
        self.camera_port = config.get('port', 80)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        
        logger.info(f"📹 Connecting to RTSP: {self.rtsp_url}")
        
        # ULTRA LOW LATENCY CONNECTION SETTINGS
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Critical optimizations
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.frame_buffer_size)
        self.cap.set(cv2.CAP_PROP_FPS, self.max_fps)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.connection_timeout)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout)
        
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        except:
            pass
        
        if not self.cap.isOpened():
            return False, "Failed to connect to RTSP stream"
        
        # Test frame read
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, "Cannot read frames from RTSP stream"
        
        with self.latest_frame_lock:
            self.latest_frame = frame.copy()
        
        # ONVIF Discovery (if camera IP provided)
        onvif_result = {"status": "skipped"}
        if self.camera_ip.strip():
            logger.info("🚀 Starting Smart ONVIF Discovery...")
            onvif_result = self.smart_onvif_discovery(self.camera_ip, self.username, self.password)
        
        # Build message
        message = f"RTSP connected successfully"
        
        if onvif_result["status"] == "connected":
            device_info = onvif_result.get("device_info")
            device_name = f"{device_info.Manufacturer} {device_info.Model}" if device_info else "Unknown Device"
            message += f" with ONVIF control - {device_name}"
        elif onvif_result["status"] == "skipped":
            message += " (video only)"
        else:
            message += " (video only - ONVIF discovery failed)"
        
        return True, message
    
    def connect_webcam(self, config):
        """Connect to webcam"""
        webcam_index = config.get('webcam_index', 0)
        
        logger.info(f"🎥 Connecting to webcam index: {webcam_index}")
        
        self.cap = cv2.VideoCapture(webcam_index)
        
        if not self.cap.isOpened():
            return False, f"Failed to open webcam at index {webcam_index}"
        
        # Optimize webcam settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, self.max_fps)
        
        # Test frame read
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, f"Cannot read frames from webcam {webcam_index}"
        
        with self.latest_frame_lock:
            self.latest_frame = frame.copy()
        
        webcam_info = config.get('webcam_info', {})
        webcam_name = webcam_info.get('name', f"Webcam {webcam_index}")
        resolution = webcam_info.get('resolution', 'Unknown')
        
        return True, f"Webcam connected: {webcam_name} ({resolution})"
    
    def connect_stream(self, config):
        """Enhanced stream connection with live timing preservation"""
        stream_url = config.get('stream_url', '')
        stream_type = config.get('stream_type', 'auto')
        
        logger.info(f"📺 Connecting to live stream: {stream_url}")
        
        # Deteksi apakah ini live stream
        self.is_live_stream, self.preserve_live_timing = self.detect_stream_type_and_timing(stream_url, stream_type)
        
        # Process stream URL
        processed_url, detected_type = self.get_stream_url(stream_url, stream_type)
        self.stream_url = processed_url  # Store for reconnection
        
        logger.info(f"🔗 Processed URL for {detected_type.upper()} stream")
        if self.is_live_stream:
            logger.info("🔴 LIVE STREAM MODE: Preserving original timing")
        
        self.cap = cv2.VideoCapture(processed_url)
        
        # LIVE STREAM SPECIFIC OPTIMIZATIONS
        if self.is_live_stream and self.preserve_live_timing:
            # Minimal buffering untuk live stream
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer minimal
            
            # Jangan paksa FPS pada live stream - biarkan menggunakan FPS asli
            # self.cap.set(cv2.CAP_PROP_FPS, self.max_fps)  # DISABLED untuk live stream
            
            # Timeout yang lebih panjang untuk koneksi live stream
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)  # 15 detik
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)   # 8 detik
            
            # Aktifkan threaded reading untuk live stream
            try:
                self.cap.set(cv2.CAP_PROP_THREADED_READ, 1)
            except:
                pass
                
        else:
            # Optimisasi normal untuk non-live stream
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        
        if not self.cap.isOpened():
            return False, f"Failed to connect to {detected_type.upper()} stream"
        
        # Test frame read dan deteksi FPS asli
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, f"Cannot read frames from {detected_type.upper()} stream"
        
        # Deteksi FPS asli dari stream
        if self.is_live_stream:
            original_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if original_fps > 0:
                self.live_stream_fps = original_fps
                logger.info(f"📊 Detected original stream FPS: {original_fps:.2f}")
            else:
                # Fallback FPS untuk live stream yang tidak melaporkan FPS
                self.live_stream_fps = 25.0
                logger.info(f"📊 Using fallback FPS for live stream: {self.live_stream_fps}")
        
        with self.latest_frame_lock:
            self.latest_frame = frame.copy()
        
        # Initialize timing untuk live stream
        if self.is_live_stream:
            self.stream_start_time = time.time()
            self.frame_timestamps = []
        
        live_mode_text = " (LIVE MODE)" if self.is_live_stream else ""
        return True, f"{detected_type.upper()} stream connected successfully{live_mode_text}"
    
    def connect_file(self, config):
        """Enhanced file connection dengan proper timing detection"""
        file_url = config.get('file_url', '')
        loop_video = config.get('loop_video', False)
        preserve_file_timing = config.get('preserve_file_timing', True)
        detected_type = config.get('detected_type', 'file')
        file_buffer_size = config.get('file_buffer_size', 'medium')
        
        logger.info(f"📁 Connecting to file/URL: {file_url}")
        
        # Deteksi apakah ini file atau live stream URL
        if detected_type == 'live_stream':
            self.is_live_stream = True
            self.preserve_live_timing = preserve_file_timing
        else:
            self.is_live_stream, self.preserve_live_timing = self.detect_url_type_and_timing(file_url)
            # Override dengan user preference
            self.preserve_live_timing = preserve_file_timing
        
        self.file_path = file_url
        self.loop_video = loop_video
        
        self.cap = cv2.VideoCapture(file_url)
        
        # Pengaturan berbeda untuk file vs live stream URL
        if self.is_live_stream:
            # Jika terdeteksi sebagai live stream URL
            logger.info("🔴 File URL detected as LIVE STREAM - using live settings")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer untuk live
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
        else:
            # Jika ini file biasa
            logger.info("📁 File URL detected as REGULAR FILE - using file settings")
            buffer_sizes = {'small': 1, 'medium': 3, 'large': 5}
            buffer_size = buffer_sizes.get(file_buffer_size, 3)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
            
        if not self.cap.isOpened():
            return False, f"Failed to open file/URL: {file_url}"
        
        # Deteksi FPS asli dari file/stream
        original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if original_fps > 0:
            self.live_stream_fps = original_fps  # Gunakan untuk timing reference
            logger.info(f"📊 Detected original FPS: {original_fps:.2f}")
        else:
            self.live_stream_fps = 25.0  # Fallback FPS
            logger.info(f"📊 Using fallback FPS: {self.live_stream_fps}")
        
        # Test frame read
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, f"Cannot read frames from file/URL: {file_url}"
        
        with self.latest_frame_lock:
            self.latest_frame = frame.copy()
        
        # Initialize timing
        self.stream_start_time = time.time()
        self.frame_timestamps = []
        
        file_name = os.path.basename(file_url)
        loop_info = " (looping)" if loop_video else ""
        timing_info = " (LIVE TIMING)" if self.is_live_stream else " (ORIGINAL TIMING)"
        
        return True, f"File connected: {file_name}{loop_info}{timing_info}"
    
    def smart_onvif_discovery(self, camera_ip, username, password):
        """Smart ONVIF discovery - tries multiple protocols and ports"""
        logger.info(f"🔍 Starting Smart ONVIF Discovery for {camera_ip}...")
        
        discovery_configs = [
            {"port": 80, "desc": "Standard HTTP", "timeout": 3},
            {"port": 8080, "desc": "Alternative HTTP", "timeout": 3},
            {"port": 8899, "desc": "NVR/Recorder ONVIF", "timeout": 3},
            {"port": 8000, "desc": "Hikvision Style", "timeout": 3},
            {"port": 554, "desc": "RTSP Port", "timeout": 3},
            {"port": 9999, "desc": "Dahua Style", "timeout": 3},
        ]
        
        auth_methods = [
            {"user": "", "pass": "", "desc": "No Auth"},
            {"user": "admin", "pass": "", "desc": "Admin No Password"},
            {"user": "admin", "pass": "admin", "desc": "Admin/Admin"},
            {"user": username.strip() if username else "", "pass": password.strip() if password else "", "desc": "User Provided"}
        ]
        
        # Remove duplicates
        if username.strip():
            auth_methods = [auth for auth in auth_methods if not (auth["user"] == username.strip() and auth["pass"] == password.strip())]
            auth_methods.append({"user": username.strip(), "pass": password.strip(), "desc": "User Provided"})
        
        total_attempts = len(discovery_configs) * len(auth_methods)
        current_attempt = 0
        
        for config in discovery_configs:
            port = config["port"]
            timeout = config["timeout"]
            
            for auth in auth_methods:
                current_attempt += 1
                auth_user = auth["user"]
                auth_pass = auth["pass"]
                auth_desc = auth["desc"]
                
                try:
                    test_cam = ONVIFCamera(
                        camera_ip, 
                        port, 
                        auth_user, 
                        auth_pass,
                        wsdl_dir=False,
                        no_cache=True
                    )
                    
                    import signal
                    def timeout_handler(signum, frame):
                        raise TimeoutError("ONVIF connection timeout")
                    
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout)
                    
                    try:
                        device_info = test_cam.devicemgmt.GetDeviceInformation()
                        signal.alarm(0)
                        
                        self.onvif_cam = test_cam
                        self.auth_method_used = f"{auth_desc} on port {port}"
                        
                        logger.info(f"✅ ONVIF Connected: {device_info.Manufacturer} {device_info.Model}")
                        
                        ptz_available = self.setup_onvif_services()
                        
                        return {
                            "status": "connected",
                            "port": port,
                            "auth_desc": auth_desc,
                            "device_info": device_info,
                            "ptz_available": ptz_available,
                            "attempt": current_attempt,
                            "total_attempts": total_attempts
                        }
                        
                    except TimeoutError:
                        signal.alarm(0)
                        continue
                        
                except Exception:
                    continue
        
        logger.warning(f"❌ ONVIF Discovery failed after {total_attempts} attempts")
        return {"status": "failed", "attempts": total_attempts}
    
    def setup_onvif_services(self):
        """Setup ONVIF services and test PTZ availability"""
        ptz_available = False
        
        try:
            self.media_service = self.onvif_cam.create_media_service()
            if self.media_service:
                profiles = self.media_service.GetProfiles()
                if profiles:
                    self.media_profile = profiles[0]
            
            self.ptz_service = self.onvif_cam.create_ptz_service()
            if self.ptz_service and self.media_profile:
                try:
                    ptz_config = self.ptz_service.GetConfiguration(self.media_profile.PTZConfiguration.token)
                    if ptz_config:
                        ptz_available = True
                        logger.info("🕹️ PTZ Available")
                except:
                    logger.info("🕹️ PTZ Not Available")
        except Exception as e:
            logger.debug(f"Service setup warning: {e}")
        
        return ptz_available
    
    def frame_capture_thread(self):
        """Enhanced frame capture thread dengan proper file timing"""
        logger.info("Ultra low-latency frame capture started")
        if self.is_live_stream:
            logger.info("🔴 LIVE STREAM MODE: Preserving original timing")
        elif self.preserve_live_timing:
            logger.info("📁 FILE MODE: Preserving original timing")
        
        consecutive_failures = 0
        max_failures = 10
        
        # Timing variables untuk semua jenis source
        last_frame_time = time.time()
        
        # Tentukan interval frame berdasarkan jenis source
        if self.preserve_live_timing and self.live_stream_fps:
            expected_frame_interval = 1.0 / self.live_stream_fps
            logger.info(f"⏱️  Using original timing: {self.live_stream_fps:.2f} FPS")
        else:
            expected_frame_interval = 1.0 / self.stream_fps
            logger.info(f"⚡ Using optimized timing: {self.stream_fps:.2f} FPS")
        
        while self.is_running:
            try:
                if self.cap is not None and self.cap.isOpened():
                    frame_start_time = time.time()
                    
                    ret, frame = self.cap.read()
                    
                    if ret and frame is not None:
                        consecutive_failures = 0
                        
                        # Timing preservation logic
                        if self.preserve_live_timing and expected_frame_interval:
                            current_time = time.time()
                            elapsed_since_last = current_time - last_frame_time
                            
                            # Kontrol timing berdasarkan FPS asli
                            if elapsed_since_last < expected_frame_interval:
                                sleep_time = expected_frame_interval - elapsed_since_last
                                if sleep_time > 0:
                                    time.sleep(sleep_time * self.timing_sensitivity)
                            
                            # Update timestamp buffer
                            self.frame_timestamps.append(current_time)
                            if len(self.frame_timestamps) > self.max_timestamp_buffer:
                                self.frame_timestamps.pop(0)
                            
                            last_frame_time = time.time()
                        
                        with self.latest_frame_lock:
                            self.latest_frame = frame.copy()
                        
                        # Queue management
                        try:
                            self.stream_frame_queue.put_nowait(frame.copy())
                        except queue.Full:
                            try:
                                self.stream_frame_queue.get_nowait()  # Buang frame lama
                                self.stream_frame_queue.put_nowait(frame.copy())
                            except queue.Empty:
                                pass
                    else:
                        # Handle end of file
                        if self.source_type == 'file' and self.loop_video:
                            logger.info("🔄 Looping video file...")
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            # Reset timing untuk loop
                            last_frame_time = time.time()
                            self.stream_start_time = time.time()
                            self.frame_timestamps = []
                            time.sleep(0.1)
                            continue
                        
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            logger.warning("Frame capture: too many failures")
                            if self.source_type in ['rtsp', 'stream', 'file']:
                                self.reconnect_source()
                            consecutive_failures = 0
                        
                        time.sleep(0.05)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                if self.is_running:
                    logger.error(f"Frame capture error: {e}")
                time.sleep(0.1)
        
        logger.info("Frame capture thread stopped")
    
    def reconnect_source(self):
        """Optimized source reconnection"""
        try:
            logger.info("Attempting source reconnection...")
            if self.cap:
                self.cap.release()
            
            time.sleep(0.5)
            
            # Reconnect based on source type
            if self.source_type == 'rtsp':
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.frame_buffer_size)
                self.cap.set(cv2.CAP_PROP_FPS, self.max_fps)
            elif self.source_type == 'webcam':
                self.cap = cv2.VideoCapture(self.webcam_index)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif self.source_type in ['stream', 'file']:
                source_url = self.stream_url if self.source_type == 'stream' else self.file_path
                self.cap = cv2.VideoCapture(source_url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.latest_frame_lock:
                    self.latest_frame = frame.copy()
                logger.info("Source reconnection successful")
                return True
            else:
                logger.warning("Source reconnection failed")
                return False
                
        except Exception as e:
            logger.error(f"Source reconnection error: {e}")
            return False
    
    def detection_thread_worker(self):
        """Detection thread for AI and motion detection with proper frame saving"""
        logger.info("Detection thread started")
        
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - self.last_detection_time < self.detection_interval:
                    time.sleep(0.05)
                    continue
                
                if not self.detection_enabled:
                    time.sleep(0.1)
                    continue
                
                with self.latest_frame_lock:
                    if self.latest_frame is not None:
                        detection_frame = self.latest_frame.copy()
                    else:
                        time.sleep(0.1)
                        continue
                
                # Process detection - PENTING: detection akan modify frame dengan bounding boxes
                if self.yolo_ready and self.yolo_enabled:
                    processed_frame, persons = self.detect_persons_yolov8_optimized(detection_frame)
                    logger.debug(f"🤖 YOLOv8 processed frame with {len(persons)} persons")
                else:
                    processed_frame, persons = self.detect_motion_optimized(detection_frame)
                    logger.debug(f"👁️ Motion detection processed frame with {len(persons)} detections")
                
                # SIMPAN frame yang sudah ada bounding boxes untuk streaming
                with self.detection_lock:
                    self.detection_frame = processed_frame.copy()  # Frame dengan bounding boxes
                    self.detection_results = persons
                    self.person_count = len(persons)
                    logger.debug(f"💾 Saved detection frame with overlays")
                
                self.performance_stats['frames_detected'] += 1
                self.last_detection_time = current_time
                
                if persons and self.motion_tracking:
                    self.thread_pool.submit(self.track_person_async, persons)
                
            except Exception as e:
                if self.is_running:
                    logger.error(f"Detection error: {e}")
                time.sleep(0.1)
        
        logger.info("Detection thread stopped")
    
    def detect_persons_yolov8_optimized(self, frame):
        """Enhanced YOLOv8 detection supporting any .pt model with comprehensive object classes"""
        try:
            height, width = frame.shape[:2]
            
            with self.yolo_settings_lock:
                current_input_size = self.yolo_input_size
                current_confidence = self.yolo_confidence_threshold
            
            # Debug logging
            logger.debug(f"🔍 YOLOv8 Detection - Frame: {width}x{height}, Confidence: {current_confidence}")
            
            # Resize frame for processing
            if width > current_input_size:
                scale = current_input_size / width
                new_width = current_input_size
                new_height = int(height * scale)
                resized_frame = cv2.resize(frame, (new_width, new_height))
            else:
                resized_frame = frame
                scale = 1.0
            
            # Enhanced YOLOv8 inference with error handling
            try:
                results = self.yolo_model.predict(
                    resized_frame,
                    conf=current_confidence,
                    iou=self.yolo_iou_threshold,
                    # Deteksi semua kelas objek, bukan hanya person (class 0)
                    device=self.yolo_device,
                    verbose=False,
                    imgsz=current_input_size
                )
                logger.debug(f"🤖 YOLOv8 inference completed, results: {len(results) if results else 0}")
            except Exception as inference_error:
                logger.error(f"YOLOv8 inference error: {inference_error}")
                return frame, []
            
            # Enhanced class names - Support for standard YOLO classes and custom models
            # This is a comprehensive list that covers most YOLO model classes
            standard_coco_classes = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
                15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
                50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
                55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
                65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
                70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
                75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
            }
            
            persons = []
            detected_objects = 0
            
            # Warna emas untuk bounding box dan text
            golden_color = (0, 215, 255)  # BGR format untuk warna emas
            text_background_color = (0, 165, 255)  # Warna emas lebih gelap untuk background text
            
            if results and len(results) > 0:
                result = results[0]
                logger.debug(f"📦 Processing result with boxes: {hasattr(result, 'boxes')}")
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    logger.debug(f"🎯 Found {len(boxes)} detection boxes")
                    
                    for i, box in enumerate(boxes):
                        try:
                            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                            conf = box.conf[0].cpu().numpy() if hasattr(box.conf[0], 'cpu') else box.conf[0]
                            cls = int(box.cls[0].cpu().numpy() if hasattr(box.cls[0], 'cpu') else box.cls[0])
                            
                            x1, y1, x2, y2 = map(int, xyxy)
                            
                            # Scale back to original frame size
                            if scale != 1.0:
                                x1 = int(x1 / scale)
                                y1 = int(y1 / scale)
                                x2 = int(x2 / scale)
                                y2 = int(y2 / scale)
                            
                            # Get class name - try standard COCO first, fallback to generic
                            class_name = standard_coco_classes.get(cls, f'object_{cls}')
                            
                            logger.debug(f"✅ Processing {class_name} at ({x1},{y1},{x2},{y2}) conf={conf:.2f}")
                            
                            # HANYA GAMBAR BOUNDING BOXES JIKA show_detection_overlay = True
                            if self.show_detection_overlay:
                                # Draw golden bounding box dengan thickness yang lebih tebal
                                cv2.rectangle(frame, (x1, y1), (x2, y2), golden_color, 3)
                                
                                # Prepare label text
                                label = f"{class_name} {conf:.2f}"
                                
                                # Get text size untuk background
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.7
                                font_thickness = 2
                                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                                
                                # Draw background rectangle untuk text (emas gelap)
                                text_bg_x1 = x1
                                text_bg_y1 = y1 - text_height - baseline - 10
                                text_bg_x2 = x1 + text_width + 10
                                text_bg_y2 = y1
                                
                                # Pastikan background text tidak keluar dari frame
                                if text_bg_y1 < 0:
                                    text_bg_y1 = y2
                                    text_bg_y2 = y2 + text_height + baseline + 10
                                
                                # Draw background rectangle
                                cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), text_background_color, -1)
                                
                                # Draw border untuk background text
                                cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), golden_color, 2)
                                
                                # Draw text label (putih untuk kontras dengan background emas)
                                text_x = x1 + 5
                                text_y = text_bg_y1 + text_height + 5 if text_bg_y1 >= 0 else text_bg_y2 - 5
                                cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
                            
                            # Jika ini person, tambahkan ke tracking (tracking tetap bekerja meski overlay hidden)
                            if cls == 0:  # person class (standard COCO)
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                persons.append((center_x, center_y))
                            
                            detected_objects += 1
                            
                        except Exception as box_error:
                            logger.error(f"Box processing error: {box_error}")
                            continue
                else:
                    logger.debug("❌ No boxes found in result")
            else:
                logger.debug("❌ No results from YOLOv8")
            
            # Update person count untuk tracking (hanya person)
            self.person_count = len(persons)
            logger.debug(f"📊 Detection summary: {detected_objects} objects, {len(persons)} persons, Overlay: {'ON' if self.show_detection_overlay else 'OFF'}")
            
            return frame, persons
            
        except Exception as e:
            logger.error(f"YOLOv8 detection error: {e}")
            return frame, []
    
    def detect_motion_optimized(self, frame):
        """Motion detection fallback with golden bounding boxes"""
        try:
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                small_frame = cv2.resize(frame, (new_width, new_height))
            else:
                small_frame = frame
                scale = 1.0
            
            fg_mask = self.background_subtractor.apply(small_frame)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            persons = []
            min_area = 1000 / (scale * scale)
            
            # Warna emas untuk motion detection juga
            golden_color = (0, 215, 255)  # BGR format untuk warna emas
            text_background_color = (0, 165, 255)  # Warna emas lebih gelap untuk background text
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    if scale != 1.0:
                        x = int(x / scale)
                        y = int(y / scale)
                        w = int(w / scale)
                        h = int(h / scale)
                    
                    aspect_ratio = h / w if w > 0 else 0
                    if 1.2 <= aspect_ratio <= 4.0:
                        # HANYA GAMBAR BOUNDING BOXES JIKA show_detection_overlay = True
                        if self.show_detection_overlay:
                            # Draw golden bounding box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), golden_color, 3)
                            
                            # Prepare label text
                            label = f"Motion Detected"
                            
                            # Get text size untuk background
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.7
                            font_thickness = 2
                            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                            
                            # Draw background rectangle untuk text (emas gelap)
                            text_bg_x1 = x
                            text_bg_y1 = y - text_height - baseline - 10
                            text_bg_x2 = x + text_width + 10
                            text_bg_y2 = y
                            
                            # Pastikan background text tidak keluar dari frame
                            if text_bg_y1 < 0:
                                text_bg_y1 = y + h
                                text_bg_y2 = y + h + text_height + baseline + 10
                            
                            # Draw background rectangle
                            cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), text_background_color, -1)
                            
                            # Draw border untuk background text
                            cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), golden_color, 2)
                            
                            # Draw text label (putih untuk kontras dengan background emas)
                            text_x = x + 5
                            text_y = text_bg_y1 + text_height + 5 if text_bg_y1 >= 0 else text_bg_y2 - 5
                            cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
                        
                        # Tetap tambahkan untuk tracking meski overlay hidden
                        center_x = x + w // 2
                        center_y = y + h // 2
                        persons.append((center_x, center_y))
            
            logger.debug(f"👁️ Motion detection: {len(persons)} detections, Overlay: {'ON' if self.show_detection_overlay else 'OFF'}")
            return frame, persons
            
        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            return frame, []
    
    def add_overlay_info(self, frame):
        """Enhanced overlay dengan info timing mode yang tepat"""
        height, width = frame.shape[:2]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        line_spacing = 20
        margin = 8
        
        bg_color = (0, 0, 0)
        bg_opacity = 0.75
        text_padding = 4
        white_color = (255, 255, 255)
        live_color = (0, 255, 0)  # Hijau untuk live
        file_color = (135, 206, 235)  # Biru untuk file
        
        # Get detection info
        detection_person_count = 0
        detection_method = "None"
        
        with self.detection_lock:
            detection_person_count = self.person_count
            if self.detection_enabled:
                detection_method = 'YOLOv8' if (self.yolo_ready and self.yolo_enabled) else 'Motion'
        
        with self.yolo_settings_lock:
            current_input_size = self.yolo_input_size
        
        # Enhanced source info dengan timing indicator
        source_text = self.source_display_name[:15]
        source_color = white_color
        
        if self.is_live_stream:
            source_text += " 🔴LIVE"
            source_color = live_color
        elif self.preserve_live_timing:
            source_text += " ⏱️ORIG"
            source_color = file_color
        
        # Enhanced FPS info
        fps_text = f"{self.performance_stats['stream_fps']:.1f}fps"
        if self.live_stream_fps and self.preserve_live_timing:
            fps_text += f" (src: {self.live_stream_fps:.1f})"
        
        texts = [
            (f"Source: {source_text}", source_color),
            (f"Stream: {fps_text}", white_color),
            (f"Quality: {self.stream_quality}", white_color),
            ("Track: ON" if self.motion_tracking else "Track: OFF", white_color),
            (f"People: {detection_person_count}", white_color),
            (f"Det: {detection_method}", white_color),
            (f"YOLO: {current_input_size}x{current_input_size}" if self.yolo_enabled else "", white_color),
            (f"Overlay: {'ON' if self.show_detection_overlay else 'OFF'}", white_color)
        ]
        
        # Enhanced model info for custom .pt files
        if self.yolo_enabled and self.yolo_model_name:
            model_display = self.yolo_model_name.replace('.pt', '').upper()[:10]
            texts.append((f"Model: {model_display}", white_color))
        
        # Tambah info timing mode
        if self.preserve_live_timing:
            if self.is_live_stream:
                texts.append(("Live Timing: ON", live_color))
            else:
                texts.append(("Original Timing: ON", file_color))
        
        # Filter out empty texts
        texts = [(text, color) for text, color in texts if text.strip()]
        
        # Calculate background
        max_text_width = 0
        for text, _ in texts:
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            max_text_width = max(max_text_width, text_size[0])
        
        overlay_width = max_text_width + (text_padding * 2)
        overlay_height = len(texts) * line_spacing + (text_padding * 2)
        
        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (margin - text_padding, margin - text_padding), 
                     (margin + overlay_width, margin + overlay_height), 
                     bg_color, -1)
        
        cv2.addWeighted(overlay, bg_opacity, frame, 1 - bg_opacity, 0, frame)
        
        # Add text
        y_position = margin + line_spacing - 5
        
        for text, color in texts:
            cv2.putText(frame, text, (margin, y_position), font, font_scale, color, thickness)
            y_position += line_spacing
        
        # Enhanced resolution info
        resolution_text = f"{width}x{height}"
        if self.is_live_stream:
            resolution_text += " LIVE"
        elif self.preserve_live_timing:
            resolution_text += " ORIG"
        
        res_font_scale = 0.4
        res_thickness = 2
        res_text_size = cv2.getTextSize(resolution_text, font, res_font_scale, res_thickness)[0]
        res_bg_x1 = width - res_text_size[0] - margin - text_padding
        res_bg_y1 = height - 20 - text_padding
        res_bg_x2 = width - margin + text_padding
        res_bg_y2 = height - margin + text_padding
        
        res_overlay = frame.copy()
        cv2.rectangle(res_overlay, (res_bg_x1, res_bg_y1), (res_bg_x2, res_bg_y2), bg_color, -1)
        cv2.addWeighted(res_overlay, bg_opacity, frame, 1 - bg_opacity, 0, frame)
        
        if self.is_live_stream:
            res_color = live_color
        elif self.preserve_live_timing:
            res_color = file_color
        else:
            res_color = white_color
            
        cv2.putText(frame, resolution_text, 
                   (width - res_text_size[0] - margin, height - margin), 
                   font, res_font_scale, res_color, res_thickness)
    
    def stream_generator_ultra_low_latency(self):
        """Enhanced stream generator dengan live timing preservation and detection overlay"""
        logger.info("Ultra low-latency stream generator started")
        if self.is_live_stream:
            logger.info("🔴 LIVE STREAM MODE: Maintaining original timing")
        elif self.preserve_live_timing:
            logger.info("📁 FILE MODE: Maintaining original timing")
        
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.stream_quality,
                        cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        
        # Timing variables
        if self.preserve_live_timing and self.live_stream_fps:
            target_interval = 1.0 / self.live_stream_fps
        else:
            target_interval = self.stream_interval
        
        last_yield_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Timing control
                if self.preserve_live_timing:
                    elapsed_since_yield = current_time - last_yield_time
                    if elapsed_since_yield < target_interval:
                        time.sleep(max(0.001, target_interval - elapsed_since_yield))
                        continue
                
                # Get frame - prioritas ke detection frame jika detection aktif
                frame = None
                
                if self.detection_enabled:
                    # Gunakan frame yang sudah ada detection overlay
                    with self.detection_lock:
                        if self.detection_frame is not None:
                            frame = self.detection_frame.copy()
                            logger.debug("📺 Using detection frame with overlays")
                
                # Fallback ke latest frame jika detection frame tidak ada
                if frame is None:
                    try:
                        frame = self.stream_frame_queue.get_nowait()
                        logger.debug("📺 Using queue frame")
                    except queue.Empty:
                        with self.latest_frame_lock:
                            if self.latest_frame is not None:
                                frame = self.latest_frame.copy()
                                logger.debug("📺 Using latest frame")
                            else:
                                time.sleep(0.01)
                                continue
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Add overlay info (tanpa detection boxes karena sudah ada di detection_frame)
                self.add_overlay_info(frame)
                
                # Fast JPEG encoding
                _, buffer = cv2.imencode('.jpg', frame, encode_params)
                frame_bytes = buffer.tobytes()
                
                self.performance_stats['frames_streamed'] += 1
                last_yield_time = time.time()
                
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Adaptive delay
                if not self.preserve_live_timing:
                    elapsed = time.time() - current_time
                    if elapsed < self.stream_interval:
                        time.sleep(max(0.001, self.stream_interval - elapsed))
                
            except Exception as e:
                if self.is_running:
                    logger.error(f"Stream generation error: {e}")
                time.sleep(0.01)
        
        logger.info("Stream generator stopped")
    
    def start_streaming(self):
        """Start streaming with multi-source support"""
        self.is_running = True
        
        # Start capture thread
        self.streaming_thread = threading.Thread(target=self.frame_capture_thread, daemon=True)
        self.streaming_thread.start()
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_thread_worker, daemon=True)
        self.detection_thread.start()
        
        logger.info("Multi-source streaming started")
    
    def stop_streaming(self):
        """Stop streaming and cleanup"""
        logger.info("Stopping multi-source streaming...")
        self.is_running = False
        
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=2)
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)
        
        if self.cap:
            try:
                self.cap.release()
                logger.info("Source released successfully")
            except Exception as e:
                logger.warning(f"Error during source release: {e}")
            finally:
                self.cap = None
        
        # Clear queues and caches
        while not self.stream_frame_queue.empty():
            try:
                self.stream_frame_queue.get_nowait()
            except queue.Empty:
                break
        
        with self.latest_frame_lock:
            self.latest_frame = None
        
        with self.detection_lock:
            self.detection_frame = None
            self.detection_results = []
        
        # Reset source info
        self.source_type = "none"
        self.source_info = {}
        self.source_display_name = "None"
        self.is_live_stream = False
        self.preserve_live_timing = True
        self.live_stream_fps = None
        self.stream_start_time = None
        self.frame_timestamps = []
        
        # Cleanup thread pool
        self.thread_pool.shutdown(wait=False)
        self.thread_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="CCTV")
        
        gc.collect()
        logger.info("Multi-source streaming stopped and resources cleaned")
    
    def move_ptz(self, pan, tilt, zoom):
        """PTZ movement (only available for RTSP/ONVIF sources)"""
        if self.ptz_moving or self.ptz_service is None or self.source_type != 'rtsp':
            return False
        
        try:
            self.ptz_moving = True
            
            pan *= self.ptz_speed_multiplier
            tilt *= self.ptz_speed_multiplier
            zoom *= self.ptz_speed_multiplier
            
            if pan == 0 and tilt == 0 and zoom == 0:
                request = self.ptz_service.create_type('GotoHomePosition')
                if self.media_profile:
                    request.ProfileToken = self.media_profile.token
                self.ptz_service.GotoHomePosition(request)
            else:
                request = self.ptz_service.create_type('ContinuousMove')
                if self.media_profile:
                    request.ProfileToken = self.media_profile.token
                    
                request.Velocity = {
                    'PanTilt': {'x': pan, 'y': tilt},
                    'Zoom': {'x': zoom}
                }
                
                self.ptz_service.ContinuousMove(request)
                time.sleep(0.3)
                
                stop_request = self.ptz_service.create_type('Stop')
                if self.media_profile:
                    stop_request.ProfileToken = self.media_profile.token
                stop_request.PanTilt = True
                stop_request.Zoom = True
                self.ptz_service.Stop(stop_request)
            
            return True
        
        except Exception as e:
            logger.error(f"PTZ move error: {e}")
            return False
        
        finally:
            self.ptz_moving = False
    
    def track_person_async(self, persons):
        """Asynchronous person tracking"""
        if not self.motion_tracking or not persons or self.ptz_service is None:
            return
        
        current_time = time.time()
        if current_time - self.last_tracking_time < self.tracking_cooldown:
            return
        
        try:
            with self.latest_frame_lock:
                if self.latest_frame is not None:
                    height, width = self.latest_frame.shape[:2]
                    center_frame_x, center_frame_y = width // 2, height // 2
                    
                    target_person = None
                    if self.last_person_position:
                        min_distance = float('inf')
                        for person in persons:
                            distance = np.sqrt((person[0] - self.last_person_position[0])**2 + 
                                             (person[1] - self.last_person_position[1])**2)
                            if distance < min_distance:
                                min_distance = distance
                                target_person = person
                    else:
                        target_person = persons[0]
                    
                    if target_person:
                        person_x, person_y = target_person
                        self.last_person_position = target_person
                        
                        pan = 0
                        tilt = 0
                        
                        diff_x = person_x - center_frame_x
                        diff_y = center_frame_y - person_y
                        
                        if abs(diff_x) > self.tracking_sensitivity:
                            pan = self.pan_speed if diff_x > 0 else -self.pan_speed
                            
                        if abs(diff_y) > self.tracking_sensitivity:
                            tilt = self.tilt_speed if diff_y > 0 else -self.tilt_speed
                        
                        if pan != 0 or tilt != 0:
                            success = self.move_ptz(pan, tilt, 0)
                            if success:
                                self.last_tracking_time = current_time
        
        except Exception as e:
            logger.error(f"Person tracking error: {e}")
    
    def update_performance_stats(self):
        """Update performance statistics"""
        current_time = time.time()
        
        if current_time - self.performance_stats['last_stats_time'] >= 1.0:
            time_diff = current_time - self.performance_stats['last_stats_time']
            
            self.performance_stats['stream_fps'] = self.performance_stats['frames_streamed'] / time_diff
            self.performance_stats['detection_fps'] = self.performance_stats['frames_detected'] / time_diff
            
            self.performance_stats['frames_streamed'] = 0
            self.performance_stats['frames_detected'] = 0
            self.performance_stats['last_stats_time'] = current_time
    
    def update_yolo_settings(self, confidence=None, input_size=None):
        """Update YOLOv8 detection settings"""
        settings_changed = False
        
        with self.yolo_settings_lock:
            if confidence is not None:
                old_confidence = self.yolo_confidence_threshold
                self.yolo_confidence_threshold = float(confidence)
                logger.info(f"✅ YOLOv8 confidence updated: {old_confidence:.2f} → {confidence:.2f}")
                settings_changed = True
            
            if input_size is not None:
                old_input_size = self.yolo_input_size
                self.yolo_input_size = int(input_size)
                logger.info(f"✅ YOLOv8 input size updated: {old_input_size}x{old_input_size} → {input_size}x{input_size}")
                settings_changed = True
                
                if self.yolo_model is not None and self.yolo_ready:
                    try:
                        logger.info(f"🔥 Re-warming YOLOv8 model with new input size: {input_size}x{input_size}")
                        dummy_image = np.zeros((self.yolo_input_size, self.yolo_input_size, 3), dtype=np.uint8)
                        self.yolo_model.predict(
                            dummy_image, 
                            verbose=False, 
                            device=self.yolo_device, 
                            imgsz=self.yolo_input_size,
                            conf=self.yolo_confidence_threshold
                        )
                        logger.info(f"✅ Model re-warmed successfully")
                    except Exception as e:
                        logger.warning(f"⚠️ Model re-warm failed: {e}")
        
        return settings_changed
    
    def get_yolo_model_info(self):
        """Get detailed YOLO model information"""
        if not self.yolo_ready:
            return {
                'available': False,
                'model_name': 'Not Available',
                'model_path': 'N/A',
                'device': 'N/A',
                'input_size': 'N/A'
            }
        
        return {
            'available': True,
            'model_name': self.yolo_model_name or 'Unknown',
            'model_path': self.yolo_model_path or 'N/A',
            'device': self.yolo_device,
            'input_size': f"{self.yolo_input_size}x{self.yolo_input_size}"
        }

# Global multi-source CCTV system instance with enhanced error handling
try:
    print("🚀 Creating Multi-Source CCTV system instance...")
    cctv_system = MultiSourceCCTV()
    if cctv_system.yolo_ready:
        print("✅ YOLOv8 AI Detection: READY")
        print(f"   📦 Model: {cctv_system.yolo_model_name}")
        print(f"   🖥️ Device: {cctv_system.yolo_device}")
    else:
        print("⚠️ YOLOv8 AI Detection: Using motion detection fallback")
        print("   💡 Place any .pt model file to enable AI detection")
except KeyboardInterrupt:
    print("\n🛑 Multi-Source CCTV system initialization interrupted by user")
    print("👋 Thank you for using Multi-Source CCTV System!")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Multi-Source CCTV system initialization failed: {e}")
    print("🔄 Please check the error and try again")
    sys.exit(1)

# Performance monitoring thread
def performance_monitor():
    """Monitor and update performance stats"""
    while True:
        try:
            cctv_system.update_performance_stats()
            time.sleep(1)
        except Exception as e:
            logger.error(f"Performance monitor error: {e}")
            time.sleep(5)

# Enhanced route handlers
@app.route('/')
def index():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Multi-Source CCTV System - YOLOv8</title></head>
        <body>
            <h1>Multi-Source CCTV System with YOLOv8</h1>
            <p>Please ensure index.html is in the same directory as app.py</p>
        </body>
        </html>
        """, 404

@app.route('/scan_webcams')
def scan_webcams():
    """Scan for available webcams"""
    try:
        webcams = cctv_system.scan_webcams()
        return jsonify({
            'success': True,
            'webcams': webcams,
            'count': len(webcams)
        })
    except Exception as e:
        logger.error(f"Webcam scan error: {e}")
        return jsonify({
            'success': False,
            'message': f"Webcam scan error: {str(e)}",
            'webcams': []
        })

@app.route('/connect_source', methods=['POST'])
def connect_source():
    """Enhanced connection endpoint dengan file timing preservation support"""
    data = request.json
    source_type = data.get('source_type', 'rtsp')
    
    logger.info(f"🔗 Multi-source connection request: {source_type.upper()}")
    
    # Enhanced file timing settings
    if source_type == 'file':
        file_url = data.get('file_url', '')
        preserve_file_timing = data.get('preserve_file_timing', True)
        auto_detect_type = data.get('auto_detect_type', 'auto')
        detected_type = data.get('detected_type', 'file')
        file_buffer_size = data.get('file_buffer_size', 'medium')
        
        logger.info(f"📁 File/URL connection settings:")
        logger.info(f"   • URL: {file_url}")
        logger.info(f"   • Preserve timing: {preserve_file_timing}")
        logger.info(f"   • Auto-detect: {auto_detect_type}")
        logger.info(f"   • Detected type: {detected_type}")
        logger.info(f"   • Buffer size: {file_buffer_size}")
        
        # Apply file-specific settings to CCTV system
        cctv_system.preserve_live_timing = preserve_file_timing
        
        # Set buffer size based on file type
        buffer_sizes = {
            'small': 1,    # For live-like URLs
            'medium': 3,   # For regular files
            'large': 5     # For local files
        }
        cctv_system.frame_buffer_size = buffer_sizes.get(file_buffer_size, 3)
        
        # Override source config with additional settings
        data.update({
            'preserve_file_timing': preserve_file_timing,
            'detected_type': detected_type,
            'file_buffer_size': file_buffer_size
        })
        
        if preserve_file_timing:
            logger.info("⏱️  File timing preservation enabled")
        
    # Live stream specific settings (existing code)
    elif source_type == 'stream':
        preserve_timing = data.get('preserve_timing', True)
        buffer_size = data.get('buffer_size', 'minimal')
        connection_timeout = data.get('connection_timeout', 15000)
        
        logger.info(f"🔴 Live stream settings:")
        logger.info(f"   • Preserve timing: {preserve_timing}")
        logger.info(f"   • Buffer size: {buffer_size}")
        logger.info(f"   • Timeout: {connection_timeout}ms")
        
        if preserve_timing:
            cctv_system.preserve_live_timing = True
            logger.info("⏱️  Live timing preservation enabled")
        
        buffer_sizes = {
            'minimal': 1,
            'small': 2,
            'medium': 3
        }
        cctv_system.frame_buffer_size = buffer_sizes.get(buffer_size, 1)
        cctv_system.connection_timeout = connection_timeout
    
    success, message = cctv_system.connect_source(data)
    
    if success:
        cctv_system.start_streaming()
        
        # Enhanced response dengan file timing info
        response_data = {
            'success': success, 
            'message': message,
            'source_type': source_type,
            'is_live_stream': cctv_system.is_live_stream,
            'preserve_timing': cctv_system.preserve_live_timing
        }
        
        # Add FPS info if available
        if hasattr(cctv_system, 'live_stream_fps') and cctv_system.live_stream_fps:
            response_data['original_fps'] = cctv_system.live_stream_fps
            if cctv_system.is_live_stream:
                logger.info(f"📊 Live stream FPS: {cctv_system.live_stream_fps}")
            else:
                logger.info(f"📊 File original FPS: {cctv_system.live_stream_fps}")
        
        logger.info(f"✅ {source_type.upper()} source connected and streaming started")
        return jsonify(response_data)
    else:
        return jsonify({
            'success': success, 
            'message': message,
            'source_type': source_type
        })

@app.route('/video_feed')
def video_feed():
    return Response(cctv_system.stream_generator_ultra_low_latency(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ptz_move', methods=['POST'])
def ptz_move():
    data = request.json
    success = cctv_system.move_ptz(
        float(data['pan']),
        float(data['tilt']),
        float(data['zoom'])
    )
    return jsonify({'success': success})

@app.route('/toggle_tracking', methods=['POST'])
def toggle_tracking():
    cctv_system.motion_tracking = not cctv_system.motion_tracking
    logger.info(f"Motion tracking toggled: {cctv_system.motion_tracking}")
    
    return jsonify({
        'success': True, 
        'tracking_enabled': cctv_system.motion_tracking
    })

@app.route('/toggle_detection_overlay', methods=['POST'])
def toggle_detection_overlay():
    """Toggle detection overlay visibility"""
    cctv_system.show_detection_overlay = not cctv_system.show_detection_overlay
    logger.info(f"Detection overlay toggled: {'ON' if cctv_system.show_detection_overlay else 'OFF'}")
    
    return jsonify({
        'success': True, 
        'overlay_enabled': cctv_system.show_detection_overlay,
        'message': f"Detection overlay {'enabled' if cctv_system.show_detection_overlay else 'disabled'}"
    })

@app.route('/toggle_yolo', methods=['POST'])
def toggle_yolo():
    if not cctv_system.yolo_ready:
        return jsonify({
            'success': False, 
            'message': 'YOLOv8 AI not available - ultralytics not installed or no .pt model found',
            'yolo_enabled': False
        })
    
    cctv_system.yolo_enabled = not cctv_system.yolo_enabled
    # PENTING: Aktifkan detection saat YOLOv8 diaktifkan
    cctv_system.detection_enabled = cctv_system.yolo_enabled or cctv_system.motion_tracking
    
    logger.info(f"YOLOv8 AI detection toggled: {cctv_system.yolo_enabled}")
    logger.info(f"Detection enabled: {cctv_system.detection_enabled}")
    
    return jsonify({
        'success': True, 
        'yolo_enabled': cctv_system.yolo_enabled,
        'detection_enabled': cctv_system.detection_enabled
    })

@app.route('/status')
def status():
    """Enhanced status endpoint dengan file timing information"""
    is_connected = cctv_system.cap is not None and cctv_system.cap.isOpened()
    
    detection_method = 'None'
    if cctv_system.detection_enabled:
        detection_method = 'YOLOv8 AI Detection' if (cctv_system.yolo_ready and cctv_system.yolo_enabled) else 'Motion Detection'
    
    onvif_status_text = 'Not Available'
    if cctv_system.onvif_cam:
        if cctv_system.ptz_service and cctv_system.media_profile:
            onvif_status_text = f'Connected (Full PTZ) - {cctv_system.auth_method_used}'
        elif cctv_system.ptz_service:
            onvif_status_text = f'Connected (Limited PTZ) - {cctv_system.auth_method_used}'
        else:
            onvif_status_text = f'Connected (Basic) - {cctv_system.auth_method_used}'
    
    yolo_info = cctv_system.get_yolo_model_info()
    
    with cctv_system.yolo_settings_lock:
        current_yolo_input_size = cctv_system.yolo_input_size
        current_yolo_confidence = cctv_system.yolo_confidence_threshold
    
    # Base status response
    status_response = {
        'connected': is_connected,
        'tracking_enabled': cctv_system.motion_tracking,
        'streaming': cctv_system.is_running,
        'ptz_available': cctv_system.ptz_service is not None and cctv_system.source_type == 'rtsp',
        'yolo_available': cctv_system.yolo_ready,
        'yolo_enabled': cctv_system.yolo_enabled,
        'show_detection_overlay': cctv_system.show_detection_overlay,  # NEW: Detection overlay status
        'person_count': cctv_system.person_count,
        'detection_method': detection_method,
        'onvif_status': onvif_status_text,
        'fps': round(cctv_system.performance_stats['stream_fps'], 1),
        'quality': cctv_system.stream_quality,
        'yolo_device': cctv_system.yolo_device if cctv_system.yolo_ready else 'N/A',
        'yolo_confidence': current_yolo_confidence,
        'yolo_input_size': current_yolo_input_size,
        'yolo_model_name': yolo_info['model_name'],
        'yolo_model_path': yolo_info['model_path'],
        'source_type': cctv_system.source_type,
        'source_display_name': cctv_system.source_display_name,
        'preserve_timing': cctv_system.preserve_live_timing
    }
    
    # Enhanced timing information untuk semua source type
    if hasattr(cctv_system, 'is_live_stream'):
        status_response['is_live_stream'] = cctv_system.is_live_stream
        
        if cctv_system.is_live_stream:
            # Live stream specific info
            status_response.update({
                'frame_timestamps_count': len(cctv_system.frame_timestamps) if hasattr(cctv_system, 'frame_timestamps') else 0,
                'live_stream_uptime': time.time() - cctv_system.stream_start_time if cctv_system.stream_start_time else 0
            })
            
            if len(cctv_system.frame_timestamps) >= 2:
                recent_timestamps = cctv_system.frame_timestamps[-5:]
                avg_interval = sum(recent_timestamps[i] - recent_timestamps[i-1] 
                                 for i in range(1, len(recent_timestamps))) / (len(recent_timestamps) - 1)
                expected_interval = 1.0 / (cctv_system.live_stream_fps or 25.0)
                timing_accuracy = abs(avg_interval - expected_interval) / expected_interval * 100
                status_response['timing_accuracy'] = round(100 - timing_accuracy, 1)
        
        else:
            # File atau source lain
            status_response['is_live_stream'] = False
            
            # Jika ini file dengan timing preservation
            if (cctv_system.source_type == 'file' and 
                cctv_system.preserve_live_timing and 
                hasattr(cctv_system, 'live_stream_fps') and 
                cctv_system.live_stream_fps):
                
                status_response.update({
                    'file_with_timing': True,
                    'file_uptime': time.time() - cctv_system.stream_start_time if cctv_system.stream_start_time else 0
                })
    
    # Add original FPS info jika tersedia
    if hasattr(cctv_system, 'live_stream_fps') and cctv_system.live_stream_fps:
        status_response['original_fps'] = cctv_system.live_stream_fps
    
    return jsonify(status_response)

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    logger.info(f"📤 Received settings update: {data}")
    
    if 'tracking_sensitivity' in data:
        cctv_system.tracking_sensitivity = int(data['tracking_sensitivity'])
    
    if 'ptz_speed' in data:
        cctv_system.ptz_speed_multiplier = float(data['ptz_speed'])
        
    if 'target_fps' in data:
        cctv_system.stream_fps = max(10, min(30, int(data['target_fps'])))
        cctv_system.stream_interval = 1.0 / cctv_system.stream_fps
    
    yolo_updated = False
    if 'yolo_confidence' in data:
        yolo_updated = cctv_system.update_yolo_settings(confidence=data['yolo_confidence']) or yolo_updated
    
    if 'yolo_input_size' in data:
        yolo_updated = cctv_system.update_yolo_settings(input_size=data['yolo_input_size']) or yolo_updated
    
    cctv_system.detection_enabled = cctv_system.yolo_enabled or cctv_system.motion_tracking
    
    with cctv_system.yolo_settings_lock:
        response_data = {
            'success': True,
            'yolo_updated': yolo_updated,
            'current_yolo_input_size': cctv_system.yolo_input_size,
            'current_yolo_confidence': cctv_system.yolo_confidence_threshold,
            'message': 'Settings updated successfully'
        }
    
    return jsonify(response_data)

@app.route('/update_file_timing', methods=['POST'])
def update_file_timing():
    """Update file timing settings on-the-fly"""
    data = request.json
    
    if cctv_system.source_type != 'file':
        return jsonify({
            'success': False,
            'message': 'Not connected to a file source'
        })
    
    updated_settings = {}
    
    # Update preserve timing setting
    if 'preserve_timing' in data:
        cctv_system.preserve_live_timing = bool(data['preserve_timing'])
        updated_settings['preserve_timing'] = cctv_system.preserve_live_timing
        logger.info(f"File timing preservation: {'enabled' if cctv_system.preserve_live_timing else 'disabled'}")
    
    # Update buffer size
    if 'buffer_size' in data:
        buffer_sizes = {
            'small': 1,
            'medium': 3,
            'large': 5
        }
        new_buffer_size = buffer_sizes.get(data['buffer_size'], 3)
        if cctv_system.cap:
            try:
                cctv_system.cap.set(cv2.CAP_PROP_BUFFERSIZE, new_buffer_size)
                cctv_system.frame_buffer_size = new_buffer_size
                updated_settings['buffer_size'] = data['buffer_size']
                logger.info(f"File buffer size updated: {data['buffer_size']}")
            except Exception as e:
                logger.warning(f"Could not update buffer size: {e}")
    
    return jsonify({
        'success': True,
        'message': 'File timing settings updated',
        'updated_settings': updated_settings,
        'current_fps': cctv_system.performance_stats['stream_fps'],
        'original_fps': cctv_system.live_stream_fps if hasattr(cctv_system, 'live_stream_fps') else None
    })

@app.route('/disconnect', methods=['POST'])
def disconnect():
    try:
        logger.info("Disconnect request received")
        cctv_system.stop_streaming()
        
        # Clear ONVIF connections
        cctv_system.onvif_cam = None
        cctv_system.ptz_service = None
        cctv_system.media_service = None
        cctv_system.media_profile = None
        cctv_system.auth_method_used = ""
        
        # Reset system state
        cctv_system.motion_tracking = False
        cctv_system.detection_enabled = False
        cctv_system.show_detection_overlay = True  # Reset to default
        cctv_system.person_count = 0
        cctv_system.last_person_position = None
        
        logger.info("Multi-source disconnected successfully")
        return jsonify({
            'success': True, 
            'message': 'Multi-source disconnected successfully'
        })
    
    except Exception as e:
        logger.error(f"Disconnect error: {e}")
        return jsonify({
            'success': False, 
            'message': f'Disconnect error: {str(e)}'
        })

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected to WebSocket')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected from WebSocket')

def background_status_update():
    """Background status updates"""
    while True:
        try:
            if cctv_system.is_running:
                yolo_info = cctv_system.get_yolo_model_info()
                
                with cctv_system.yolo_settings_lock:
                    current_yolo_input_size = cctv_system.yolo_input_size
                
                socketio.emit('status_update', {
                    'connected': cctv_system.cap is not None and cctv_system.cap.isOpened(),
                    'tracking_enabled': cctv_system.motion_tracking,
                    'streaming': cctv_system.is_running,
                    'ptz_available': cctv_system.ptz_service is not None and cctv_system.source_type == 'rtsp',
                    'yolo_available': cctv_system.yolo_ready,
                    'yolo_enabled': cctv_system.yolo_enabled,
                    'show_detection_overlay': cctv_system.show_detection_overlay,
                    'person_count': cctv_system.person_count,
                    'fps': round(cctv_system.performance_stats['stream_fps'], 1),
                    'quality': cctv_system.stream_quality,
                    'yolo_device': cctv_system.yolo_device if cctv_system.yolo_ready else 'N/A',
                    'yolo_model_name': yolo_info['model_name'],
                    'yolo_input_size': current_yolo_input_size,
                    'source_type': cctv_system.source_type,
                    'source_display_name': cctv_system.source_display_name,
                    'is_live_stream': getattr(cctv_system, 'is_live_stream', False),
                    'original_fps': getattr(cctv_system, 'live_stream_fps', None),
                    'preserve_timing': cctv_system.preserve_live_timing
                })
            time.sleep(2)
        except Exception as e:
            logger.error(f"Background status update error: {e}")
            time.sleep(5)

# Enhanced shutdown handler
def server_shutdown_handler(signum, frame):
    print("\n")
    print("🛑 Server shutdown signal received!")
    print("🔄 Gracefully shutting down Multi-Source CCTV system...")
    
    try:
        if cctv_system.is_running:
            cctv_system.stop_streaming()
        print("📹 All sources stopped")
        print("🌐 Web server stopping...")
        print("✅ Multi-Source CCTV system shutdown completed")
        print("👋 Thank you for using Multi-Source CCTV System!")
    except Exception as e:
        print(f"⚠️ Shutdown error: {e}")
    
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, server_shutdown_handler)
    signal.signal(signal.SIGTERM, server_shutdown_handler)
    
    print("=" * 90)
    print("🚀 MULTI-SOURCE CCTV SYSTEM - Enhanced with Universal .pt Model Support")
    print("=" * 90)
    
    print("\n📡 SUPPORTED VIDEO SOURCES:")
    print("   🎥 RTSP/IP Cameras with ONVIF PTZ control ✅")
    print("   📹 USB/Built-in Webcams with auto-detection ✅")
    print("   📺 Live Streaming URLs (YouTube, Twitch, HLS) ✅")
    print("   📁 Video Files (MP4, AVI, MOV, etc.) with proper timing ✅")
    print("   🌐 HTTP/HTTPS Direct Streams ✅")
    print("   📡 DASH Adaptive Streams ✅")
    
    print("\n⏱️ FIXED FILE TIMING PRESERVATION:")
    print("   • Auto-detects File vs Live Stream URLs ✅")
    print("   • Preserves original playback speed for files ✅")
    print("   • Maintains natural FPS from source ✅")
    print("   • Separate timing controls for each source type ✅")
    print("   • Fixed fast playback issue ✅")
    print("   • Original timing indicator ✅")
    
    print("\n🔴 LIVE STREAM TIMING PRESERVATION:")
    print("   • Auto-detects live streams from URL patterns ✅")
    print("   • Preserves original FPS and timing ✅")
    print("   • Minimal buffering for ultra-low latency ✅")
    print("   • Real-time timing drift detection ✅")
    print("   • Adaptive frame timing based on source ✅")
    print("   • Stream health monitoring ✅")
    
    print("\n🎯 ENHANCED FEATURES:")
    print("   • Smart webcam detection with device info ✅")
    print("   • YouTube Live stream extraction ✅")
    print("   • Twitch stream support ✅")
    print("   • HLS (.m3u8) stream support ✅")
    print("   • Video file looping for testing ✅")
    print("   • Auto stream type detection ✅")
    print("   • Source-specific optimizations ✅")
    print("   • Enhanced error handling per source type ✅")
    print("   • File timing preservation controls ✅")
    print("   • Fixed YOLOv8 import timeout issues ✅")
    print("   • Toggleable detection overlay (hide/show bounding boxes) ✅")
    
    if YT_DLP_AVAILABLE:
        print("   • yt-dlp integration for enhanced streaming ✅")
    else:
        print("   • yt-dlp integration: Install with 'pip install yt-dlp' ⚠️")
    
    print("\n🤖 YOLOv8 AI DETECTION (All Sources):")
    if YOLO_AVAILABLE:
        yolo_info = cctv_system.get_yolo_model_info()
        print(f"   • Model: {yolo_info['model_name']} ✅")
        print(f"   • Device: {yolo_info['device']} ✅")
        print(f"   • Input Size: {yolo_info['input_size']} (Dynamic) ✅")
        print(f"   • Auto-detects official YOLO models (yolo11n.pt, yolov8n.pt, etc.) ✅")
        print(f"   • Supports ANY .pt file (custom models, fine-tuned, etc.) ✅")
        print(f"   • Smart model selection with priority system ✅")
        print(f"   • Real-time object detection on any source ✅")
        print(f"   • Confidence threshold adjustment ✅")
        print(f"   • GPU acceleration (if available) ✅")
        print(f"   • Enhanced import timeout handling ✅")
        print(f"   • Toggleable detection overlay (hide/show boxes) ✅")
    else:
        print("   • Install with: pip install ultralytics ⚠️")
        print("   • Will use motion detection fallback ✅")
        print("   • Fixed import timeout issues ✅")
        print("   • Auto-detects ANY .pt model files ✅")
        print("   • Toggleable detection overlay (hide/show boxes) ✅")
    
    print("\n⚡ ULTRA LOW-LATENCY STREAMING:")
    print("   • Separated capture and streaming threads ✅")
    print("   • Minimal frame buffering ✅")
    print("   • Optimized JPEG encoding ✅")
    print("   • Source-specific optimizations ✅")
    print("   • Real-time performance monitoring ✅")
    print("   • Original timing preservation for all sources ✅")
    
    print("\n🕹️ PTZ CONTROL (RTSP/ONVIF Sources):")
    print("   • Smart ONVIF discovery across multiple ports ✅")
    print("   • Auto authentication detection ✅")
    print("   • Pan/Tilt/Zoom control ✅")
    print("   • Auto person tracking ✅")
    print("   • Adjustable tracking sensitivity ✅")
    
    print("\n🌐 WEB INTERFACE:")
    print("   • Source type selection with templates ✅")
    print("   • Live webcam detection and selection ✅")
    print("   • Stream URL templates (YouTube, Twitch, etc.) ✅")
    print("   • Real-time performance monitoring ✅")
    print("   • Responsive design for mobile/desktop ✅")
    print("   • Fullscreen video viewing ✅")
    print("   • File timing controls ✅")
    print("   • Toggleable detection overlay (show/hide boxes) ✅")
    
    print("\n💡 QUICK START EXAMPLES:")
    print("   📹 RTSP: rtsp://192.168.1.100:554/stream1")
    print("   🎥 Webcam: Select from auto-detected list")
    print("   📺 YouTube: https://www.youtube.com/watch?v=VIDEO_ID")
    print("   🎮 Twitch: https://www.twitch.tv/CHANNEL_NAME")
    print("   🌐 HLS: https://example.com/stream.m3u8")
    print("   📁 File: /path/to/video.mp4 or http://example.com/video.mp4")
    
    print("\n🤖 AI MODEL SUPPORT:")
    print("   📦 Official YOLO models: yolo11n.pt, yolov8n.pt, yolov5n.pt, etc.")
    print("   🎯 Custom trained models: my_model.pt, best.pt, custom_weights.pt")
    print("   📂 Auto-detection in: ./models/, ./weights/, current directory")
    print("   ⚡ Smart priority system: Official > Custom > Size-based selection")
    print("   💾 Size filtering: 1MB-500MB (excludes corrupted/invalid files)")
    print("   🔍 Pattern matching: Prioritizes YOLO-like named files")
    
    print("\n📡 Server running at: http://0.0.0.0:4000")
    print("   🎥 Live Stream: http://0.0.0.0:4000/video_feed")
    print("   🛑 Safe to interrupt anytime with Ctrl+C")
    print("=" * 90)
    
    # Start performance monitor thread
    try:
        monitor_thread = threading.Thread(target=performance_monitor, daemon=True)
        monitor_thread.start()
        
        # Start background status update thread
        status_thread = threading.Thread(target=background_status_update, daemon=True)
        status_thread.start()
        
        print("🚀 Starting Multi-Source Flask-SocketIO server...")
        print("⏱️ File Timing Preservation: ENABLED")
        print("   • Original playback speed for all file types ✅")
        print("   • Auto-detection File vs Live Stream URLs ✅")
        print("   • Fixed fast playback issue ✅")
        print("   • Enhanced YOLOv8 import handling ✅")
        print("🔲 Detection Overlay Toggle: ENABLED")
        print("   • Hide/show bounding boxes while keeping detection active ✅")
        print("   • Independent control for clean video view ✅")
        print("   • Background processing continues regardless ✅")
        print("🤖 Universal .pt Model Support: ENABLED")
        print("   • Official YOLO models (yolo11n.pt, yolov8n.pt, etc.) ✅")
        print("   • Custom .pt files with ANY name (my_model.pt, best.pt, etc.) ✅")
        print("   • Smart priority detection system ✅")
        print("   • Size validation and filtering ✅")
        print("   • Cross-platform model discovery ✅")
        
        socketio.run(app, host='0.0.0.0', port=4000, debug=False)
        
    except KeyboardInterrupt:
        print("\n🛑 Server interrupted by user")
        print("👋 Thank you for using Multi-Source CCTV System!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server startup error: {e}")
        print(f"\n❌ Server failed to start: {e}")
        sys.exit(1)