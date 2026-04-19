
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BUILDING_MODEL_PATH = os.path.join(MODELS_DIR, 'deeplabv3plus_buildings_state_20250930_021413.pth')
SOLAR_PANEL_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pth')

# Upload and result directories
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
RESULT_FOLDER = os.path.join(STATIC_DIR, 'results')

# Flask configuration
SECRET_KEY = 'your-secret-key-change-this-in-production'
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Detection parameters
BUILDING_DETECTION_THRESHOLD = 1.0  # Percentage of image that must contain buildings
BUILDING_CONFIDENCE_THRESHOLD = 0.5
SOLAR_PANEL_CONFIDENCE_THRESHOLD = 0.5
SOLAR_PANEL_WINDOW_SIZE = 512
SOLAR_PANEL_OVERLAP = 256

# Server configuration
DEFAULT_PORT = 5000
DEBUG_MODE = True  # Set to False in production

# Device configuration (will be auto-detected, but can be overridden)
# Auto-detect: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
# DEVICE = 'cuda'  # for NVIDIA GPU
# DEVICE = 'mps'   # for Apple Silicon
# DEVICE = 'cpu'   # for CPU only
DEVICE = None  # Auto-detect (recommended)

# Visualization settings
MAX_VISUALIZATION_DIMENSION = 2048  # Max dimension for visualization images
VISUALIZATION_DPI = 100

# Create directories if they don't exist
def create_directories():
    """Create necessary directories for the application"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    print(f"✓ Created directories:")
    print(f"  - Models: {MODELS_DIR}")
    print(f"  - Uploads: {UPLOAD_FOLDER}")
    print(f"  - Results: {RESULT_FOLDER}")

def check_models():
    """Check if required model files exist"""
    missing_models = []
    
    if not os.path.exists(BUILDING_MODEL_PATH):
        missing_models.append(f"Building model: {BUILDING_MODEL_PATH}")
    
    if not os.path.exists(SOLAR_PANEL_MODEL_PATH):
        missing_models.append(f"Solar panel model: {SOLAR_PANEL_MODEL_PATH}")
    
    if missing_models:
        print("⚠ Warning: Missing model files:")
        for model in missing_models:
            print(f"  - {model}")
        return False
    else:
        print("✓ All model files found")
        return True

if __name__ == "__main__":
    print("Configuration Check")
    print("="*60)
    create_directories()
    check_models()
    print("="*60)
