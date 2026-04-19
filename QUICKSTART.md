# 🚀 Quick Start Guide - Combined Detection System

This guide will help you set up and run the combined building and solar panel detection system in just a few minutes.

## ⚡ Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- 4GB+ RAM (8GB+ recommended)
- GPU with CUDA support (optional, but recommended for faster processing)

## 📦 Installation Steps

### 1. Navigate to the Project Directory

```powershell
cd e:\combine\combined-detection-app
```

### 2. Create a Virtual Environment (Recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

**Note:** Some packages may take time to install, especially PyTorch and its dependencies.

#### Special Installation Notes:

**For geoai library:**
```powershell
# If geoai is not available via pip, you may need to install from source
# or contact your organization for the package
pip install geoai
```

**For GPU Support (Optional):**
```powershell
# If you have NVIDIA GPU and CUDA installed, use:
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### 4. Copy Model Files

**Option A: Run the setup script (from parent directory):**
```powershell
cd ..
.\setup.ps1
cd combined-detection-app
```

**Option B: Manual copy:**
```powershell
# From the e:\combine directory
Copy-Item "building-detection-app-main\models\deeplabv3plus_buildings_state_20250930_021413.pth" -Destination "combined-detection-app\models\"
Copy-Item "output\models\best_model.pth" -Destination "combined-detection-app\models\"
```

### 5. Verify Setup

```powershell
python config.py
```

You should see:
```
✓ Created directories
✓ All model files found
```

## 🎮 Running the Application

### Start the Flask Server

```powershell
python app.py
```

You should see output like:
```
============================================================
Combined Building & Solar Panel Detection Web App
============================================================
Building Model - Using device: cuda (or cpu)
✓ Building detection model loaded successfully!
Solar Panel Model: Loaded
============================================================

 * Running on http://0.0.0.0:5000
```

### Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## 📸 Testing the System

### Using Sample Images

1. **Prepare a test image:**
   - For full functionality: Use a **GeoTIFF** file with building structures
   - For building detection only: Any PNG/JPG image will work

2. **Upload and detect:**
   - Click the upload area or drag & drop your image
   - Click "Start Sequential Detection"
   - Wait for results (may take 30 seconds to a few minutes)

3. **View results:**
   - Building detection: Segmentation mask and statistics
   - Solar panel detection: Bounding boxes and panel count (if buildings found)

## 🔍 Understanding the Results

### Building Detection
- **Green status**: Buildings detected (>1% coverage)
- **Yellow status**: No significant buildings found
- **Statistics**: Coverage %, confidence, pixel counts

### Solar Panel Detection
- **Only runs if buildings are detected**
- **Green status**: Solar panels found
- **Blue status**: No solar panels found
- **Shows**: Panel locations, confidence scores, GeoJSON data

## 🛠️ Troubleshooting

### "Model not loaded" Error
```powershell
# Check if model files exist
ls models\
# Should show both .pth files
```

### "Import geoai" Error
```powershell
# Try installing geoai
pip install geoai
# If not available, check with your organization
```

### Out of Memory
```powershell
# Edit config.py and reduce:
# - SOLAR_PANEL_WINDOW_SIZE (default: 512)
# - MAX_VISUALIZATION_DIMENSION (default: 2048)
```

### Port Already in Use
```powershell
# Change port in config.py or run:
$env:PORT=5001
python app.py
```

## 📊 File Size Limits

- Maximum upload size: **100 MB**
- Recommended image size: **<50 MB** for faster processing
- Larger images will be automatically downscaled for visualization

## 🔐 Security Note

The default `SECRET_KEY` in config.py should be changed in production:
```python
SECRET_KEY = 'your-unique-secret-key-here'
```

## 🎯 Next Steps

1. **Test with your own images**
2. **Adjust detection thresholds** in `config.py`
3. **Review results** and fine-tune parameters
4. **Export GeoJSON data** for GIS applications

## 💡 Tips for Best Results

1. **Use GeoTIFF format** for solar panel detection
2. **Ensure good image quality** (clear, high resolution)
3. **Avoid heavily clouded images**
4. **Buildings should be clearly visible** from aerial view
5. **Solar panels should be on building roofs** for best detection

## 📞 Getting Help

If you encounter issues:

1. Check the console output for detailed error messages
2. Review the main README.md for detailed documentation
3. Verify all dependencies are installed: `pip list`
4. Check model files are in the correct location
5. Ensure sufficient disk space and memory

## 🎉 Success Indicators

You know the system is working correctly when:
- ✓ Both models load without errors
- ✓ Health check endpoint returns "healthy": `http://localhost:5000/health`
- ✓ You can upload and process test images
- ✓ Results display with visualizations and statistics

---

**Happy Detecting! 🏢☀️**
