
import os
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, url_for
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import io
import base64
from datetime import datetime
import numpy as np
import json
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus import Image as ReportLabImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Import custom modules
from models_module import BuildingDetectionModel, SolarPanelDetectionModel

# Lazy import geoai to avoid early import conflicts
geoai = None
def get_geoai():
    """Lazy load geoai module when needed"""
    global geoai
    if geoai is None:
        try:
            import geoai as _geoai
            geoai = _geoai
            print("✓ geoai module loaded successfully")
        except Exception as e:
            print(f"✗ Error loading geoai module: {e}")
            raise
    return geoai

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Configure upload folders
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Model paths
BUILDING_MODEL_PATH = 'models/deeplabv3plus_buildings_state_20250930_021413.pth'
SOLAR_PANEL_MODEL_PATH = 'models/best_model.pth'

# Initialize models (loaded once at startup)
building_model = None
solar_panel_model = None

try:
    print("="*60)
    print("Loading Building Detection Model (DeepLab V3+)...")
    if os.path.exists(BUILDING_MODEL_PATH):
        # Check if file is valid (not a stub)
        file_size = os.path.getsize(BUILDING_MODEL_PATH)
        if file_size > 1000:  # Must be larger than 1KB to be a real model
            building_model = BuildingDetectionModel(BUILDING_MODEL_PATH)
            print("✓ Building detection model loaded successfully!")
        else:
            print(f"⚠ Warning: Building model file is too small ({file_size} bytes) - appears to be a stub file")
            print(f"   The actual trained model weights are not available.")
            print(f"   Building detection will be DISABLED.")
    else:
        print(f"⚠ Warning: Building model not found at {BUILDING_MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading building model: {str(e)}")
    print(f"   Building detection will be DISABLED.")
    import traceback
    traceback.print_exc()

try:
    print("="*60)
    print("Loading Solar Panel Detection Model (RCNN)...")
    if os.path.exists(SOLAR_PANEL_MODEL_PATH):
        solar_panel_model = SolarPanelDetectionModel(SOLAR_PANEL_MODEL_PATH)
        print("✓ Solar panel detection model loaded successfully!")
    else:
        print(f"⚠ Warning: Solar panel model not found at {SOLAR_PANEL_MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading solar panel model: {str(e)}")
    import traceback
    traceback.print_exc()

print("="*60)


def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_solar_panel_visualization(input_path, masks_path, gdf, output_path, max_dimension=2048):
    """Create a visualization of solar panel detections overlaid on the original image."""
    try:
        # Read the original image metadata first
        with rasterio.open(input_path) as src:
            original_height = src.height
            original_width = src.width
            
            # Calculate scaling factor BEFORE reading full image
            scale_factor = 1.0
            if max(original_height, original_width) > max_dimension:
                scale_factor = max_dimension / max(original_height, original_width)
            
            new_height = int(original_height * scale_factor)
            new_width = int(original_width * scale_factor)
            
            print(f"Original size: {original_width}x{original_height}, scaled to: {new_width}x{new_height} ({scale_factor:.4f})")
            
            # Read image with downsampling
            if scale_factor < 1.0:
                img = src.read(
                    [1, 2, 3],
                    out_shape=(3, new_height, new_width),
                    resampling=rasterio.enums.Resampling.bilinear
                )
            else:
                img = src.read([1, 2, 3])
            
            img = np.transpose(img, (1, 2, 0))
            
            # Normalize to 0-255 for display
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img_normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_normalized = img.astype(np.uint8)
        
        # Read the masks and downsample
        with rasterio.open(masks_path) as src:
            if scale_factor < 1.0:
                masks = src.read(
                    1,
                    out_shape=(new_height, new_width),
                    resampling=rasterio.enums.Resampling.nearest
                )
            else:
                masks = src.read(1)
        
        # Set matplotlib backend to Agg to avoid GUI issues
        plt.switch_backend('Agg')
        
        # Create figure with fixed size
        fig = plt.figure(figsize=(20, 10), dpi=100)
        gs = fig.add_gridspec(1, 2, hspace=0.05, wspace=0.05)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Left: Original image only
        ax1.imshow(img_normalized)
        ax1.set_title('Original Image', fontsize=18, fontweight='bold', pad=20)
        ax1.axis('off')
        
        # Right: Original image with highlighted detections
        ax2.imshow(img_normalized)
        
        # Create colored overlay for detected solar panels
        if masks is not None and np.any(masks > 0):
            overlay = np.zeros((*masks.shape, 4), dtype=np.float32)
            mask_binary = masks > 0
            overlay[mask_binary] = [1.0, 0.8, 0.0, 0.6]  # Yellow with transparency
            ax2.imshow(overlay)
        
        # Draw bounding boxes and labels
        if gdf is not None and len(gdf) > 0 and 'geometry' in gdf.columns:
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                
                bounds = geom.bounds
                x, y, x2, y2 = bounds
                
                # Scale coordinates if image was downsampled
                x *= scale_factor
                y *= scale_factor
                x2 *= scale_factor
                y2 *= scale_factor
                
                width = x2 - x
                height = y2 - y
                
                if width < 2 or height < 2:
                    continue
                
                rect = patches.Rectangle(
                    (x, y), width, height,
                    linewidth=3, edgecolor='#00FF00', facecolor='none',
                    linestyle='-'
                )
                ax2.add_patch(rect)
                
                confidence = row.get('confidence', 1.0)
                label_text = f"Panel #{idx+1}"
                if confidence < 1.0:
                    label_text += f"\n{confidence*100:.0f}%"
                
                font_size = max(6, min(11, int(11 * scale_factor * 2)))
                
                ax2.text(
                    x + width/2, y - 5, label_text,
                    color='white', fontsize=font_size, fontweight='bold',
                    ha='center', va='bottom',
                    bbox=dict(
                        boxstyle='round,pad=0.5',
                        facecolor='#00FF00',
                        edgecolor='white',
                        linewidth=2,
                        alpha=0.9
                    )
                )
        
        detection_count = len(gdf) if gdf is not None else 0
        ax2.set_title(
            f'Detected Solar Panels - {detection_count} Found',
            fontsize=18, fontweight='bold', pad=20,
            color='#00AA00'
        )
        ax2.axis('off')
        
        title_text = 'Solar Panel Detection Results'
        if scale_factor < 1.0:
            title_text += f' (Image scaled to {int(scale_factor*100)}% for visualization)'
        fig.suptitle(title_text, fontsize=20, fontweight='bold', y=0.98)
        
        fig.savefig(output_path, format='png', dpi=100, facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"Visualization saved successfully to {output_path}")
        return True
    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/')
def index():
    """Render the main upload page"""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """
    Main detection endpoint - performs sequential detection:
    1. Building detection first
    2. Solar panel detection if buildings are found
    """
    try:
        # Clear any cached tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Check if request has files
        try:
            files = request.files
        except Exception as e:
            print(f"Error reading request files: {e}")
            return jsonify({'error': 'Failed to read uploaded file. The file may be too large or the connection was interrupted. Please try with a smaller file or check your network connection.'}), 400
        
        if 'file' not in files:
            return jsonify({'error': 'No file part'}), 400
        
        file = files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or TIFF images.'}), 400
        
        # Check if building model is available
        building_model_available = building_model is not None and building_model.model is not None
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name, ext = os.path.splitext(filename)
        unique_filename = f"{base_name}_{timestamp}{ext}"
        
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(upload_path)
        
        # Convert original image to PNG for browser compatibility
        original_display_filename = f"original_{base_name}_{timestamp}.png"
        original_display_path = os.path.join(app.config['UPLOAD_FOLDER'], original_display_filename)
        
        try:
            original_img = Image.open(upload_path).convert('RGB')
            original_img.save(original_display_path, 'PNG')
        except Exception as e:
            print(f"Warning: Could not convert original image to PNG: {e}")
            original_display_filename = unique_filename
        
        # STEP 1: Building Detection
        print(f"\n{'='*60}")
        print(f"STEP 1: Detecting Buildings...")
        print(f"{'='*60}")
        
        if building_model_available:
            building_result = building_model.predict(upload_path)
        else:
            # Skip building detection if model not available
            print("⚠ Building model not available - skipping building detection")
            print("   Assuming buildings are present and proceeding to solar panel detection")
            building_result = None
        
        # Save building detection results
        if building_result is not None:
            building_result_filename = f"buildings_{base_name}_{timestamp}.png"
            building_overlay_filename = f"buildings_overlay_{base_name}_{timestamp}.png"
            
            building_result_path = os.path.join(app.config['RESULT_FOLDER'], building_result_filename)
            building_overlay_path = os.path.join(app.config['RESULT_FOLDER'], building_overlay_filename)
            
            building_result['prediction_image'].save(building_result_path)
            building_result['overlay_image'].save(building_overlay_path)
            
            building_percentage = building_result['building_percentage']
            buildings_detected = building_percentage > 1.0  # Threshold: 1% of image contains buildings
        else:
            # No building detection - assume buildings present
            building_percentage = 100.0
            buildings_detected = True
            building_result_filename = None
            building_overlay_filename = None
        
        print(f"\nBuilding Detection Results:")
        print(f"  - Building Coverage: {building_percentage:.2f}%")
        print(f"  - Buildings Detected: {'YES' if buildings_detected else 'NO'}")
        
        # Prepare response data
        response_data = {
            'success': True,
            'original_image': url_for('static', filename=f'uploads/{original_display_filename}'),
            'building_detection': {},
            'solar_panel_detection': None
        }
        
        if building_result is not None:
            response_data['building_detection'] = {
                'detected': bool(buildings_detected),
                'prediction_mask': url_for('static', filename=f'results/{building_result_filename}'),
                'overlay_image': url_for('static', filename=f'results/{building_overlay_filename}'),
                'statistics': {
                    'building_pixels': int(building_result['building_pixels']),
                    'total_pixels': int(building_result['total_pixels']),
                    'building_percentage': float(building_percentage),
                    'confidence_mean': float(building_result['confidence_mean']),
                    'image_size': [int(building_result['image_size'][0]), int(building_result['image_size'][1])]
                }
            }
        else:
            # Building model not available - skip building detection
            response_data['building_detection'] = {
                'detected': True,
                'skipped': True,
                'message': 'Building detection model not available. Assuming buildings present.',
                'statistics': {
                    'building_pixels': 0,
                    'total_pixels': 0,
                    'building_percentage': 100.0,
                    'confidence_mean': 1.0,
                    'image_size': [0, 0]
                }
            }
        
        # STEP 2: Solar Panel Detection (only if buildings detected)
        if buildings_detected:
            print(f"\n{'='*60}")
            print(f"STEP 2: Buildings Found! Detecting Solar Panels...")
            print(f"{'='*60}")
            
            if solar_panel_model is None:
                print("⚠ Warning: Solar panel model not loaded. Skipping solar panel detection.")
                response_data['solar_panel_detection'] = {
                    'error': 'Solar panel model not loaded',
                    'detected': False
                }
            else:
                # Check if file is TIFF for geoai processing
                is_tiff = ext.lower() in ['.tif', '.tiff']
                
                if is_tiff:
                    # Use geoai for TIFF files
                    try:
                        # Get parameters from request
                        confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
                        window_size = int(request.form.get('window_size', 512))
                        overlap = int(request.form.get('overlap', 256))
                        
                        # Generate output filenames
                        masks_path = os.path.join(app.config['RESULT_FOLDER'], f"solar_masks_{base_name}_{timestamp}.tif")
                        output_vector_path = os.path.join(app.config['RESULT_FOLDER'], f"solar_{base_name}_{timestamp}.geojson")
                        visualization_path = os.path.join(app.config['RESULT_FOLDER'], f"solar_viz_{base_name}_{timestamp}.png")
                        
                        # Run object detection
                        geoai_module = get_geoai()  # Lazy load geoai
                        geoai_module.object_detection(
                            upload_path,
                            masks_path,
                            SOLAR_PANEL_MODEL_PATH,
                            window_size=window_size,
                            overlap=overlap,
                            confidence_threshold=confidence_threshold,
                            batch_size=4,
                            num_channels=3,
                        )
                        
                        # Check if masks file was created
                        if os.path.exists(masks_path):
                            # Vectorize masks
                            try:
                                gdf = geoai_module.orthogonalize(masks_path, output_vector_path, epsilon=2)
                                
                                # Validate GeoDataFrame has geometry
                                if gdf is not None and len(gdf) > 0:
                                    # Ensure geometry column exists and is valid
                                    if not hasattr(gdf, 'geometry') or 'geometry' not in gdf.columns:
                                        raise ValueError("GeoDataFrame missing geometry column")
                                    if gdf.geometry.isna().all():
                                        raise ValueError("GeoDataFrame has no valid geometries")
                                
                                if gdf is not None and len(gdf) > 0 and 'geometry' in gdf.columns:
                                    # Create visualization
                                    create_solar_panel_visualization(upload_path, masks_path, gdf, visualization_path)
                                    
                                    # Extract geographic information from TIFF
                                    geo_info = {}
                                    try:
                                        with rasterio.open(upload_path) as src:
                                            geo_info = {
                                                'crs': str(src.crs) if src.crs else 'Unknown',
                                                'bounds': list(src.bounds) if src.bounds else None,
                                                'transform': list(src.transform)[:6] if src.transform else None
                                            }
                                    except Exception as geo_err:
                                        print(f"Warning: Could not extract geo info: {geo_err}")
                                    
                                    # Extract features with enhanced coordinate information
                                    features = []
                                    for idx, row in gdf.iterrows():
                                        geom = row.geometry
                                        if geom is None or geom.is_empty:
                                            continue
                                        
                                        confidence = row.get('confidence', 1.0)
                                        coordinates = list(geom.exterior.coords) if hasattr(geom, 'exterior') else []
                                        
                                        # Calculate centroid for lat/lon display
                                        centroid = geom.centroid
                                        center_x, center_y = float(centroid.x), float(centroid.y)
                                        
                                        # Get bounds
                                        bounds = geom.bounds  # (minx, miny, maxx, maxy)
                                        
                                        # Convert all values to JSON-serializable types
                                        features.append({
                                            'id': int(idx),
                                            'geometry_type': str(geom.geom_type),
                                            'coordinates': [[float(x), float(y)] for x, y in coordinates],
                                            'bounds': [float(b) for b in bounds],
                                            'centroid': {
                                                'x': center_x,
                                                'y': center_y
                                            },
                                            'area': float(geom.area),
                                            'confidence': float(confidence)
                                        })
                                    
                                    print(f"\n✓ Solar Panel Detection Complete: {len(features)} panels found")
                                    
                                    response_data['solar_panel_detection'] = {
                                        'detected': True,
                                        'count': len(features),
                                        'detections': features,
                                        'geo_info': geo_info,
                                        'visualization': url_for('static', filename=f'results/{os.path.basename(visualization_path)}'),
                                        'geojson_path': output_vector_path if os.path.exists(output_vector_path) else None
                                    }
                                else:
                                    print("\n⚠ No solar panels detected")
                                    response_data['solar_panel_detection'] = {
                                        'detected': False,
                                        'count': 0,
                                        'message': 'No solar panels detected'
                                    }
                            except Exception as vec_error:
                                print(f"✗ Vectorization error: {vec_error}")
                                import traceback
                                traceback.print_exc()
                                
                                # Provide more helpful error message for common issues
                                error_msg = str(vec_error)
                                if 'geometry column' in error_msg.lower() or 'crs' in error_msg.lower():
                                    # This typically means no solar panels were detected
                                    response_data['solar_panel_detection'] = {
                                        'detected': False,
                                        'count': 0,
                                        'message': 'No solar panels found'
                                    }
                                else:
                                    response_data['solar_panel_detection'] = {
                                        'detected': False,
                                        'error': f'Vectorization failed: {error_msg}'
                                    }
                        else:
                            response_data['solar_panel_detection'] = {
                                'detected': False,
                                'message': 'No masks generated'
                            }
                    except Exception as solar_error:
                        print(f"✗ Solar panel detection error: {solar_error}")
                        import traceback
                        traceback.print_exc()
                        response_data['solar_panel_detection'] = {
                            'detected': False,
                            'error': str(solar_error)
                        }
                else:
                    # For non-TIFF files, use simple model prediction
                    print("⚠ Solar panel detection requires TIFF format for full functionality")
                    response_data['solar_panel_detection'] = {
                        'detected': False,
                        'message': 'Solar panel detection requires TIFF/GeoTIFF format'
                    }
        else:
            print(f"\n⚠ No buildings detected. Skipping solar panel detection.")
        
        print(f"\n{'='*60}")
        print(f"Detection Complete!")
        print(f"{'='*60}\n")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"✗ Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'building_model_loaded': building_model is not None and building_model.model is not None,
        'solar_panel_model_loaded': solar_panel_model is not None
    }), 200


@app.route('/generate_pdf_report', methods=['POST'])
def generate_pdf_report():
    """Generate a PDF report of the detection results"""
    try:
        data = request.json
        
        # Create PDF filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"solar_detection_report_{timestamp}.pdf"
        pdf_path = os.path.join(app.config['RESULT_FOLDER'], pdf_filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter, 
                               topMargin=0.75*inch, bottomMargin=0.75*inch,
                               leftMargin=0.75*inch, rightMargin=0.75*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1e3c72'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#f2994a'),
            spaceAfter=15,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        )
        
        subheading_style = ParagraphStyle(
            'SubHeading',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#1e3c72'),
            spaceAfter=8,
            fontName='Helvetica-Bold'
        )
        
        # Title with colored box
        title_table = Table([["SolarScope Detection Report"]], colWidths=[6.5*inch])
        title_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1e3c72')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 24),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ]))
        story.append(title_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Report info
        report_info = [
            ['Report Generated:', datetime.now().strftime("%B %d, %Y at %H:%M:%S")],
            ['Original Image:', data.get('filename', 'N/A')]
        ]
        
        info_table = Table(report_info, colWidths=[2.5*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bbdefb')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.4*inch))
        
        # Original Image
        if data.get('original_image_path'):
            try:
                original_path = data.get('original_image_path')
                if os.path.exists(original_path):
                    story.append(Paragraph("Original Image", heading_style))
                    # Calculate proper aspect ratio
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(original_path)
                    img_width, img_height = pil_img.size
                    aspect = img_height / img_width
                    
                    # Set max dimensions
                    max_width = 5.5*inch
                    max_height = 4*inch
                    
                    if aspect > max_height / max_width:
                        img_height = max_height
                        img_width = img_height / aspect
                    else:
                        img_width = max_width
                        img_height = img_width * aspect
                    
                    img = ReportLabImage(original_path, width=img_width, height=img_height)
                    story.append(img)
                    story.append(Spacer(1, 0.3*inch))
            except Exception as img_err:
                print(f"Error adding original image: {img_err}")
        
        # Building Detection Results
        if data.get('building_detection'):
            building = data['building_detection']
            story.append(Paragraph("Building Detection Results", heading_style))
            
            if building.get('detected') and not building.get('skipped'):
                stats = building.get('statistics', {})
                building_data = [
                    ['Metric', 'Value'],
                    ['Buildings Detected', 'YES' if building.get('detected') else 'NO'],
                    ['Building Coverage', f"{stats.get('building_percentage', 0):.2f}%"],
                    ['Average Confidence', f"{stats.get('confidence_mean', 0)*100:.1f}%"],
                    ['Building Pixels', f"{stats.get('building_pixels', 0):,}"],
                    ['Total Pixels', f"{stats.get('total_pixels', 0):,}"],
                    ['Image Size', f"{stats.get('image_size', [0,0])[0]} x {stats.get('image_size', [0,0])[1]}"]
                ]
                
                building_table = Table(building_data, colWidths=[2.5*inch, 4*inch])
                building_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3c72')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#1e3c72')),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(building_table)
                story.append(Spacer(1, 0.3*inch))
                
                # Add building detection images
                try:
                    from PIL import Image as PILImage
                    images_added = False
                    
                    if building.get('prediction_mask_path'):
                        mask_path = building.get('prediction_mask_path')
                        if os.path.exists(mask_path):
                            story.append(Paragraph("Building Segmentation Mask", subheading_style))
                            pil_img = PILImage.open(mask_path)
                            img_width, img_height = pil_img.size
                            aspect = img_height / img_width
                            display_width = 4.5*inch
                            display_height = display_width * aspect
                            if display_height > 3.5*inch:
                                display_height = 3.5*inch
                                display_width = display_height / aspect
                            img = ReportLabImage(mask_path, width=display_width, height=display_height)
                            story.append(img)
                            story.append(Spacer(1, 0.25*inch))
                            images_added = True
                    
                    if building.get('overlay_image_path'):
                        overlay_path = building.get('overlay_image_path')
                        if os.path.exists(overlay_path):
                            story.append(Paragraph("Building Detection Overlay", subheading_style))
                            pil_img = PILImage.open(overlay_path)
                            img_width, img_height = pil_img.size
                            aspect = img_height / img_width
                            display_width = 4.5*inch
                            display_height = display_width * aspect
                            if display_height > 3.5*inch:
                                display_height = 3.5*inch
                                display_width = display_height / aspect
                            img = ReportLabImage(overlay_path, width=display_width, height=display_height)
                            story.append(img)
                            story.append(Spacer(1, 0.25*inch))
                            images_added = True
                            
                except Exception as img_err:
                    print(f"Error adding building images: {img_err}")
            else:
                story.append(Paragraph("Building detection was skipped or model not available.", styles['Normal']))
            
            story.append(Spacer(1, 0.3*inch))
        
        # Solar Panel Detection Results
        if data.get('solar_panel_detection'):
            solar = data['solar_panel_detection']
            story.append(Paragraph("Solar Panel Detection Results", heading_style))
            
            if solar.get('detected') and solar.get('count', 0) > 0:
                # Summary box
                summary_data = [['Total Panels Detected:', str(solar.get('count', 0))]]
                summary_table = Table(summary_data, colWidths=[3*inch, 3.5*inch])
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff9f0')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1e3c72')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 2, colors.HexColor('#f2994a')),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(summary_table)
                story.append(Spacer(1, 0.25*inch))
                
                # Geographic Info
                if solar.get('geo_info'):
                    geo_info = solar['geo_info']
                    story.append(Paragraph("Geographic Information:", subheading_style))
                    geo_data = [
                        ['Property', 'Value'],
                        ['Coordinate System', str(geo_info.get('crs', 'N/A'))],
                    ]
                    if geo_info.get('bounds'):
                        bounds = geo_info['bounds']
                        bounds_str = f"[{', '.join([f'{b:.6f}' for b in bounds])}]"
                        geo_data.append(['Image Bounds', bounds_str])
                    
                    geo_table = Table(geo_data, colWidths=[2.5*inch, 4*inch])
                    geo_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2196f3')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('FONTSIZE', (0, 1), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),
                        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#2196f3')),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    story.append(geo_table)
                    story.append(Spacer(1, 0.3*inch))
                
                # Panel Details
                detections = solar.get('detections', [])
                if detections:
                    story.append(Paragraph("Detected Solar Panels Details:", subheading_style))
                    story.append(Spacer(1, 0.15*inch))
                    
                    for idx, panel in enumerate(detections[:10]):  # Limit to first 10 panels for PDF
                        centroid_x = panel.get('centroid', {}).get('x', 0)
                        centroid_y = panel.get('centroid', {}).get('y', 0)
                        bounds = panel.get('bounds', [])
                        bounds_str = f"[{', '.join([f'{b:.2f}' for b in bounds])}]" if bounds else 'N/A'
                        
                        panel_data = [
                            [f"Panel #{idx + 1}", ""],
                            ['Center Coordinates', f"({centroid_x:.6f}, {centroid_y:.6f})"],
                            ['Bounds', bounds_str],
                            ['Area', f"{panel.get('area', 0):.2f} sq units"],
                            ['Confidence', f"{panel.get('confidence', 0)*100:.1f}%"]
                        ]
                        
                        panel_table = Table(panel_data, colWidths=[2.5*inch, 4*inch])
                        panel_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f2994a')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 11),
                            ('FONTSIZE', (0, 1), (-1, -1), 9),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                            ('TOPPADDING', (0, 0), (-1, -1), 8),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fff9f0')),
                            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#f2994a')),
                            ('SPAN', (0, 0), (-1, 0)),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ]))
                        story.append(panel_table)
                        story.append(Spacer(1, 0.2*inch))
                    
                    if len(detections) > 10:
                        story.append(Paragraph(f"<i>... and {len(detections) - 10} more panels</i>", styles['Italic']))
                        story.append(Spacer(1, 0.2*inch))
                
                # Add solar panel visualization image
                try:
                    if solar.get('visualization_path'):
                        viz_path = solar.get('visualization_path')
                        if os.path.exists(viz_path):
                            from PIL import Image as PILImage
                            story.append(Paragraph("Solar Panel Detection Visualization", subheading_style))
                            pil_img = PILImage.open(viz_path)
                            img_width, img_height = pil_img.size
                            aspect = img_height / img_width
                            display_width = 5.5*inch
                            display_height = display_width * aspect
                            if display_height > 4*inch:
                                display_height = 4*inch
                                display_width = display_height / aspect
                            img = ReportLabImage(viz_path, width=display_width, height=display_height)
                            story.append(img)
                            story.append(Spacer(1, 0.2*inch))
                except Exception as img_err:
                    print(f"Error adding solar visualization: {img_err}")
            else:
                story.append(Paragraph("No solar panels detected.", styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 0.4*inch))
        footer_data = [[
            "Generated by SolarScope - Detecting Rooftops Ready For Solar Energy",
            "Made with love by Team SolarScope",
            "Ayesha Shaikh, Meeti Shah, Neel Shah, Tushar Surti",
            "Under the guidance of Professor Kavita Bathe",
            "© 2025 SolarScope"
        ]]
        footer_table = Table(footer_data, colWidths=[6.5*inch])
        footer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#666666')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#cccccc')),
        ]))
        story.append(footer_table)
        
        # Build PDF
        doc.build(story)
        
        # Return the PDF file
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=pdf_filename
        )
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error generating PDF: {str(e)}'}), 500


@app.route('/about')
def about():
    """Render the about page with model information"""
    model_info = {
        'building_detection': {
            'architecture': 'DeepLab v3+ with ResNet-50 backbone',
            'classes': 'Binary (Building vs Background)',
            'input_size': '512x512',
            'framework': 'PyTorch'
        },
        'solar_panel_detection': {
            'architecture': 'Faster R-CNN',
            'classes': 'Solar Panels',
            'framework': 'PyTorch with geoai',
            'input_format': 'GeoTIFF'
        }
    }
    return render_template('about.html', model_info=model_info)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error. Please try again.'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return jsonify({'error': 'File too large. Maximum file size is 200MB.'}), 413


@app.errorhandler(Exception)
def handle_exception(error):
    """Handle all unhandled exceptions"""
    from werkzeug.exceptions import ClientDisconnected
    
    # Handle client disconnection gracefully
    if isinstance(error, ClientDisconnected):
        print("Client disconnected during upload")
        return jsonify({'error': 'Upload interrupted. The file may be too large or your connection was lost. Please try again with a smaller file.'}), 400
    
    print(f"Unhandled exception: {str(error)}")
    import traceback
    traceback.print_exc()
    return jsonify({'error': f'An unexpected error occurred: {str(error)}'}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Combined Building & Solar Panel Detection Web App")
    print("="*60)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Result folder: {RESULT_FOLDER}")
    if building_model:
        print(f"Building Model Device: {building_model.device}")
    if solar_panel_model:
        print(f"Solar Panel Model: Loaded")
    print("="*60 + "\n")
    
    # Get port from environment variable
    port = int(os.environ.get('PORT', 5001))
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=port)
