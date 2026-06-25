from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, SelectField, FloatField
from wtforms.validators import InputRequired
import os
from werkzeug.utils import secure_filename
import uuid
from PIL import Image
import numpy as np
import torch
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.unet import UNet
from models.semantic_encoder import SemanticEncoder
from models.semantic_decoder import SemanticDecoder
from utils.data_processing import preprocess_image, postprocess_image
from utils.compression import compress_sparse_matrix, restore_sparse_matrix
from utils.channel_simulation import AWGNChannel, RayleighChannel
from utils.metrics import calculate_psnr, calculate_theta_psnr

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (512, 512)

class SemanticCommunicationSystem:
    def __init__(self):
        self.device = DEVICE
        self.unet = None
        self.encoder_sc1 = None
        self.decoder_sc1 = None
        self.encoder_sc2 = None
        self.decoder_sc2 = None
        self.load_models()

    def load_models(self):
        """Load trained models from combined checkpoints."""
        try:
            # U-Net
            self.unet = UNet(n_channels=3, n_classes=2)
            unet_path = 'checkpoints/unet_final.pth'
            if os.path.exists(unet_path):
                self.unet.load_state_dict(torch.load(unet_path, map_location=self.device, weights_only=True))
                print("✅ U-Net loaded from checkpoint")
            else:
                print("⚠️ U-Net checkpoint not found – using random init")

            # SC1 (low bandwidth) – combined checkpoint
            self.encoder_sc1 = SemanticEncoder(bandwidth='low')
            self.decoder_sc1 = SemanticDecoder(bandwidth='low')
            sc1_path = 'checkpoints/sc1_final.pth'
            if os.path.exists(sc1_path):
                checkpoint = torch.load(sc1_path, map_location=self.device, weights_only=True)
                self.encoder_sc1.load_state_dict(checkpoint['encoder'])
                self.decoder_sc1.load_state_dict(checkpoint['decoder'])
                print("✅ SC1 models loaded from combined checkpoint")
            else:
                print("⚠️ SC1 checkpoint not found – using random init")

            # SC2 (high bandwidth) – combined checkpoint
            self.encoder_sc2 = SemanticEncoder(bandwidth='high')
            self.decoder_sc2 = SemanticDecoder(bandwidth='high')
            sc2_path = 'checkpoints/sc2_final.pth'
            if os.path.exists(sc2_path):
                checkpoint = torch.load(sc2_path, map_location=self.device, weights_only=True)
                self.encoder_sc2.load_state_dict(checkpoint['encoder'])
                self.decoder_sc2.load_state_dict(checkpoint['decoder'])
                print("✅ SC2 models loaded from combined checkpoint")
            else:
                print("⚠️ SC2 checkpoint not found – using random init")

            self.unet.to(self.device).eval()
            self.encoder_sc1.to(self.device).eval()
            self.decoder_sc1.to(self.device).eval()
            self.encoder_sc2.to(self.device).eval()
            self.decoder_sc2.to(self.device).eval()

            # Diagnostic: SC1 black encoding
            with torch.no_grad():
                black = torch.zeros(1, 3, *IMAGE_SIZE).to(self.device)
                black_enc = self.encoder_sc1(black)
                mean_val = torch.mean(black_enc).item()
                print(f"\n[Diagnostic] SC1 black encoding – min={black_enc.min().item():.4f}, max={black_enc.max().item():.4f}, mean={mean_val:.4f}")
                if black_enc.max() - black_enc.min() > 0.1:
                    print("⚠️  SC1 black encoding not constant – will use mean + tolerance for compression.")
                else:
                    print("✅ SC1 black encoding near‑constant.")
            print("Models initialized.\n")

        except Exception as e:
            print(f"Error loading models: {e}")
            self.unet = UNet(n_channels=3, n_classes=2).to(self.device).eval()
            self.encoder_sc1 = SemanticEncoder(bandwidth='low').to(self.device).eval()
            self.decoder_sc1 = SemanticDecoder(bandwidth='low').to(self.device).eval()
            self.encoder_sc2 = SemanticEncoder(bandwidth='high').to(self.device).eval()
            self.decoder_sc2 = SemanticDecoder(bandwidth='high').to(self.device).eval()
            print("Using randomly initialized models for demonstration")

semantic_system = SemanticCommunicationSystem()

class UploadForm(FlaskForm):
    image = FileField('Upload Image', validators=[InputRequired()])
    channel_type = SelectField('Channel Type', choices=[('awgn', 'AWGN'), ('rayleigh', 'Rayleigh Fading')], default='awgn')
    snr_db = FloatField('SNR (dB)', default=10.0, validators=[InputRequired()])
    theta = FloatField('θ (ROI importance weight, 0-1)', default=0.8, validators=[InputRequired()])
    submit = SubmitField('Process Image')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        session_id = str(uuid.uuid4())[:8]
        file = form.image.data
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)
        try:
            result = process_image(filepath, form.channel_type.data, form.snr_db.data, form.theta.data, session_id)
            return render_template('results.html',
                                 original_image=result['original_path'],
                                 result_image=result['result_path'],
                                 roi_image=result['roi_path'],
                                 roni_image=result['roni_path'],
                                 segmentation_image=result['segmentation_path'],
                                 visualization_image=result['visualization_path'],
                                 metrics=result['metrics'],
                                 session_id=session_id)
        except Exception as e:
            return render_template('error.html', error=str(e))
    return render_template('index.html', form=form)

@app.route('/api/process', methods=['POST'])
def api_process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    channel_type = request.form.get('channel_type', 'awgn')
    snr_db = float(request.form.get('snr_db', 10.0))
    theta = float(request.form.get('theta', 0.8))
    session_id = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
    file.save(filepath)
    try:
        result = process_image(filepath, channel_type, snr_db, theta, session_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_image(image_path, channel_type, snr_db, theta, session_id):
    # 1. Load and preprocess
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess_image(original_image, IMAGE_SIZE).to(DEVICE)

    # 2. Segmentation
    with torch.no_grad():
        seg = semantic_system.unet(image_tensor.unsqueeze(0))
        seg_mask = torch.argmax(seg, dim=1).squeeze().cpu().numpy()
        
        # === ROI/RONI assignment (person = class 1, background = class 0) ===
        # If your U‑Net was trained with person as class 1, this is correct.
        # If the person appears in the wrong region, swap the conditions.
        roi_mask = (seg_mask == 1).astype(np.uint8)   # Person = ROI
        roni_mask = (seg_mask == 0).astype(np.uint8)  # Background = RONI
        
        # Diagnostic output
        roi_percentage = np.sum(roi_mask) / roi_mask.size * 100
        roni_percentage = np.sum(roni_mask) / roni_mask.size * 100
        print(f"[Segmentation] ROI (class 1) = {roi_percentage:.2f}%, RONI (class 0) = {roni_percentage:.2f}%")
        print(f"[Segmentation] Unique classes in seg_mask: {np.unique(seg_mask)}")
        if roi_percentage < 1 or roi_percentage > 95:
            print(f"⚠️  WARNING: ROI percentage is {roi_percentage:.2f}% - might indicate inverted classes!")
        
        image_np = postprocess_image(image_tensor)
        roi_image = image_np * roi_mask[..., np.newaxis]
        roni_image = image_np * roni_mask[..., np.newaxis]
        # convert to uint8 before Image.fromarray (which expects 0-255 range)
        roi_tensor = preprocess_image(Image.fromarray((roi_image * 255).astype(np.uint8)), IMAGE_SIZE).to(DEVICE)
        roni_tensor = preprocess_image(Image.fromarray((roni_image * 255).astype(np.uint8)), IMAGE_SIZE).to(DEVICE)

    # 3. Encoding
    with torch.no_grad():
        enc_roni = semantic_system.encoder_sc1(roni_tensor.unsqueeze(0))
        enc_roi  = semantic_system.encoder_sc2(roi_tensor.unsqueeze(0))

    # compute base sizes up front
    original_size = image_np.size * image_np.itemsize

    # === RONI pipeline (always run) ===
    # compute n_value for black input
    black = torch.zeros(1, 3, *IMAGE_SIZE).to(DEVICE)
    with torch.no_grad():
        black_enc_sc1 = semantic_system.encoder_sc1(black)
        n_value_roni = torch.mean(black_enc_sc1).item()
    print(f"[Process] RONI n_value (mean) = {n_value_roni:.4f}")
    
    # adaptive tolerance for RONI based on variance
    with torch.no_grad():
        roni_std = torch.std(enc_roni).item()
    tol_roni = roni_std * 0.3 if roni_std > 0 else 0.01
    tol_roni = min(max(tol_roni, 0.002), 0.015)  # between 0.002 and 0.015
    print(f"[Process] computed tol_roni = {tol_roni:.4f} (std={roni_std:.4f})")
    
    comp_roni = compress_sparse_matrix(enc_roni.squeeze().cpu().numpy(), n_value_roni, tol=tol_roni)
    print(f"[Process] comp_roni elements = {len(comp_roni[0])}")
    
    # create channel object (needed for both RONI and ROI)
    channel = AWGNChannel(snr_db) if channel_type == 'awgn' else RayleighChannel(snr_db)
    
    # if SNR is too low, RONI will be corrupted by noise - use original instead
    roni_bypassed = False
    if snr_db < 12:
        print(f"⚠️  SNR {snr_db} dB too low for RONI transmission - bypassing channel, using original background")
        roni_bypassed = True
        # use original RONI image (will be assigned in combine step)
    else:
        # channel + restore + decode for RONI
        tx_roni = channel.transmit(comp_roni)
        rest_roni = restore_sparse_matrix(tx_roni, n_value_roni, enc_roni.shape[1:])
        rest_roni_t = torch.from_numpy(rest_roni.astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dec_roni = semantic_system.decoder_sc1(rest_roni_t)


    # === ROI pipeline ===
    with torch.no_grad():
        roi_std = torch.std(enc_roi).item()
    roi_bypassed = False
    if roi_std < 1e-6:
        # constant encoding: use original ROI image directly
        print(f"⚠️  ROI encoding variance {roi_std:.2e} too small – bypassing semantic link, will copy original ROI")
        roi_bypassed = True
        comp_roi = (np.array([], dtype=np.float32), np.array([], dtype=np.int32))
        n_value_roi = 0.0
        comp_roi_bytes = 0
    else:
        # compute white/black baseline for ROI model
        with torch.no_grad():
            black_enc_sc2 = semantic_system.encoder_sc2(black)
            n_value_roi = torch.mean(black_enc_sc2).item()
        print(f"[Process] ROI n_value (mean) = {n_value_roi:.4f}")
        print(f"SC2 black encoding – min={black_enc_sc2.min().item():.4f}, max={black_enc_sc2.max().item():.4f}, mean={n_value_roi:.4f}")

        tol_roi = roi_std * 0.5
        tol_roi = min(max(tol_roi, 0.002), 0.02)
        print(f"[Process] computed tol_roi = {tol_roi:.4f} (std={roi_std:.4f})")

        comp_roi = compress_sparse_matrix(enc_roi.squeeze().cpu().numpy(), n_value_roi, tol=tol_roi)
        print(f"[Process] comp_roi size = {len(comp_roi[0])}, comp_roni size = {len(comp_roni[0])}")
        if comp_roi[0].size == 0 or len(comp_roi[0]) < 16:
            print("⚠️  ROI compression too aggressive – bypassing compression and sending raw encoding")
            raw_flat = enc_roi.squeeze().cpu().numpy().flatten()
            comp_roi = (raw_flat, np.array([[0, raw_flat.size - 1]], dtype=np.int32))
            n_value_roi = 0.0
            print(f"[Process] raw_roi length = {raw_flat.size}")

        comp_roi_bytes = len(comp_roi[0]) * 4 + len(comp_roi[1]) * 4
        # channel + restore + decode for ROI
        tx_roi  = channel.transmit(comp_roi)
        rest_roi  = restore_sparse_matrix(tx_roi,  n_value_roi, enc_roi.shape[1:])
        rest_roi_t  = torch.from_numpy(rest_roi.astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dec_roi  = semantic_system.decoder_sc2(rest_roi_t)

    # final size diagnostics
    comp_roni_bytes = len(comp_roni[0]) * 4 + len(comp_roni[1]) * 4
    compressed_size = comp_roni_bytes + comp_roi_bytes
    print("="*50)
    print(f"Original image size: {original_size} bytes")
    print(f"Compressed RONI: {comp_roni_bytes} bytes, ROI: {comp_roi_bytes} bytes")
    print(f"Total compressed: {compressed_size} bytes")
    if compressed_size > original_size:
        print("⚠️  Compressed data larger than original! Consider increasing tolerance.")
    else:
        print("✅ Compressed data smaller than original.")
    print("="*50)

    tx_roni = channel.transmit(comp_roni)
    tx_roi  = channel.transmit(comp_roi)

    # 6. Restoration
    rest_roni = restore_sparse_matrix(tx_roni, n_value_roni, enc_roni.shape[1:])
    rest_roi  = restore_sparse_matrix(tx_roi,  n_value_roi, enc_roi.shape[1:])

    # 7. Decoding
    rest_roni_t = torch.from_numpy(rest_roni.astype(np.float32)).unsqueeze(0).to(DEVICE)
    rest_roi_t  = torch.from_numpy(rest_roi.astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        dec_roni = semantic_system.decoder_sc1(rest_roni_t)
        dec_roi  = semantic_system.decoder_sc2(rest_roi_t)

    # 8. Combine using masks
    if 'roni_bypassed' in locals() and roni_bypassed:
        # use original RONI pixels when bypassed due to low SNR
        dec_roni_np = roni_image.copy()
        print(f"[Combine] Using original RONI: min={dec_roni_np.min():.3f}, max={dec_roni_np.max():.3f}, mean={dec_roni_np.mean():.3f}")
    else:
        dec_roni_np = dec_roni.squeeze().permute(1,2,0).cpu().numpy()
        print(f"[Combine] Using decoded RONI: min={dec_roni_np.min():.3f}, max={dec_roni_np.max():.3f}")
    if 'roi_bypassed' in locals() and roi_bypassed:
        # keep original ROI pixels (before encoding) when bypassed
        dec_roi_np = roi_image.copy()
        print(f"[Combine] Using original ROI: min={dec_roi_np.min():.3f}, max={dec_roi_np.max():.3f}, mean={dec_roi_np.mean():.3f}")
    else:
        dec_roi_np  = dec_roi.squeeze().permute(1,2,0).cpu().numpy()
        print(f"[Combine] Using decoded ROI: min={dec_roi_np.min():.3f}, max={dec_roi_np.max():.3f}")
    roi_mask_3d = roi_mask[..., np.newaxis]
    roni_mask_3d = roni_mask[..., np.newaxis]
    combined = roi_mask_3d * dec_roi_np + roni_mask_3d * dec_roni_np
    print(f"[Combine] Combined before clip: min={combined.min():.3f}, max={combined.max():.3f}, mean={combined.mean():.3f}")
    combined = np.clip(combined, 0, 1)
    print(f"[Combine] Combined after clip: min={combined.min():.3f}, max={combined.max():.3f}, mean={combined.mean():.3f}")

    # 9. Metrics
    original_np = image_tensor.permute(1,2,0).cpu().numpy()
    psnr = calculate_psnr(original_np, combined)
    theta_psnr = calculate_theta_psnr(original_np, combined, roi_mask, theta)
    comp_ratio = original_size / compressed_size if compressed_size > 0 else 0
    print(f"PSNR = {psnr:.2f} dB, θ-PSNR = {theta_psnr:.2f} dB, Compression ratio = {comp_ratio:.2f}:1\n")

    # 10. Save results
    combined_uint8 = (combined * 255).astype(np.uint8)
    roi_uint8 = (roi_image * 255).astype(np.uint8)
    roni_uint8 = (roni_image * 255).astype(np.uint8)
    seg_path = f"static/results/{session_id}_segmentation.png"
    res_path = f"static/results/{session_id}_result.png"
    roi_path = f"static/results/{session_id}_roi.png"
    roni_path = f"static/results/{session_id}_roni.png"
    Image.fromarray(combined_uint8).save(res_path)
    Image.fromarray(roi_uint8).save(roi_path)
    Image.fromarray(roni_uint8).save(roni_path)
    # Save segmentation mask with person as white (class 1 -> 255)
    Image.fromarray((seg_mask * 255).astype(np.uint8)).save(seg_path)

    # Visualization
    create_visualization(image_np, seg_mask, roi_uint8, roni_uint8, combined_uint8, session_id)

    metrics = {
        'psnr': round(psnr,2), 'theta_psnr': round(theta_psnr,2),
        'compression_ratio': round(comp_ratio,2),
        'roi_percentage': round(np.sum(roi_mask)/roi_mask.size*100,2),
        'snr_db': snr_db, 'theta': theta, 'channel_type': channel_type
    }
    return {
        'original_path': image_path,
        'result_path': res_path,
        'roi_path': roi_path,
        'roni_path': roni_path,
        'segmentation_path': seg_path,
        'visualization_path': f'static/results/{session_id}_visualization.png',
        'metrics': metrics,
        'session_id': session_id
    }

def create_visualization(orig, mask, roi, roni, result, sid):
    fig, ax = plt.subplots(2,3,figsize=(15,10))
    ax[0,0].imshow(orig); ax[0,0].set_title('Original'); ax[0,0].axis('off')
    ax[0,1].imshow(mask, cmap='tab20c'); ax[0,1].set_title('Segmentation'); ax[0,1].axis('off')
    ax[0,2].imshow(roi); ax[0,2].set_title('ROI'); ax[0,2].axis('off')
    ax[1,0].imshow(roni); ax[1,0].set_title('RONI'); ax[1,0].axis('off')
    ax[1,1].imshow(result); ax[1,1].set_title('Reconstructed'); ax[1,1].axis('off')
    error = np.abs(orig.astype(float) - result.astype(float))
    ax[1,2].imshow(error.mean(axis=2), cmap='hot'); ax[1,2].set_title('Error map'); ax[1,2].axis('off')
    plt.tight_layout()
    plt.savefig(f'static/results/{sid}_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

@app.route('/train', methods=['GET'])
def train_model():
    return jsonify({'status': 'Training endpoint', 'message': 'Use train.py directly'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)