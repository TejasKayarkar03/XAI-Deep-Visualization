"""
Flask Web Application & REST API for Explainable AI (XAI) Deep Feature Visualization.
Serves an interactive dashboard for exploring internal CNN representations,
feature maps, filter weights, Grad-CAM, saliency maps, and activation maximization.
"""

import os
from typing import Optional
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import torch
import torch.nn.functional as F

from src.models.cnn_model import DeepVisualNet, build_model
from src.data.dataset import (
    CLASS_NAMES,
    pil_to_tensor,
    tensor_to_pil,
    pil_to_base64,
    base64_to_pil,
    generate_synthetic_demo_bank
)
from src.xai.feature_maps import (
    FeatureExtractor,
    visualize_channel_map,
    generate_feature_grid,
    get_layer_statistics
)
from src.xai.filters import (
    extract_layer_weights,
    visualize_conv_filters
)
from src.xai.gradcam import (
    GradCAM,
    overlay_heatmap_on_image
)
from src.xai.saliency import (
    compute_vanilla_saliency,
    compute_smoothgrad_saliency,
    saliency_to_image
)
from src.xai.activation_maximization import (
    maximize_feature_activation
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load model
CHECKPOINT_PATH = "models/checkpoints/deep_visual_net.pth"
model = DeepVisualNet(num_classes=len(CLASS_NAMES)).to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"[Server] Loaded trained model from {CHECKPOINT_PATH}")
else:
    print("[Server] Warning: Checkpoint not found. Model running with initial weights.")

model.eval()

# Pre-generate sample bank in static/samples
SAMPLES_DIR = os.path.join("static", "samples")
SAMPLE_BANK = generate_synthetic_demo_bank(SAMPLES_DIR)

# Layer metadata for frontend guidance
LAYER_METADATA = {
    "conv1": {
        "title": "Conv Layer 1 (Low-Level Primitives)",
        "channels": 32,
        "kernel_size": "3x3",
        "description": "Extracts low-level sensory features: oriented Gabor edges, color boundaries, high-frequency corners, and light-dark gradients."
    },
    "conv2": {
        "title": "Conv Layer 2 (Textures & Geometries)",
        "channels": 64,
        "kernel_size": "3x3",
        "description": "Combines primitive edges into repetitive textures, concentric patterns, angles, and surface motifs."
    },
    "conv3": {
        "title": "Conv Layer 3 (Object Sub-Components)",
        "channels": 128,
        "kernel_size": "3x3",
        "description": "Assembles textures into identifiable structural parts: wheel arcs, wing contours, eyes, animal ears, and silhouettes."
    },
    "conv4": {
        "title": "Conv Layer 4 (High-Level Semantic Categories)",
        "channels": 256,
        "kernel_size": "3x3",
        "description": "Encodes holistic class-discriminative semantics and spatial layouts used directly for classification."
    }
}


@app.route("/")
def index():
    """Renders the main XAI dashboard."""
    return render_template("index.html")


@app.route("/api/samples")
def get_samples():
    """Returns curated demo samples for instant zero-latency exploration."""
    return jsonify({
        "success": True,
        "samples": SAMPLE_BANK
    })


@app.route("/api/layers")
def get_layers():
    """Returns visualizable layer metadata."""
    return jsonify({
        "success": True,
        "layers": LAYER_METADATA,
        "classes": CLASS_NAMES
    })


@app.route("/api/explain", methods=["POST"])
def explain():
    """
    Main XAI endpoint:
    Performs model inference, extracts feature maps, computes layer statistics,
    generates filter weights preview, Grad-CAM attention heatmap, and saliency map.
    """
    try:
        data = request.get_json(force=True)
        image_data = data.get("image")
        sample_name = data.get("sample_name")
        target_layer = data.get("layer", "conv1")
        channel_idx = int(data.get("channel", 0))
        colormap = data.get("colormap", "viridis")
        saliency_mode = data.get("saliency_mode", "vanilla")

        # Resolve image
        pil_img = None
        if image_data:
            pil_img = base64_to_pil(image_data)
        elif sample_name:
            sample_path = os.path.join(SAMPLES_DIR, os.path.basename(sample_name))
            if os.path.exists(sample_path):
                pil_img = Image.open(sample_path).convert("RGB")

        if pil_img is None:
            # Fallback to first demo sample
            default_path = os.path.join(SAMPLES_DIR, f"{CLASS_NAMES[0]}.png")
            pil_img = Image.open(default_path).convert("RGB")

        input_tensor = pil_to_tensor(pil_img, image_size=(32, 32)).to(DEVICE)

        # 1. Forward Pass & Feature Maps
        extractor = FeatureExtractor(model, target_layers=list(LAYER_METADATA.keys()))
        logits, activations = extractor.forward(input_tensor)
        extractor.remove_hooks()

        probabilities = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        top5_indices = probabilities.argsort()[::-1][:5]
        predictions = [
            {
                "class_id": int(idx),
                "class_name": CLASS_NAMES[idx],
                "probability": float(probabilities[idx]),
                "percentage": f"{probabilities[idx] * 100.0:.2f}%"
            }
            for idx in top5_indices
        ]
        pred_class_id = int(top5_indices[0])

        # 2. Selected Layer Feature Map & Statistics
        if target_layer not in activations:
            target_layer = "conv1"
        act_tensor = activations[target_layer]

        single_channel_img = visualize_channel_map(
            act_tensor,
            channel_idx=channel_idx,
            cmap=colormap,
            upsample_size=(160, 160)
        )
        channel_base64 = pil_to_base64(single_channel_img)

        feature_grid_img = generate_feature_grid(
            act_tensor,
            max_channels=16,
            cols=4,
            cmap=colormap
        )
        grid_base64 = pil_to_base64(feature_grid_img)

        layer_stats = get_layer_statistics(act_tensor)

        # 3. Filter Kernel Weights
        try:
            weights_tensor = extract_layer_weights(model, target_layer)
            filter_img = visualize_conv_filters(weights_tensor, max_filters=24, cols=6, cmap="magma")
            filter_base64 = pil_to_base64(filter_img)
        except Exception as e:
            filter_base64 = ""

        # 4. Grad-CAM
        cam_explainer = GradCAM(model, target_layer_name="conv4")
        heatmap, evaluated_class, confidence = cam_explainer.generate_heatmap(
            input_tensor,
            target_class=pred_class_id
        )
        cam_explainer.remove_hooks()
        cam_overlay = overlay_heatmap_on_image(pil_img, heatmap, alpha=0.55, colormap_name="jet")
        gradcam_base64 = pil_to_base64(cam_overlay)

        # 5. Saliency Map
        if saliency_mode == "smoothgrad":
            saliency_map, _ = compute_smoothgrad_saliency(model, input_tensor, target_class=pred_class_id, num_samples=10)
        else:
            saliency_map, _ = compute_vanilla_saliency(model, input_tensor, target_class=pred_class_id)

        saliency_img = saliency_to_image(saliency_map, cmap="hot")
        saliency_base64 = pil_to_base64(saliency_img)

        return jsonify({
            "success": True,
            "predictions": predictions,
            "selected_layer": target_layer,
            "layer_metadata": LAYER_METADATA.get(target_layer, {}),
            "channel_idx": channel_idx,
            "channel_map": channel_base64,
            "feature_grid": grid_base64,
            "layer_stats": layer_stats,
            "filter_weights": filter_base64,
            "gradcam": gradcam_base64,
            "saliency": saliency_base64,
            "input_preview": pil_to_base64(pil_img)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/maximize", methods=["POST"])
def dream_feature():
    """
    Activation Maximization endpoint:
    Performs gradient ascent on Gaussian noise to generate the synthetic input
    that maximally activates a chosen filter or target class.
    """
    try:
        data = request.get_json(force=True)
        layer_name = data.get("layer")
        channel_idx = int(data.get("channel", 0))
        target_class = data.get("target_class")
        steps = int(data.get("steps", 40))

        if target_class is not None and str(target_class).isdigit():
            target_class = int(target_class)
            target_layer = None
        else:
            target_class = None
            target_layer = layer_name if layer_name in LAYER_METADATA else "conv2"

        synth_img = maximize_feature_activation(
            model=model,
            layer_name=target_layer,
            channel_idx=channel_idx,
            target_class=target_class,
            steps=steps,
            lr=0.12,
            image_size=(64, 64)
        )

        return jsonify({
            "success": True,
            "synthesized_image": pil_to_base64(synth_img),
            "layer": target_layer,
            "channel": channel_idx,
            "target_class": CLASS_NAMES[target_class] if target_class is not None else None
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"  Explainable AI (XAI) Deep Feature Visualization Study")
    print(f"  PBL Interactive Dashboard Running at:")
    print(f"  --> http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
