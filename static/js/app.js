/**
 * Deep Feature Visualization Studio - Frontend Controller
 * Manages state, API communication, and dynamic interactive rendering.
 */

// Application State
const state = {
    selectedSample: "airplane.png",
    customImageBase64: null,
    currentLayer: "conv1",
    currentChannel: 0,
    colormap: "viridis",
    saliencyMode: "vanilla",
    layerChannels: {
        conv1: 32,
        conv2: 64,
        conv3: 128,
        conv4: 256
    },
    classNames: [],
    isComputing: false
};

// DOM Elements
const elements = {
    samplesCarousel: document.getElementById("samples-carousel"),
    uploadDropzone: document.getElementById("upload-dropzone"),
    fileInput: document.getElementById("file-input"),
    inputPreviewImg: document.getElementById("input-preview-img"),
    predictedBadge: document.getElementById("predicted-badge"),
    predictionBarsContainer: document.getElementById("prediction-bars-container"),
    layerButtons: document.querySelectorAll(".layer-btn"),
    channelSlider: document.getElementById("channel-slider"),
    channelValueDisplay: document.getElementById("channel-value-display"),
    sliderMaxDisplay: document.getElementById("slider-max-display"),
    colormapSelect: document.getElementById("colormap-select"),
    saliencyModeSelect: document.getElementById("saliency-mode-select"),
    btnRunAnalysis: document.getElementById("btn-run-analysis"),

    // Layer Info
    layerTitleText: document.getElementById("layer-title-text"),
    layerDescText: document.getElementById("layer-desc-text"),
    statRes: document.getElementById("stat-res"),
    statSparsity: document.getElementById("stat-sparsity"),

    // Tabs
    tabButtons: document.querySelectorAll(".tab-btn"),
    tabContents: document.querySelectorAll(".tab-content"),

    // Visual Displays
    imgSingleChannel: document.getElementById("img-single-channel"),
    imgFeatureGrid: document.getElementById("img-feature-grid"),
    badgeActiveLayer: document.getElementById("badge-active-layer"),
    singleChannelTitle: document.getElementById("single-channel-title"),
    imgGradcam: document.getElementById("img-gradcam"),
    gradcamClassBadge: document.getElementById("gradcam-class-badge"),
    imgSaliency: document.getElementById("img-saliency"),
    saliencyModeBadge: document.getElementById("saliency-mode-badge"),
    imgFilterWeights: document.getElementById("img-filter-weights"),
    badgeKernelLayer: document.getElementById("badge-kernel-layer"),

    // Activation Maximization
    radioDreamChannel: document.getElementById("radio-dream-channel"),
    radioDreamClass: document.getElementById("radio-dream-class"),
    dreamClassSelectGroup: document.getElementById("dream-class-select-group"),
    dreamClassSelect: document.getElementById("dream-class-select"),
    dreamStepsSlider: document.getElementById("dream-steps-slider"),
    dreamStepsDisplay: document.getElementById("dream-steps-display"),
    btnTriggerDream: document.getElementById("btn-trigger-dream"),
    imgDreamResult: document.getElementById("img-dream-result"),
    dreamSpinner: document.getElementById("dream-spinner"),
    dreamStatusBadge: document.getElementById("dream-status-badge"),
    dreamMetaSubtitle: document.getElementById("dream-meta-subtitle")
};

// Initialize Application
document.addEventListener("DOMContentLoaded", async () => {
    initTabs();
    initEventListeners();
    await loadInitialData();
    await triggerAnalysis();
});

// Load Samples and Layer Info
async function loadInitialData() {
    try {
        // Fetch layers and classes
        const layersRes = await fetch("/api/layers");
        const layersData = await layersRes.json();
        if (layersData.success) {
            state.classNames = layersData.classes;
            populateClassSelect(layersData.classes);
        }

        // Fetch samples
        const samplesRes = await fetch("/api/samples");
        const samplesData = await samplesRes.json();
        if (samplesData.success && samplesData.samples.length > 0) {
            renderSampleCarousel(samplesData.samples);
        }
    } catch (err) {
        console.error("Failed to load initial metadata:", err);
    }
}

// Render Sample Carousel
function renderSampleCarousel(samples) {
    elements.samplesCarousel.innerHTML = "";
    samples.forEach((sample, idx) => {
        const item = document.createElement("div");
        item.className = `sample-item ${idx === 0 ? "active" : ""}`;
        item.dataset.filename = sample.filename;
        item.id = `sample-item-${sample.class_name}`;

        item.innerHTML = `
            <img src="${sample.url}" alt="${sample.class_name}">
            <span class="sample-name-tooltip">${sample.class_name}</span>
        `;

        item.addEventListener("click", () => {
            document.querySelectorAll(".sample-item").forEach(s => s.classList.remove("active"));
            item.classList.add("active");
            state.selectedSample = sample.filename;
            state.customImageBase64 = null;
            elements.inputPreviewImg.src = sample.url;
            triggerAnalysis();
        });

        elements.samplesCarousel.appendChild(item);
    });
}

// Populate Target Class Dropdown for Dreaming
function populateClassSelect(classes) {
    elements.dreamClassSelect.innerHTML = "";
    classes.forEach((name, idx) => {
        const opt = document.createElement("option");
        opt.value = idx;
        opt.textContent = `Class ${idx}: ${name.charAt(0).toUpperCase() + name.slice(1)}`;
        elements.dreamClassSelect.appendChild(opt);
    });
}

// Tab Switching
function initTabs() {
    elements.tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            elements.tabButtons.forEach(b => b.classList.remove("active"));
            elements.tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const targetId = btn.dataset.tab;
            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add("active");
            }
        });
    });
}

// Event Listeners
function initEventListeners() {
    // Layer Buttons
    elements.layerButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            elements.layerButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            state.currentLayer = btn.dataset.layer;
            updateChannelSliderLimits();
            triggerAnalysis();
        });
    });

    // Channel Slider
    elements.channelSlider.addEventListener("input", (e) => {
        state.currentChannel = parseInt(e.target.value, 10);
        elements.channelValueDisplay.textContent = `Ch #${state.currentChannel}`;
    });

    elements.channelSlider.addEventListener("change", () => {
        triggerAnalysis();
    });

    // Colormap & Saliency Dropdowns
    elements.colormapSelect.addEventListener("change", (e) => {
        state.colormap = e.target.value;
        triggerAnalysis();
    });

    elements.saliencyModeSelect.addEventListener("change", (e) => {
        state.saliencyMode = e.target.value;
        triggerAnalysis();
    });

    // Run Button
    elements.btnRunAnalysis.addEventListener("click", () => {
        triggerAnalysis();
    });

    // File Upload & Drag-and-Drop
    elements.uploadDropzone.addEventListener("click", () => elements.fileInput.click());
    elements.fileInput.addEventListener("change", handleFileUpload);

    elements.uploadDropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        elements.uploadDropzone.classList.add("dragover");
    });

    elements.uploadDropzone.addEventListener("dragleave", () => {
        elements.uploadDropzone.classList.remove("dragover");
    });

    elements.uploadDropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        elements.uploadDropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            processFile(e.dataTransfer.files[0]);
        }
    });

    // Activation Maximization Controls
    elements.radioDreamChannel.addEventListener("change", () => {
        elements.dreamClassSelectGroup.style.display = "none";
    });

    elements.radioDreamClass.addEventListener("change", () => {
        elements.dreamClassSelectGroup.style.display = "block";
    });

    elements.dreamStepsSlider.addEventListener("input", (e) => {
        elements.dreamStepsDisplay.textContent = `${e.target.value} Steps`;
    });

    elements.btnTriggerDream.addEventListener("click", triggerDreamSynthesis);
}

// Update Channel Slider based on current layer channels
function updateChannelSliderLimits() {
    const maxChannels = state.layerChannels[state.currentLayer] || 32;
    elements.channelSlider.max = maxChannels - 1;
    elements.sliderMaxDisplay.textContent = maxChannels - 1;
    if (state.currentChannel >= maxChannels) {
        state.currentChannel = 0;
        elements.channelSlider.value = 0;
        elements.channelValueDisplay.textContent = `Ch #0`;
    }
}

// File Upload Handler
function handleFileUpload(e) {
    if (e.target.files && e.target.files[0]) {
        processFile(e.target.files[0]);
    }
}

function processFile(file) {
    const reader = new FileReader();
    reader.onload = (event) => {
        state.customImageBase64 = event.target.result;
        state.selectedSample = null;
        document.querySelectorAll(".sample-item").forEach(s => s.classList.remove("active"));
        elements.inputPreviewImg.src = state.customImageBase64;
        triggerAnalysis();
    };
    reader.readAsDataURL(file);
}

// Main Analysis Trigger
async function triggerAnalysis() {
    if (state.isComputing) return;
    state.isComputing = true;
    elements.btnRunAnalysis.disabled = true;
    elements.btnRunAnalysis.innerHTML = `<span class="loader-circle" style="width:16px;height:16px;border-width:2px;"></span> Computing...`;

    try {
        const payload = {
            sample_name: state.selectedSample,
            image: state.customImageBase64,
            layer: state.currentLayer,
            channel: state.currentChannel,
            colormap: state.colormap,
            saliency_mode: state.saliencyMode
        };

        const res = await fetch("/api/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.success) {
            renderAnalysisResults(data);
        } else {
            console.error("API error:", data.error);
        }
    } catch (err) {
        console.error("Analysis request failed:", err);
    } finally {
        state.isComputing = false;
        elements.btnRunAnalysis.disabled = false;
        elements.btnRunAnalysis.innerHTML = `<i data-lucide="zap"></i> Recompute Visualizations`;
        lucide.createIcons();
    }
}

// Render Results to UI
function renderAnalysisResults(data) {
    // Predictions
    if (data.predictions && data.predictions.length > 0) {
        const top = data.predictions[0];
        elements.predictedBadge.textContent = `${top.class_name.toUpperCase()} (${top.percentage})`;

        elements.predictionBarsContainer.innerHTML = "";
        data.predictions.slice(0, 3).forEach(pred => {
            const row = document.createElement("div");
            row.className = "pred-bar-row";
            row.innerHTML = `
                <div class="pred-meta">
                    <span class="pred-name">${pred.class_name}</span>
                    <span class="pred-pct">${pred.percentage}</span>
                </div>
                <div class="pred-track">
                    <div class="pred-fill" style="width: ${pred.percentage};"></div>
                </div>
            `;
            elements.predictionBarsContainer.appendChild(row);
        });

        elements.gradcamClassBadge.textContent = `Target: ${top.class_name.toUpperCase()}`;
    }

    // Layer Info Banner
    if (data.layer_metadata) {
        elements.layerTitleText.textContent = data.layer_metadata.title;
        elements.layerDescText.textContent = data.layer_metadata.description;
    }

    if (data.layer_stats) {
        elements.statRes.textContent = data.layer_stats.spatial_resolution;
        elements.statSparsity.textContent = `${(data.layer_stats.sparsity_ratio * 100).toFixed(1)}%`;
    }

    // Feature Map Views
    if (data.channel_map) {
        elements.imgSingleChannel.src = data.channel_map;
    }
    if (data.feature_grid) {
        elements.imgFeatureGrid.src = data.feature_grid;
    }

    elements.badgeActiveLayer.textContent = `${data.selected_layer} #${data.channel_idx}`;
    elements.singleChannelTitle.textContent = `Channel #${data.channel_idx} Spatial Activation`;

    // Grad-CAM & Saliency
    if (data.gradcam) {
        elements.imgGradcam.src = data.gradcam;
    }
    if (data.saliency) {
        elements.imgSaliency.src = data.saliency;
    }
    elements.saliencyModeBadge.textContent = state.saliencyMode === "smoothgrad" ? "SmoothGrad (10x Avg)" : "Vanilla Backprop";

    // Filter Weights
    if (data.filter_weights) {
        elements.imgFilterWeights.src = data.filter_weights;
        elements.badgeKernelLayer.textContent = `${data.selected_layer} Kernels`;
    }
}

// Trigger Activation Maximization (Feature Dreaming)
async function triggerDreamSynthesis() {
    elements.btnTriggerDream.disabled = true;
    elements.dreamSpinner.style.display = "flex";
    elements.dreamStatusBadge.textContent = "Optimizing...";

    const isClassDream = elements.radioDreamClass.checked;
    const targetClass = isClassDream ? elements.dreamClassSelect.value : null;
    const steps = parseInt(elements.dreamStepsSlider.value, 10);

    try {
        const payload = {
            layer: state.currentLayer,
            channel: state.currentChannel,
            target_class: targetClass,
            steps: steps
        };

        const res = await fetch("/api/maximize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.success && data.synthesized_image) {
            elements.imgDreamResult.src = data.synthesized_image;
            elements.dreamStatusBadge.textContent = "Synthesized";
            if (isClassDream && data.target_class) {
                elements.dreamMetaSubtitle.textContent = `Optimized noise hallucinating Class "${data.target_class.toUpperCase()}" archetype`;
            } else {
                elements.dreamMetaSubtitle.textContent = `Optimized stimulus maximizing ${data.layer} Channel #${data.channel}`;
            }
        }
    } catch (err) {
        console.error("Dream synthesis error:", err);
        elements.dreamStatusBadge.textContent = "Error";
    } finally {
        elements.dreamSpinner.style.display = "none";
        elements.btnTriggerDream.disabled = false;
    }
}
