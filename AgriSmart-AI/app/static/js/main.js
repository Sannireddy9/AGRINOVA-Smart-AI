/**
 * AgriSmart AI — Client Application Logic
 * ========================================
 * Handles drag-and-drop, camera uploads, client preview,
 * asynchronous analysis submission, and dynamic results display.
 */

document.addEventListener("DOMContentLoaded", () => {
    // ── DOM References ──────────────────────────────────────────────────
    const dropzone = document.getElementById("dropzone");
    const dropzonePrompt = document.getElementById("dropzone-prompt");
    const imageInput = document.getElementById("image-input");
    const cameraInput = document.getElementById("camera-input");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const previewFilename = document.getElementById("preview-filename");
    const previewFilesize = document.getElementById("preview-filesize");
    const btnRemoveImage = document.getElementById("btn-remove-image");
    const btnAnalyze = document.getElementById("btn-analyze");
    const uploadForm = document.getElementById("upload-form");
    const errorBox = document.getElementById("error-box");
    const errorText = document.getElementById("error-text");

    const uploadContainer = document.getElementById("upload-container");
    const loadingContainer = document.getElementById("loading-container");
    const resultContainer = document.getElementById("result-container");

    // Result DOM Elements
    const mockWarningBox = document.getElementById("mock-warning-box");
    const mockDisclaimerText = document.getElementById("mock-disclaimer-text");
    const resultImageDisplay = document.getElementById("result-image-display");
    const statusOverlayTag = document.getElementById("status-overlay-tag");
    const healthStatusBadge = document.getElementById("health-status-badge");
    const statusIcon = document.getElementById("status-icon");
    const statusLabel = document.getElementById("status-label");
    const diagnosisTitle = document.getElementById("diagnosis-title");
    const diagnosisCropName = document.getElementById("diagnosis-crop-name");
    const diagnosisConditionName = document.getElementById("diagnosis-condition-name");
    const confidenceLabel = document.getElementById("confidence-label");
    const confidenceVal = document.getElementById("confidence-val");
    const confidenceMeterFill = document.getElementById("confidence-meter-fill");
    const confidenceNote = document.getElementById("confidence-note");
    const badgeSimulated = document.getElementById("badge-simulated");
    const guidanceList = document.getElementById("guidance-list");
    const accordionTitle = document.getElementById("accordion-title");
    const topPredictionsList = document.getElementById("top-predictions-list");
    const btnResetAnalysis = document.getElementById("btn-reset-analysis");

    // Assistant Modal
    const btnOpenAssistant = document.getElementById("btn-open-assistant");
    const assistantModal = document.getElementById("assistant-modal");
    const btnCloseAssistant = document.getElementById("btn-close-assistant");
    const btnCloseAssistantFooter = document.getElementById("btn-close-assistant-footer");

    let currentSelectedFile = null;

    // ── Utility: Format Bytes ───────────────────────────────────────────
    function formatBytes(bytes) {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
    }

    // ── Utility: Error Display ──────────────────────────────────────────
    function showError(message) {
        errorText.textContent = message;
        errorBox.style.display = "flex";
    }

    function clearError() {
        errorBox.style.display = "none";
        errorText.textContent = "";
    }

    // ── Handle File Selection & Preview ─────────────────────────────────
    function handleFile(file) {
        clearError();

        if (!file) return;

        // Check if file is an image
        if (!file.type.startsWith("image/")) {
            showError("Please select a valid image file (JPG, PNG, WEBP).");
            return;
        }

        // Check size (16 MB limit)
        if (file.size > 16 * 1024 * 1024) {
            showError("Image size exceeds 16 MB. Please select a smaller photo.");
            return;
        }

        currentSelectedFile = file;

        // Display client-side preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewFilename.textContent = file.name;
            previewFilesize.textContent = formatBytes(file.size);

            dropzonePrompt.style.display = "none";
            previewContainer.style.display = "flex";
            btnAnalyze.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // ── Event Listeners: File Inputs ────────────────────────────────────
    imageInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    cameraInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    btnRemoveImage.addEventListener("click", (e) => {
        e.stopPropagation();
        resetUploadInput();
    });

    function resetUploadInput() {
        currentSelectedFile = null;
        imageInput.value = "";
        cameraInput.value = "";
        imagePreview.src = "";
        previewContainer.style.display = "none";
        dropzonePrompt.style.display = "block";
        btnAnalyze.disabled = true;
        clearError();
    }

    // ── Drag and Drop Handlers ──────────────────────────────────────────
    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files[0]) {
            handleFile(dt.files[0]);
        }
    });

    // ── Form Submission & Analysis ──────────────────────────────────────
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        if (!currentSelectedFile) {
            showError("Please select an image first before analyzing.");
            return;
        }

        clearError();

        // 1. Show loading state
        uploadContainer.style.display = "none";
        resultContainer.style.display = "none";
        loadingContainer.style.display = "block";

        const formData = new FormData();
        formData.append("image", currentSelectedFile);

        try {
            const response = await fetch("/analyze", {
                method: "POST",
                body: formData,
                headers: {
                    "X-Requested-With": "XMLHttpRequest",
                },
            });

            const data = await response.json();

            if (!response.ok || data.status === "error") {
                throw new Error(data.message || "Analysis failed. Please try again.");
            }

            // 2. Render diagnosis results
            renderResults(data);
        } catch (err) {
            console.error("Diagnosis error:", err);
            loadingContainer.style.display = "none";
            uploadContainer.style.display = "block";
            showError(err.message || "Network error. Failed to communicate with diagnosis service.");
        }
    });

    // ── Render Diagnosis Results ────────────────────────────────────────
    function renderResults(data) {
        loadingContainer.style.display = "none";

        // 1. Check Mock Mode Disclaimer & update wording
        if (data.is_mock) {
            mockWarningBox.style.display = "flex";
            mockDisclaimerText.textContent = data.mock_disclaimer;
            if (confidenceLabel) confidenceLabel.textContent = "Demo Confidence (Simulated)";
            if (badgeSimulated) badgeSimulated.style.display = "inline-flex";
            if (confidenceNote) {
                confidenceNote.textContent = "⚠️ Simulated demonstration score for UI evaluation — NOT real AI model accuracy or confidence.";
            }
            if (accordionTitle) accordionTitle.textContent = "Simulated Alternate Diagnoses (Demo Only)";
        } else {
            mockWarningBox.style.display = "none";
            if (confidenceLabel) confidenceLabel.textContent = "Diagnosis Confidence";
            if (badgeSimulated) badgeSimulated.style.display = "none";
            if (confidenceNote) {
                confidenceNote.textContent = "Calculated via multi-class softmax probability score from trained checkpoint.";
            }
            if (accordionTitle) accordionTitle.textContent = "Alternative Differential Diagnoses";
        }

        // 2. Set Image
        if (imagePreview.src) {
            resultImageDisplay.src = imagePreview.src;
        }

        // 3. Health status badge
        const isHealthy = data.is_healthy;
        if (isHealthy) {
            healthStatusBadge.className = "health-status-badge healthy";
            statusIcon.textContent = "✓";
            statusLabel.textContent = "HEALTHY PLANT";
            statusOverlayTag.textContent = "Healthy";
            statusOverlayTag.style.backgroundColor = "rgba(22, 163, 74, 0.85)";
        } else {
            healthStatusBadge.className = "health-status-badge diseased";
            statusIcon.textContent = "⚠️";
            statusLabel.textContent = "DISEASE DETECTED";
            statusOverlayTag.textContent = "Disease Detected";
            statusOverlayTag.style.backgroundColor = "rgba(220, 38, 38, 0.85)";
        }

        // 4. Titles & Meta
        diagnosisTitle.textContent = data.display_title || `${data.crop} — ${data.condition}`;
        diagnosisCropName.textContent = data.crop || "Crop";
        diagnosisConditionName.textContent = data.condition || "Condition";

        // 5. Confidence
        const confPct = data.confidence_percentage ?? Math.round((data.confidence || 0) * 100);
        confidenceVal.textContent = `${confPct}%`;
        setTimeout(() => {
            confidenceMeterFill.style.width = `${confPct}%`;
        }, 100);

        // 6. Precautionary Guidance List
        guidanceList.innerHTML = "";
        const guidanceItems = data.precautionary_guidance || [];
        guidanceItems.forEach((item, index) => {
            const li = document.createElement("li");
            li.className = "guidance-item";
            li.innerHTML = `
                <span class="guidance-bullet">${index + 1}</span>
                <span>${item}</span>
            `;
            guidanceList.appendChild(li);
        });

        // 7. Alternate Predictions (Top-K)
        topPredictionsList.innerHTML = "";
        const topPreds = data.top_predictions || [];
        if (topPreds.length > 1) {
            topPreds.forEach((pred) => {
                const row = document.createElement("div");
                row.className = "pred-row";
                const cleanName = pred.class.replace(/___/g, " — ").replace(/_/g, " ");
                const confScore = (pred.confidence * 100).toFixed(1);
                const simNote = data.is_mock ? ' <span style="font-size:0.75rem;color:var(--color-warning);font-weight:600;margin-left:0.4rem;">(simulated)</span>' : '';
                row.innerHTML = `
                    <span class="pred-class">${cleanName}</span>
                    <span class="pred-conf">${confScore}%${simNote}</span>
                `;
                topPredictionsList.appendChild(row);
            });
            document.getElementById("top-predictions-accordion").style.display = "block";
        } else {
            document.getElementById("top-predictions-accordion").style.display = "none";
        }

        // 8. Reveal Result Card & Scroll
        resultContainer.style.display = "block";
        resultContainer.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ── Reset & Analyze Another Image ───────────────────────────────────
    btnResetAnalysis.addEventListener("click", () => {
        resetUploadInput();
        resultContainer.style.display = "none";
        uploadContainer.style.display = "block";
        uploadContainer.scrollIntoView({ behavior: "smooth", block: "start" });
    });

    // ── AI Assistant Modal ──────────────────────────────────────────────
    if (btnOpenAssistant && assistantModal) {
        btnOpenAssistant.addEventListener("click", () => {
            assistantModal.style.display = "flex";
        });

        [btnCloseAssistant, btnCloseAssistantFooter].forEach((btn) => {
            if (btn) {
                btn.addEventListener("click", () => {
                    assistantModal.style.display = "none";
                });
            }
        });

        assistantModal.addEventListener("click", (e) => {
            if (e.target === assistantModal) {
                assistantModal.style.display = "none";
            }
        });
    }
});
