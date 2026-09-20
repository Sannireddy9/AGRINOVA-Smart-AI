/**
 * AgriSmart AI — Crop Recommendation Frontend Controller
 * ======================================================
 * Handles asynchronous form submission, input validation, live/demo mode labeling,
 * and dynamic rendering of ranked crop recommendations with agronomic rationale.
 */

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("crop-recommendation-form");
    if (!form) return;

    const loadingContainer = document.getElementById("crop-loading-container");
    const resultsContainer = document.getElementById("crop-results-container");
    const errorBox = document.getElementById("form-error-box");
    const errorMessage = document.getElementById("form-error-message");
    const modeAlert = document.getElementById("result-mode-alert");
    const modeAlertBadge = document.getElementById("mode-alert-badge");
    const modeAlertText = document.getElementById("mode-alert-text");
    const cardsGrid = document.getElementById("recommendations-cards-grid");
    const explanationText = document.getElementById("explanation-text");
    const auditModelFeatures = document.getElementById("audit-model-features");
    const auditContextFeatures = document.getElementById("audit-context-features");
    const btnRecalculate = document.getElementById("btn-recalculate");

    // Initialize any server-rendered suitability meters on load
    document.querySelectorAll(".rec-meter-fill[data-suitability]").forEach((fill) => {
        const pct = fill.getAttribute("data-suitability");
        if (pct) {
            setTimeout(() => {
                fill.style.width = `${pct}%`;
            }, 100);
        }
    });

    function showError(msg) {
        if (errorMessage) errorMessage.textContent = msg;
        if (errorBox) errorBox.style.display = "flex";
    }

    function clearError() {
        if (errorBox) errorBox.style.display = "none";
    }

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        clearError();

        // 1. Validate inputs
        const ph = parseFloat(document.getElementById("ph").value);
        const temp = parseFloat(document.getElementById("temperature").value);
        const hum = parseFloat(document.getElementById("humidity").value);
        const rain = parseFloat(document.getElementById("rainfall").value);
        const n = parseFloat(document.getElementById("n").value);
        const p = parseFloat(document.getElementById("p").value);
        const k = parseFloat(document.getElementById("k").value);

        if (isNaN(ph) || ph < 3.5 || ph > 9.5) {
            showError("Soil pH must be between 3.5 and 9.5.");
            return;
        }
        if (isNaN(temp) || temp < -10 || temp > 60) {
            showError("Temperature must be between -10°C and 60°C.");
            return;
        }
        if (isNaN(hum) || hum < 10 || hum > 100) {
            showError("Relative Humidity must be between 10% and 100%.");
            return;
        }
        if (isNaN(rain) || rain < 10 || rain > 3000) {
            showError("Expected Rainfall must be between 10 mm and 3000 mm.");
            return;
        }
        if (isNaN(n) || n < 0 || n > 300) {
            showError("Nitrogen (N) must be between 0 and 300 kg/ha.");
            return;
        }
        if (isNaN(p) || p < 0 || p > 300) {
            showError("Phosphorus (P) must be between 0 and 300 kg/ha.");
            return;
        }
        if (isNaN(k) || k < 0 || k > 300) {
            showError("Potassium (K) must be between 0 and 300 kg/ha.");
            return;
        }

        const payload = {
            ph: ph,
            temperature: temp,
            humidity: hum,
            rainfall: rain,
            n: n,
            p: p,
            k: k,
            soil_type: document.getElementById("soil_type").value,
            water_availability: document.getElementById("water_availability").value,
            season: document.getElementById("season").value,
            location: document.getElementById("location").value,
            previous_crop: document.getElementById("previous_crop").value,
        };

        // 2. Show loading spinner
        if (resultsContainer) resultsContainer.style.display = "none";
        if (loadingContainer) loadingContainer.style.display = "block";

        try {
            const response = await fetch("/recommend-crops", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Requested-With": "XMLHttpRequest",
                },
                body: JSON.stringify(payload),
            });

            const data = await response.json();

            if (!response.ok || data.status === "error") {
                throw new Error(data.message || "Failed to compute recommendations.");
            }

            // 3. Render recommendations
            renderResults(data);
        } catch (err) {
            console.error("Crop recommendation error:", err);
            showError(err.message || "Network error. Failed to reach recommendation service.");
            if (loadingContainer) loadingContainer.style.display = "none";
        }
    });

    function renderResults(data) {
        if (loadingContainer) loadingContainer.style.display = "none";

        // Mode alert banner
        if (modeAlert && modeAlertBadge && modeAlertText) {
            if (data.is_mock) {
                modeAlert.className = "result-mode-alert mock-alert";
                modeAlertBadge.textContent = "SIMULATED DEMO MODE";
            } else {
                modeAlert.className = "result-mode-alert live-alert";
                modeAlertBadge.textContent = "LIVE ML INFERENCE";
            }
            modeAlertText.textContent = data.disclaimer;
        }

        // Recommendations cards
        if (cardsGrid) {
            cardsGrid.innerHTML = "";
            const recs = data.recommendations || [];
            recs.forEach((rec) => {
                const card = document.createElement("div");
                card.className = `rec-card rank-${rec.rank}`;

                const simBadge = data.is_mock ? ' <span class="badge-simulated" style="font-size:0.7rem;margin-left:0.4rem;">(SIMULATED)</span>' : '';
                card.innerHTML = `
                    <div class="rec-rank-badge">#${rec.rank}</div>
                    <div class="rec-crop-info">
                        <h3 class="rec-crop-name">${rec.crop}${simBadge}</h3>
                        <div class="rec-suitability-wrapper">
                            <div class="rec-suitability-header">
                                <span class="suitability-label">Suitability Score</span>
                                <span class="suitability-value">${rec.suitability_percentage}%</span>
                            </div>
                            <div class="rec-meter-track">
                                <div class="rec-meter-fill" style="width: 0%;"></div>
                            </div>
                            <span class="suitability-disclaimer">Model probability estimate (not a yield guarantee)</span>
                        </div>
                    </div>
                `;
                cardsGrid.appendChild(card);

                // Animate progress meter
                setTimeout(() => {
                    const fill = card.querySelector(".rec-meter-fill");
                    if (fill) fill.style.width = `${rec.suitability_percentage}%`;
                }, 100);
            });
        }

        // Agronomic explanation
        if (explanationText) {
            explanationText.textContent = data.agronomic_explanation || "";
        }

        // Audit lists
        if (auditModelFeatures && data.model_features_used) {
            auditModelFeatures.innerHTML = "";
            for (const [key, val] of Object.entries(data.model_features_used)) {
                const li = document.createElement("li");
                li.innerHTML = `<strong>${key}:</strong> ${val}`;
                auditModelFeatures.appendChild(li);
            }
        }

        if (auditContextFeatures && data.additional_context_collected) {
            auditContextFeatures.innerHTML = "";
            for (const [key, val] of Object.entries(data.additional_context_collected)) {
                if (key.startsWith("_")) continue;
                const cleanKey = key.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
                const li = document.createElement("li");
                li.innerHTML = `<strong>${cleanKey}:</strong> ${val || "—"}`;
                auditContextFeatures.appendChild(li);
            }
        }

        // Reveal and scroll smoothly
        if (resultsContainer) {
            resultsContainer.style.display = "block";
            resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }

    if (btnRecalculate) {
        btnRecalculate.addEventListener("click", () => {
            const formCard = document.querySelector(".form-card");
            if (formCard) formCard.scrollIntoView({ behavior: "smooth", block: "start" });
        });
    }
});
