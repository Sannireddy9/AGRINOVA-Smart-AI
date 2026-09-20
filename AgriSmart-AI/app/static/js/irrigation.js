/**
 * AgriSmart AI — Smart Irrigation Advisor Frontend Controller
 * ============================================================
 * Manages farm parameter inputs, soil moisture slider synchronization,
 * single-shot refresh requests, and dynamic rendering of rule-based
 * irrigation recommendations.
 */

window.setLocation = function (locString) {
    const locInput = document.getElementById("location");
    if (locInput) {
        locInput.value = locString;
        const form = document.getElementById("irrigation-query-form");
        if (form) {
            form.dispatchEvent(new Event("submit", { cancelable: true }));
        }
    }
};

window.syncMoistureDisplay = function (val) {
    const numInput = document.getElementById("soil_moisture_num");
    const displayBadge = document.getElementById("moisture-display");
    const parsed = Math.min(100, Math.max(0, parseInt(val, 10) || 0));
    if (numInput) numInput.value = parsed;
    if (displayBadge) displayBadge.textContent = `${parsed}%`;
};

window.syncMoistureSlider = function (val) {
    const slider = document.getElementById("soil_moisture");
    const displayBadge = document.getElementById("moisture-display");
    const parsed = Math.min(100, Math.max(0, parseInt(val, 10) || 0));
    if (slider) slider.value = parsed;
    if (displayBadge) displayBadge.textContent = `${parsed}%`;
};

window.retryLiveIrrigation = function () {
    const form = document.getElementById("irrigation-query-form");
    if (form) {
        window._forceNextRefresh = true;
        form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
};

window.forceRefreshIrrigation = function () {
    const form = document.getElementById("irrigation-query-form");
    if (form) {
        window._forceNextRefresh = true;
        form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
};

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("irrigation-query-form");
    if (!form) return;

    const loadingElem = document.getElementById("irrigation-loading");
    const resultsContainer = document.getElementById("irrigation-results-container");
    const errorBox = document.getElementById("irrigation-error-box");
    const errorMessage = document.getElementById("irrigation-error-message");
    const btnSubmit = document.getElementById("btn-calculate-irrigation");

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

        const locationVal = document.getElementById("location").value.trim();
        if (!locationVal) {
            showError("Please enter your farm location so we can retrieve the correct weather forecast.");
            return;
        }

        const soilMoistureVal = parseFloat(document.getElementById("soil_moisture").value);
        if (isNaN(soilMoistureVal) || soilMoistureVal < 0 || soilMoistureVal > 100) {
            showError("Manual soil moisture must be a number between 0% and 100%.");
            return;
        }

        const cropVal = document.getElementById("crop").value;
        const stageVal = document.getElementById("growth_stage").value;
        const soilTypeVal = document.getElementById("soil_type").value;
        const irrigMethodVal = document.getElementById("irrigation_method").value;

        const isForceRefresh = !!window._forceNextRefresh;
        window._forceNextRefresh = false; // single-shot only

        const payload = {
            location: locationVal,
            crop: cropVal,
            growth_stage: stageVal,
            soil_moisture: soilMoistureVal,
            soil_type: soilTypeVal,
            irrigation_method: irrigMethodVal,
            force_refresh: isForceRefresh,
        };

        if (loadingElem) loadingElem.style.display = "flex";
        if (btnSubmit) btnSubmit.disabled = true;

        try {
            const resp = await fetch("/calculate-irrigation", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Requested-With": "XMLHttpRequest",
                },
                body: JSON.stringify(payload),
            });

            const data = await resp.json();

            if (!resp.ok || data.status === "error") {
                showError(data.message || "Failed to calculate irrigation advisory. Please verify your inputs.");
                return;
            }

            renderIrrigationResults(data);
            updateBannersAndFreshness(data);

        } catch (err) {
            showError("Network connection error reaching irrigation advisor service. Please retry.");
        } finally {
            if (loadingElem) loadingElem.style.display = "none";
            if (btnSubmit) btnSubmit.disabled = false;
        }
    });

    function updateBannersAndFreshness(data) {
        let demoBanner = document.getElementById("irrigation-demo-banner");
        const statusBadge = document.getElementById("irrigation-status-badge");
        const retrievedElem = document.getElementById("retrieved-timestamp");
        const displayLocation = document.getElementById("display-target-location");

        if (displayLocation && data.location) {
            displayLocation.textContent = data.location;
        }
        if (retrievedElem && data.retrieved_at) {
            retrievedElem.textContent = data.retrieved_at;
        }

        if (data.is_mock) {
            if (!demoBanner) {
                demoBanner = document.createElement("div");
                demoBanner.className = "demo-mode-banner";
                demoBanner.id = "irrigation-demo-banner";
                demoBanner.setAttribute("role", "alert");
                const wrapper = document.querySelector(".irrigation-wrapper");
                if (wrapper) wrapper.insertBefore(demoBanner, wrapper.firstChild);
            }
            demoBanner.innerHTML = `
                <div class="banner-content">
                    <span class="banner-icon">⚠️</span>
                    <div class="banner-text">
                        <strong>DEMO WEATHER — IRRIGATION ADVISORY IS BASED ON SIMULATED WEATHER</strong>
                        <span>
                            Live meteorological API unavailable: <em>${data.error_reason || "Simulated weather active"}</em>.
                            Displaying simulated forecast and heuristic advisory for demonstration.
                        </span>
                    </div>
                    <button type="button" class="btn btn-sm btn-outline-white" id="btn-retry-irrigation" onclick="retryLiveIrrigation()">
                        🔄 Retry Live Weather
                    </button>
                </div>
            `;
            demoBanner.style.display = "block";

            if (statusBadge) {
                statusBadge.className = "badge badge-mock";
                statusBadge.innerHTML = '<span class="badge-dot"></span> Irrigation Advisor: DEMO';
                statusBadge.title = "Simulated weather data active";
            }
        } else {
            if (demoBanner) {
                demoBanner.style.display = "none";
            }
            if (statusBadge) {
                statusBadge.className = "badge badge-live";
                statusBadge.innerHTML = '<span class="badge-dot"></span> Irrigation Advisor: LIVE';
                statusBadge.title = "Live Open-Meteo meteorological feed active";
            }
        }
    }

    function renderIrrigationResults(data) {
        if (!resultsContainer) return;

        const rec = data.recommendation;
        const why = data.why_recommendation;
        const cropStage = data.crop_stage_context;
        const methodCtx = data.irrigation_method_context;
        const futureNotice = data.future_rain_notice;
        const ws = data.weather_summary;
        const inputs = data.inputs;

        // 1. Advance notice banner
        let futureNoticeHtml = "";
        if (futureNotice) {
            futureNoticeHtml = `
                <div class="future-rain-notice-box">
                    <span class="notice-icon">📢</span>
                    <span class="notice-text"><strong>Advance Notice (Planning Only):</strong> ${futureNotice.warning}</span>
                </div>
            `;
        }

        // 2. Why list
        const whyItems = why.inputs_evaluated.map(item => `<li><span class="why-check">✓</span> ${item}</li>`).join("");

        const html = `
            <div class="irrigation-results-block">

                <!-- Advance Notice Banner -->
                ${futureNoticeHtml}

                <!-- Section C: Main Recommendation Card -->
                <div class="irrigation-card-hero advisory-${rec.severity}">
                    <div class="hero-top-row">
                        <div class="hero-tag-group">
                            <span class="hero-type-tag">💧 Irrigation Action Advisory</span>
                            <span class="advisory-badge-pill pill-${rec.severity}">
                                ${rec.badge_text}
                            </span>
                        </div>
                        <div class="decision-basis-chip chip-basis-${rec.decision_basis.toLowerCase()}">
                            Decision Basis: <strong>${rec.decision_basis}</strong>
                        </div>
                    </div>

                    <h2 class="advisory-headline">${rec.headline}</h2>
                    <p class="advisory-reason">${rec.reason}</p>

                    <div class="action-timeline-row">
                        <span class="timeline-icon">⏱️</span>
                        <span class="timeline-label">Action Window:</span>
                        <span class="timeline-value"><strong>${rec.timeline}</strong></span>
                    </div>

                    <!-- Metrics Chip Row -->
                    <div class="metrics-chip-row">
                        <span class="metric-chip">Manual Moisture: <strong>${Math.round(inputs.soil_moisture)}%</strong></span>
                        <span class="metric-chip">Near-term Rain Prob: <strong>${Math.round(inputs.rain_probability)}%</strong></span>
                        <span class="metric-chip">Expected Rain: <strong>${inputs.expected_rainfall.toFixed(1)} mm</strong></span>
                        <span class="metric-chip">Temperature: <strong>${inputs.temperature.toFixed(1)}°C</strong></span>
                    </div>
                </div>

                <!-- Section D: Why This Recommendation? -->
                <div class="why-recommendation-box">
                    <div class="why-box-header">
                        <span class="why-icon">🔍</span>
                        <span class="why-title">WHY THIS RECOMMENDATION?</span>
                    </div>
                    <ul class="why-inputs-list">
                        ${whyItems}
                    </ul>
                    <div class="why-rule-triggered">
                        <strong>Rule Triggered:</strong> ${why.rule_triggered}
                    </div>
                    <p class="why-explanation-text">
                        ${why.explanation}
                    </p>
                </div>

                <!-- Section E: Context Detail Cards Grid -->
                <div class="context-cards-grid">
                    <!-- Crop & Stage Card -->
                    <div class="context-detail-card">
                        <h4 class="context-card-title">🌱 Crop Phenological Sensitivity</h4>
                        <p class="context-crop-name"><strong>${cropStage.crop}</strong> (${cropStage.growth_stage} Stage)</p>
                        <p class="context-body-text">${cropStage.stage_guidance}</p>
                        <p class="context-body-text sub-note">${cropStage.crop_guidance}</p>
                    </div>

                    <!-- Delivery Method Card -->
                    <div class="context-detail-card">
                        <h4 class="context-card-title">⚙️ Delivery Method Context</h4>
                        <p class="context-crop-name"><strong>${methodCtx.method} System</strong></p>
                        <p class="context-body-text">${methodCtx.guidance}</p>
                        <div class="method-reminder-note">
                            ℹ️ Note: Method descriptions provide operational context and do not assume precise numerical efficiency coefficients.
                        </div>
                    </div>
                </div>

                <!-- Section F: Weather Summary Strip -->
                <div class="weather-summary-strip">
                    <div class="weather-summary-header">
                        <span class="summary-icon">⛅</span>
                        <span class="summary-label">Current & Near-Term Meteorological Context (Open-Meteo):</span>
                    </div>
                    <div class="weather-chips-list">
                        <span class="w-chip">${ws.icon} ${ws.temperature.toFixed(1)}°C (Feels ${ws.apparent_temperature.toFixed(1)}°C)</span>
                        <span class="w-chip">💧 ${Math.round(ws.relative_humidity)}% Humidity</span>
                        <span class="w-chip">💨 ${ws.wind_speed.toFixed(1)} km/h Wind</span>
                        <span class="w-chip">🌧️ ${Math.round(ws.near_term_rain_prob_max)}% Max Rain Prob</span>
                        <span class="w-chip">☔ ${ws.near_term_rain_sum_mm.toFixed(1)} mm Expected Rain</span>
                    </div>
                </div>

                <!-- Section G: Water-Saving Logic -->
                <div class="water-saving-box">
                    <div class="water-saving-header">
                        <span class="water-icon">💧</span>
                        <h4 class="water-title">WATER-SAVING LOGIC</h4>
                    </div>
                    <p class="water-text">
                        ${data.water_saving_logic}
                        By aligning scheduled irrigation with forecast rain events and verified manual soil moisture,
                        farmers avoid wasteful runoff, root-zone saturation, and nutrient leaching.
                    </p>
                </div>

                <!-- Section H: Transparency & Methodology Notice -->
                <div class="methodology-card">
                    <div class="methodology-icon">ℹ️</div>
                    <div class="methodology-body">
                        <h4 class="methodology-title">Rule-Based Decision Support Transparency & Heuristics Notice</h4>
                        <p>
                            <strong>Deterministic Heuristics (Not Machine Learning):</strong>
                            This module evaluates agrometeorological rules in a documented priority hierarchy (A through E).
                            Configurable thresholds (Critical: 25%, Depleted: 35%, Adequate: 65%) are heuristics for decision
                            support. Actual irrigation depends on crop variety, soil profile, root depth, climate, and local practice.
                        </p>
                        <p>
                            <strong>Zero-IoT Architecture:</strong>
                            Soil moisture is purely a farmer-provided manual observation. No physical hardware sensors, ESP32 microcontrollers,
                            or telemetry devices are required or simulated.
                        </p>
                        <p>
                            <strong>Weather Provider & Refresh:</strong>
                            Meteorological conditions are retrieved from <a href="https://open-meteo.com/en/docs" target="_blank" rel="noopener">Open-Meteo</a>
                            and normalized through AgriSmart AI's weather intelligence layer.
                        </p>
                    </div>
                </div>

            </div>
        `;

        resultsContainer.innerHTML = html;

        // Smooth scroll with sticky navbar clearance
        const navElem = document.getElementById("navbar") || document.querySelector(".navbar");
        const navHeight = navElem ? navElem.offsetHeight : 75;
        const rect = resultsContainer.getBoundingClientRect();
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const targetY = Math.max(0, rect.top + scrollTop - navHeight - 20);
        window.scrollTo({ top: targetY, behavior: "smooth" });
    }
});
