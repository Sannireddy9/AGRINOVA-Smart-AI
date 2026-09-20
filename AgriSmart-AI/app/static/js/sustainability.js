/**
 * AgriSmart AI — Farm Sustainability Score Frontend Controller
 * ============================================================
 * Manages manual farm practice inputs, soil moisture synchronization,
 * asynchronous calculation requests, dynamic weight renormalization
 * presentation, and client-side DOM rendering.
 */

window.setLoc = function (locString) {
    const locInput = document.getElementById("location");
    if (locInput) {
        locInput.value = locString;
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

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("sustainability-query-form");
    if (!form) return;

    const loadingElem = document.getElementById("sustainability-loading");
    const resultsContainer = document.getElementById("sustainability-results-container");
    const errorBox = document.getElementById("sustainability-error-box");
    const errorMessage = document.getElementById("sustainability-error-message");
    const btnSubmit = document.getElementById("btn-calculate-sustainability");

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

        const sliderInput = document.getElementById("soil_moisture");
        const numInput = document.getElementById("soil_moisture_num");
        let soilMoistureVal = sliderInput ? parseFloat(sliderInput.value) : NaN;
        if (isNaN(soilMoistureVal) && numInput) {
            soilMoistureVal = parseFloat(numInput.value);
        }

        if (isNaN(soilMoistureVal) || soilMoistureVal < 0 || soilMoistureVal > 100) {
            showError("Please enter your manual soil moisture percentage (0–100%).");
            return;
        }

        const irrigMethodVal = document.getElementById("irrigation_method").value;
        if (!irrigMethodVal) {
            showError("Please select an irrigation system.");
            return;
        }

        const locationVal = document.getElementById("location").value.trim();
        const waterAvailVal = document.getElementById("water_availability").value;
        const nutrientVal = document.getElementById("nutrient_practice").value;
        const soilCoverVal = document.getElementById("soil_cover").value;
        const cropHealthVal = document.getElementById("crop_health_status").value;
        const cropVal = document.getElementById("crop").value;
        const stageVal = document.getElementById("growth_stage").value;

        const payload = {
            location: locationVal || null,
            soil_moisture: soilMoistureVal,
            irrigation_method: irrigMethodVal,
            water_availability: waterAvailVal,
            nutrient_practice: nutrientVal,
            soil_cover: soilCoverVal,
            crop_health_status: cropHealthVal,
            crop: cropVal,
            growth_stage: stageVal,
        };

        console.log("Sustainability calculation payload:", payload);

        if (loadingElem) loadingElem.style.display = "flex";
        if (btnSubmit) btnSubmit.disabled = true;

        try {
            const resp = await fetch("/calculate-sustainability", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Requested-With": "XMLHttpRequest",
                },
                body: JSON.stringify(payload),
            });

            const data = await resp.json();

            if (!resp.ok || data.status === "error") {
                showError(data.message || "Failed to calculate sustainability score. Please verify your inputs.");
                return;
            }

            renderSustainabilityResults(data);

            if (resultsContainer) {
                resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
            }

        } catch (err) {
            showError("Network connection error reaching sustainability advisor. Please retry.");
        } finally {
            if (loadingElem) loadingElem.style.display = "none";
            if (btnSubmit) btnSubmit.disabled = false;
        }
    });

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text == null ? "" : String(text);
        return div.innerHTML;
    }

    function renderSustainabilityResults(data) {
        if (!resultsContainer) return;

        const water = data.components.water_efficiency;
        const resource = data.components.resource_use;
        const health = data.components.crop_health;
        const isRenorm = data.weights.is_renormalized;

        // 1. Water breakdown html
        const waterBreakdownHtml = (water.breakdown || []).map(item => `
            <div class="breakdown-item">
                <span class="item-icon">✓</span>
                <div class="item-text">
                    <strong>${escapeHtml(item.factor)}:</strong> ${escapeHtml(item.detail)}
                    <span class="item-points">(${escapeHtml(item.points)} pts)</span>
                </div>
            </div>
        `).join("");

        // 2. Resource breakdown html
        const resourceBreakdownHtml = (resource.breakdown || []).map(item => `
            <div class="breakdown-item">
                <span class="item-icon">✓</span>
                <div class="item-text">
                    <strong>${escapeHtml(item.factor)}:</strong> ${escapeHtml(item.detail)}
                    <span class="item-points">(${escapeHtml(item.points)} pts)</span>
                </div>
            </div>
        `).join("");

        // 3. Health breakdown html
        let healthCardHtml = "";
        if (health.is_assessed) {
            const healthBreakdownItems = (health.breakdown || []).map(item => `
                <div class="breakdown-item">
                    <span class="item-icon">✓</span>
                    <div class="item-text">
                        <strong>${escapeHtml(item.factor)}:</strong> ${escapeHtml(item.detail)}
                        <span class="item-points">(${escapeHtml(item.points)} pts)</span>
                    </div>
                </div>
            `).join("");

            healthCardHtml = `
                <div class="component-card card-health">
                    <div class="card-header-row">
                        <div class="card-title-group">
                            <span class="card-icon">🌿</span>
                            <h3 class="card-title">Crop Foliar Health</h3>
                        </div>
                        <div class="card-weight-badge">
                            Weight: ${Math.round(health.effective_weight_pct)}%
                        </div>
                    </div>
                    <div class="card-score-row">
                        <div class="score-value">${health.score}<span class="score-max">/100</span></div>
                        <div class="score-bar-track">
                            <div class="score-bar-fill fill-health" style="width: ${health.score}%;"></div>
                        </div>
                    </div>
                    <div class="breakdown-items-list">
                        ${healthBreakdownItems}
                    </div>
                </div>
            `;
        } else {
            healthCardHtml = `
                <div class="component-card card-health">
                    <div class="card-header-row">
                        <div class="card-title-group">
                            <span class="card-icon">🌿</span>
                            <h3 class="card-title">Crop Foliar Health</h3>
                        </div>
                        <div class="card-weight-badge">
                            Not Assessed (0%)
                        </div>
                    </div>
                    <div class="card-score-row">
                        <div class="score-value score-unassessed">N/A</div>
                        <div class="score-bar-track">
                            <div class="score-bar-fill fill-unassessed" style="width: 0%;"></div>
                        </div>
                    </div>
                    <div class="breakdown-items-list">
                        <div class="breakdown-item text-muted">
                            <span class="item-icon">ℹ️</span>
                            <div class="item-text">
                                <em>No foliar health observation provided. To prevent penalizing your score, its 30% weight was dynamically reallocated to Water Efficiency and Resource Use.</em>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // 4. Recommendations html
        const recsHtml = (data.recommendations || []).map(rec => `
            <div class="recommendation-item priority-${escapeHtml((rec.priority || "").toLowerCase())}">
                <div class="rec-header-line">
                    <span class="rec-badge badge-priority-${escapeHtml((rec.priority || "").toLowerCase())}">${escapeHtml(rec.priority)} Priority</span>
                    <span class="rec-category-tag">${escapeHtml(rec.category)}</span>
                    <h4 class="rec-item-title">${escapeHtml(rec.title)}</h4>
                </div>
                <p class="rec-action-text"><strong>Action:</strong> ${escapeHtml(rec.action)}</p>
                <p class="rec-rationale-text"><strong>Why:</strong> ${escapeHtml(rec.why || rec.rationale)}</p>
            </div>
        `).join("");

        // 4b. Crop context card HTML
        const cropCtx = data.crop_sustainability_context;
        let cropContextHtml = "";
        if (cropCtx) {
            cropContextHtml = `
                <div class="crop-context-card">
                    <div class="crop-context-header">
                        <div class="crop-context-title-group">
                            <span class="crop-context-icon">🌾</span>
                            <div>
                                <h3 class="crop-context-title">Crop-Specific Agronomic & Water Context</h3>
                                <span class="crop-context-subtitle">Empirical benchmarks from FAO Irrigation & Drainage Papers (56, 33) & ICAR</span>
                            </div>
                        </div>
                        <div class="crop-status-badge">
                            ${cropCtx.is_available ? `
                                <span class="badge-pill badge-primary">FAO DATA AVAILABLE</span>
                            ` : `
                                <span class="badge-pill badge-neutral">CROP NOT IN DATABASE</span>
                            `}
                        </div>
                    </div>
                    <div class="crop-context-grid">
                        <div class="crop-context-item">
                            <span class="context-label">Cultivated Crop</span>
                            <span class="context-value">
                                <strong>${escapeHtml(cropCtx.crop_name)}</strong>
                                ${cropCtx.scientific_name && cropCtx.scientific_name !== 'N/A' ? `<em>(${escapeHtml(cropCtx.scientific_name)})</em>` : ''}
                            </span>
                        </div>
                        <div class="crop-context-item">
                            <span class="context-label">Growth Stage</span>
                            <span class="context-value">
                                <strong>${escapeHtml(cropCtx.growth_stage)}</strong>
                                ${cropCtx.is_critical_stage_active ? '<span class="badge-critical-stage">⚠️ Documented Critical Stage</span>' : ''}
                            </span>
                        </div>
                        <div class="crop-context-item">
                            <span class="context-label">Source Citation</span>
                            <span class="context-value">${escapeHtml(cropCtx.source_reference)}</span>
                        </div>
                        <div class="crop-context-item">
                            <span class="context-label">Crop Water Context (Seasonal mm)</span>
                            <span class="context-value">
                                ${escapeHtml(cropCtx.seasonal_water_need_mm)}
                                ${cropCtx.is_available ? `<span class="derived-badge">AgriSmart derived category: ${escapeHtml(cropCtx.agrismart_water_need_category)}</span>` : ''}
                            </span>
                        </div>
                        <div class="crop-context-item">
                            <span class="context-label">Depletion Fraction (p)</span>
                            <span class="context-value">
                                ${cropCtx.fao_depletion_fraction_p != null && cropCtx.fao_depletion_fraction_p > 0 ? `
                                    <strong>${Number(cropCtx.fao_depletion_fraction_p).toFixed(2)}</strong>
                                    <span class="context-hint">(FAO 56 Table 22: Fraction of TAW depletable before stress begins)</span>
                                ` : '<span class="text-muted">N/A</span>'}
                            </span>
                        </div>
                        <div class="crop-context-item full-width">
                            <span class="context-label">Agronomic Assessment</span>
                            <span class="context-value">${escapeHtml(cropCtx.growth_stage_information)}</span>
                        </div>
                    </div>
                    <div class="crop-context-footer">
                        <span class="scoring-rule-tag">AgriSmart Project-Defined Scoring Rules</span>
                        <p class="scoring-rule-note">
                            ${escapeHtml(cropCtx.scoring_basis_note)}
                        </p>
                    </div>
                </div>
            `;
        }

        // 5. Build full HTML
        resultsContainer.innerHTML = `
            <div class="sustainability-results-block">

                <!-- Overall Score Hero Banner -->
                <div class="sustainability-score-hero hero-${escapeHtml(data.severity)}">
                    <div class="score-display-column">
                        <div class="score-radial-wrapper">
                            <div class="score-number-large">${data.score}</div>
                            <div class="score-denominator">/ 100</div>
                        </div>
                    </div>
                    <div class="score-details-column">
                        <div class="hero-top-row">
                            <span class="advisory-badge-pill pill-${escapeHtml(data.severity)}">
                                ${escapeHtml(data.category)}
                            </span>
                            ${isRenorm ? `
                                <span class="badge-pill badge-warning-subtle">
                                    ⚖️ Renormalized (Crop Health Excluded)
                                </span>
                            ` : `
                                <span class="badge-pill badge-neutral">
                                    ⚖️ Complete Assessment (100% Evaluated)
                                </span>
                            `}
                        </div>
                        <h2 class="score-headline">${escapeHtml(data.description)}</h2>
                        <div class="score-weights-summary">
                            <span>Evaluated Weights: </span>
                            <strong>Water: ${Math.round(water.effective_weight_pct)}%</strong> •
                            <strong>Resources: ${Math.round(resource.effective_weight_pct)}%</strong>
                            ${health.is_assessed ? `• <strong>Crop Health: ${Math.round(health.effective_weight_pct)}%</strong>` : '• <span class="text-muted">Crop Health: Not Assessed</span>'}
                        </div>
                    </div>
                </div>

                <!-- 3-Component Score Cards Grid -->
                <div class="component-cards-grid">

                    <!-- Component 1: Water Efficiency -->
                    <div class="component-card card-water">
                        <div class="card-header-row">
                            <div class="card-title-group">
                                <span class="card-icon">💧</span>
                                <h3 class="card-title">Water Efficiency</h3>
                            </div>
                            <div class="card-weight-badge">
                                Weight: ${Math.round(water.effective_weight_pct)}%
                            </div>
                        </div>
                        <div class="card-score-row">
                            <div class="score-value">${water.score}<span class="score-max">/100</span></div>
                            <div class="score-bar-track">
                                <div class="score-bar-fill fill-water" style="width: ${water.score}%;"></div>
                            </div>
                        </div>
                        <div class="breakdown-items-list">
                            ${waterBreakdownHtml}
                        </div>
                    </div>

                    <!-- Component 2: Resource Use & Conservation -->
                    <div class="component-card card-resource">
                        <div class="card-header-row">
                            <div class="card-title-group">
                                <span class="card-icon">🌱</span>
                                <h3 class="card-title">Resource Use</h3>
                            </div>
                            <div class="card-weight-badge">
                                Weight: ${Math.round(resource.effective_weight_pct)}%
                            </div>
                        </div>
                        <div class="card-score-row">
                            <div class="score-value">${resource.score}<span class="score-max">/100</span></div>
                            <div class="score-bar-track">
                                <div class="score-bar-fill fill-resource" style="width: ${resource.score}%;"></div>
                            </div>
                        </div>
                        <div class="breakdown-items-list">
                            ${resourceBreakdownHtml}
                        </div>
                    </div>

                    <!-- Component 3: Crop Health -->
                    ${healthCardHtml}

                </div>

                <!-- Crop-Specific Agronomic Context Card -->
                ${cropContextHtml}

                <!-- Section D: Why This Score? (Step-by-step arithmetic box) -->
                <div class="why-score-box">
                    <div class="why-box-header">
                        <span class="why-icon">📐</span>
                        <span class="why-title">TRANSPARENT ARITHMETIC CALCULATION (ZERO BLACK-BOX)</span>
                    </div>
                    <div class="calc-expression-display">
                        <code>${escapeHtml(data.calculation)}</code>
                    </div>
                    <p class="calc-explanation">
                        ${isRenorm ? `
                            <strong>Weight Renormalization in Effect:</strong> Because Crop Health was not assessed, the evaluated weight sum is ${Math.round(data.weights.total_evaluated_weight * 100)}%.
                            The raw weighted score (${water.score} × 0.40 + ${resource.score} × 0.30) is divided by ${data.weights.total_evaluated_weight} to renormalize the final score onto a standard 0–100 scale without fabricating unobserved data.
                        ` : `
                            <strong>Standard 3-Component Evaluation:</strong> All three sustainability dimensions were assessed. The final score is the direct weighted sum:
                            Water Efficiency (40%) + Resource Use (30%) + Crop Health (30%) = 100%.
                        `}
                    </p>
                </div>

                <!-- Section E: Actionable Sustainability Recommendations -->
                <div class="sustainability-recommendations-card">
                    <div class="recommendations-header">
                        <span class="rec-icon">🎯</span>
                        <h3 class="rec-title">Targeted Agronomic Improvements</h3>
                    </div>
                    <div class="recommendations-list">
                        ${recsHtml}
                    </div>
                </div>

                <!-- Section F: Transparency & Methodology Notice -->
                <div class="methodology-card">
                    <div class="methodology-icon">ℹ️</div>
                    <div class="methodology-body">
                        <h4 class="methodology-title">Sustainability Index Methodology & Zero-IoT Disclosure</h4>
                        <p>
                            <strong>Advisory Decision-Support Index:</strong>
                            ${escapeHtml(data.disclaimer)}
                        </p>
                        <p>
                            <strong>Strict Zero-IoT Architecture:</strong>
                            All moisture estimates and field practices are farmer-reported observations. AgriSmart AI does not require, simulate, or pretend to interface with IoT telemetry or physical soil probes.
                        </p>
                        <p>
                            <strong>Zero Unsupported Claims:</strong>
                            This system provides relative agronomic ranking and actionable guidance. It does not compute unverified carbon credits, liters of water saved, or absolute greenhouse gas emission reductions.
                        </p>
                    </div>
                </div>

            </div>
        `;
    }
});
