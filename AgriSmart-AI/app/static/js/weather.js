/**
 * AgriSmart AI — Weather Intelligence Frontend Controller
 * ========================================================
 * Manages meteorological queries, form handling, single force-refresh requests,
 * live/demo indicators, and dynamic rendering of forecast charts and actionable advisories.
 */

window.setLocation = function (locString) {
    const locInput = document.getElementById("location");
    if (locInput) {
        locInput.value = locString;
        const form = document.getElementById("weather-query-form");
        if (form) {
            form.dispatchEvent(new Event("submit", { cancelable: true }));
        }
    }
};

window.retryLiveWeather = function () {
    const form = document.getElementById("weather-query-form");
    if (form) {
        window._forceNextRefresh = true;
        form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
};

window.forceRefreshWeather = function () {
    const form = document.getElementById("weather-query-form");
    if (form) {
        window._forceNextRefresh = true;
        form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
};

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("weather-query-form");
    if (!form) return;

    const loadingElem = document.getElementById("weather-loading");
    const resultsContainer = document.getElementById("weather-results-container");
    const errorBox = document.getElementById("weather-error-box");
    const errorMessage = document.getElementById("weather-error-message");
    const btnSubmit = document.getElementById("btn-analyze-weather");

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
            showError("Please enter a valid location or district name.");
            return;
        }

        const cropVal = document.getElementById("crop").value;
        const stageVal = document.getElementById("growth_stage").value;
        const soilMoistureVal = parseFloat(document.getElementById("soil_moisture").value);
        const soilTypeVal = document.getElementById("soil_type").value;
        const irrigMethodVal = document.getElementById("irrigation_method").value;

        const isForceRefresh = !!window._forceNextRefresh;
        window._forceNextRefresh = false; // strictly single-shot, never loop

        const payload = {
            location: locationVal,
            crop: cropVal,
            growth_stage: stageVal,
            soil_moisture: isNaN(soilMoistureVal) ? 50.0 : soilMoistureVal,
            soil_type: soilTypeVal,
            irrigation_method: irrigMethodVal,
            force_refresh: isForceRefresh,
        };

        if (loadingElem) loadingElem.style.display = "flex";
        if (btnSubmit) btnSubmit.disabled = true;

        try {
            const resp = await fetch("/analyze-weather", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Requested-With": "XMLHttpRequest",
                },
                body: JSON.stringify(payload),
            });

            const data = await resp.json();

            if (!resp.ok || data.status === "error") {
                showError(data.message || "Failed to retrieve weather intelligence. Please verify the location.");
                return;
            }

            renderWeatherResults(data);
            updateFreshnessAndBanners(data);

        } catch (err) {
            showError("Network error connecting to weather intelligence service. Please retry.");
        } finally {
            if (loadingElem) loadingElem.style.display = "none";
            if (btnSubmit) btnSubmit.disabled = false;
        }
    });

    function updateFreshnessAndBanners(data) {
        // Update top demo banner
        let demoBanner = document.getElementById("weather-demo-banner");
        const statusBadge = document.getElementById("weather-status-badge");
        const retrievedElem = document.getElementById("retrieved-timestamp");

        if (retrievedElem && data.retrieved_at) {
            retrievedElem.textContent = data.retrieved_at;
        }

        if (data.is_mock) {
            if (!demoBanner) {
                demoBanner = document.createElement("div");
                demoBanner.className = "demo-mode-banner";
                demoBanner.id = "weather-demo-banner";
                demoBanner.setAttribute("role", "alert");
                const wrapper = document.querySelector(".weather-wrapper");
                if (wrapper) wrapper.insertBefore(demoBanner, wrapper.firstChild);
            }
            demoBanner.innerHTML = `
                <div class="banner-content">
                    <span class="banner-icon">⚠️</span>
                    <div class="banner-text">
                        <strong>DEMO WEATHER — SIMULATED DATA</strong>
                        <span>
                            Live meteorological API unavailable: <em>${data.error_reason || "Simulated mode"}</em>.
                            Displaying simulated forecast and heuristic advisory for demonstration.
                        </span>
                    </div>
                    <button type="button" class="btn btn-sm btn-outline-white" id="btn-retry-weather" onclick="retryLiveWeather()">
                        🔄 Retry Live Weather
                    </button>
                </div>
            `;
            demoBanner.style.display = "block";

            if (statusBadge) {
                statusBadge.className = "badge badge-mock";
                statusBadge.innerHTML = '<span class="badge-dot"></span> Weather: DEMO';
                statusBadge.title = "Simulated weather conditions active";
            }
        } else {
            if (demoBanner) {
                demoBanner.style.display = "none";
            }
            if (statusBadge) {
                statusBadge.className = "badge badge-live";
                statusBadge.innerHTML = '<span class="badge-dot"></span> Weather: LIVE';
                statusBadge.title = "Live meteorological data queried from Open-Meteo";
            }
        }

        // Update location details in form card if element exists
        const locRequested = document.getElementById("loc-requested");
        const locResolved = document.getElementById("loc-resolved");
        const locCoords = document.getElementById("loc-coords");

        if (locRequested) locRequested.textContent = data.requested_location || data.location;
        if (locResolved) locResolved.textContent = data.resolved_location || data.location;
        if (locCoords && data.coordinates) {
            locCoords.textContent = `${data.coordinates.latitude.toFixed(4)}°N, ${data.coordinates.longitude.toFixed(4)}°E`;
        }
    }

    function formatFarmerDate(dateStr) {
        if (!dateStr) return "";
        try {
            const parts = dateStr.split("-");
            if (parts.length === 3) {
                const year = parseInt(parts[0], 10);
                const month = parseInt(parts[1], 10) - 1;
                const day = parseInt(parts[2], 10);
                const d = new Date(Date.UTC(year, month, day, 12, 0, 0));
                return d.toLocaleDateString("en-US", {
                    weekday: "short",
                    month: "short",
                    day: "numeric",
                    timeZone: "UTC"
                });
            }
        } catch (e) {}
        return dateStr;
    }

    function renderWeatherResults(data) {
        if (!resultsContainer) return;

        const cw = data.current_weather;
        const irr = data.irrigation_advice;
        const dis = data.disease_weather_risk;
        const hs = data.heat_and_spray_advisory;
        const forecast = data.forecast || [];
        const extremes = data.forecast_extremes || {};
        const summary = data.farm_advisory_summary || {};

        // 1. Build 7-day forecast cards
        const forecastCardsHtml = forecast.map((d) => {
            const probClass = Math.min(10, Math.max(0, Math.round(d.precipitation_probability_max / 10)));
            const displayDate = formatFarmerDate(d.date);
            return `
                <div class="forecast-day-card">
                    <span class="forecast-date" title="${d.date}">${displayDate}</span>
                    <span class="forecast-icon">${d.icon}</span>
                    <span class="forecast-cond">${d.condition}</span>
                    <div class="forecast-temp-range">
                        <span class="temp-high">${Math.round(d.temp_max)}°</span>
                        <span class="temp-separator">/</span>
                        <span class="temp-low">${Math.round(d.temp_min)}°</span>
                    </div>
                    <div class="forecast-rain-group">
                        <span class="rain-prob-label">🌧️ ${Math.round(d.precipitation_probability_max)}% chance</span>
                        <div class="rain-bar-track">
                            <div class="rain-bar-fill rain-prob-${probClass}"></div>
                        </div>
                        <span class="rain-sum-label"><strong>${d.precipitation_sum.toFixed(1)} mm</strong> expected</span>
                    </div>
                </div>
            `;
        }).join("");

        // 2. Build Why This Recommendation for Irrigation
        let whyIrrigationHtml = "";
        if (irr.why_recommendation) {
            const items = irr.why_recommendation.inputs_evaluated.map(inp => `<li><span class="why-check">✓</span> ${inp}</li>`).join("");
            whyIrrigationHtml = `
                <div class="why-recommendation-box">
                    <span class="why-title">WHY THIS RECOMMENDATION?</span>
                    <ul class="why-inputs-list">${items}</ul>
                    <div class="why-rule-triggered">
                        <strong>Rule triggered:</strong> "${irr.why_recommendation.rule_triggered}"
                    </div>
                </div>
            `;
        }

        // 3. Build Why This Recommendation for Disease
        let whyDiseaseHtml = "";
        if (dis.why_recommendation) {
            const items = dis.why_recommendation.inputs_evaluated.map(inp => `<li><span class="why-check">✓</span> ${inp}</li>`).join("");
            whyDiseaseHtml = `
                <div class="why-recommendation-box">
                    <span class="why-title">WHY THIS RECOMMENDATION?</span>
                    <ul class="why-inputs-list">${items}</ul>
                    <div class="why-rule-triggered">
                        <strong>Rule triggered:</strong> "${dis.why_recommendation.rule_triggered}"
                    </div>
                </div>
            `;
        }

        // 4. Build Why This Recommendation for Spray & Heat
        let whyHeatSprayHtml = "";
        if (hs.why_recommendation) {
            const items = hs.why_recommendation.inputs_evaluated.map(inp => `<li><span class="why-check">✓</span> ${inp}</li>`).join("");
            const rules = hs.why_recommendation.rules_triggered.join("; ");
            whyHeatSprayHtml = `
                <div class="why-recommendation-box">
                    <span class="why-title">WHY THIS RECOMMENDATION?</span>
                    <ul class="why-inputs-list">${items}</ul>
                    <div class="why-rule-triggered">
                        <strong>Rules evaluated:</strong> ${rules}
                    </div>
                </div>
            `;
        }

        // 5. Advance notice banner for irrigation
        let advanceWarningHtml = "";
        if (irr.future_rain_warning) {
            advanceWarningHtml = `
                <div class="future-rain-notice-box">
                    <span class="notice-icon">ℹ️</span>
                    <span class="notice-text"><strong>Advance Notice:</strong> ${irr.future_rain_warning}</span>
                </div>
            `;
        }

        // 6. Farm Advisory Summary Box
        let advisorySummaryHtml = "";
        if (summary.primary_action) {
            advisorySummaryHtml = `
                <div class="farm-advisory-summary-box">
                    <div class="summary-box-header">
                        <span class="summary-icon">📋</span>
                        <h3 class="summary-title">TODAY'S FARM ADVISORY</h3>
                    </div>
                    <div class="summary-items-list">
                        <div class="summary-line">
                            <span class="summary-tag tag-primary">Primary Action:</span>
                            <span class="summary-text">${summary.primary_action}</span>
                        </div>
                        <div class="summary-line">
                            <span class="summary-tag tag-watch">Watch:</span>
                            <span class="summary-text">${summary.watch_condition}</span>
                        </div>
                        <div class="summary-line">
                            <span class="summary-tag tag-field">Field Condition:</span>
                            <span class="summary-text">${summary.field_condition}</span>
                        </div>
                    </div>
                </div>
            `;
        }

        // 7. Forecast Extremes Strip
        let extremesStripHtml = "";
        if (extremes.highest_rain_probability && extremes.highest_expected_rainfall) {
            const probDateFmt = formatFarmerDate(extremes.highest_rain_probability.date);
            const rainDateFmt = formatFarmerDate(extremes.highest_expected_rainfall.date);
            extremesStripHtml = `
                <div class="forecast-extremes-strip">
                    <span class="extreme-badge" title="${extremes.highest_rain_probability.date}">🌧️ Rain Prob Peak: <strong>${Math.round(extremes.highest_rain_probability.probability)}%</strong> on ${probDateFmt}</span>
                    <span class="extreme-badge" title="${extremes.highest_expected_rainfall.date}">💧 Rainfall Peak: <strong>${Number(extremes.highest_expected_rainfall.rainfall_mm).toFixed(1)} mm</strong> on ${rainDateFmt}</span>
                </div>
            `;
        }

        const html = `
            <div class="weather-results-block">

                <!-- 1. Current Weather Hero Banner -->
                <div class="current-weather-card">
                    <div class="current-weather-main">
                        <div class="weather-icon-large">${cw.icon}</div>
                        <div class="current-temp-group">
                            <span class="temp-value">${cw.temperature.toFixed(1)}°C</span>
                            <span class="condition-desc">${cw.condition}</span>
                            <span class="apparent-temp">Feels like ${cw.apparent_temperature.toFixed(1)}°C</span>
                        </div>
                    </div>

                    <div class="current-weather-meta-grid">
                        <div class="meta-item">
                            <span class="meta-label">💧 Relative Humidity</span>
                            <span class="meta-val">${Math.round(cw.relative_humidity)}%</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">💨 Wind Speed</span>
                            <span class="meta-val">${cw.wind_speed.toFixed(1)} km/h</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">🌧️ Current Precipitation</span>
                            <span class="meta-val">${cw.precipitation.toFixed(1)} mm</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">📍 Resolved Location</span>
                            <span class="meta-val" id="display-location">${data.location}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">🌐 Data Provider</span>
                            <span class="meta-val"><a href="https://open-meteo.com/en/docs" target="_blank" rel="noopener" class="provider-link">${data.provider} API ↗</a></span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">🕒 Observation Time</span>
                            <span class="meta-val">${cw.timestamp}</span>
                        </div>
                    </div>
                </div>

                <!-- 2. Today's Farm Advisory Summary Box -->
                ${advisorySummaryHtml}

                <!-- 3. Primary Actionable Intelligence Cards (SIH Focus) -->
                <div class="intelligence-cards-grid">

                    <!-- Advisory 1: Irrigation Timing -->
                    <div class="advisory-card advisory-${irr.severity}">
                        <div class="advisory-card-header">
                            <span class="advisory-type-tag">💧 Irrigation Timing</span>
                            <span class="advisory-badge-pill pill-${irr.severity}">
                                ${irr.badge_text}
                            </span>
                        </div>
                        <h3 class="advisory-headline">${irr.headline}</h3>
                        <p class="advisory-reason">${irr.reason}</p>
                        ${advanceWarningHtml}
                        <div class="metrics-chip-row">
                            <span class="metric-chip">Rain Prob: ${Math.round(irr.metrics_evaluated.max_rain_probability_pct)}%</span>
                            <span class="metric-chip">Rain Expected: ${irr.metrics_evaluated.expected_precipitation_mm} mm</span>
                            <span class="metric-chip">Soil Moisture: ${irr.metrics_evaluated.manual_soil_moisture_pct}%</span>
                        </div>
                        ${whyIrrigationHtml}
                    </div>

                    <!-- Advisory 2: Disease Weather Risk -->
                    <div class="advisory-card advisory-${dis.severity}">
                        <div class="advisory-card-header">
                            <span class="advisory-type-tag">🍃 Foliar Disease Weather Risk</span>
                            <span class="advisory-badge-pill pill-${dis.severity}">
                                ${dis.badge_text}
                            </span>
                        </div>
                        <h3 class="advisory-headline">${dis.headline}</h3>
                        <p class="advisory-reason">${dis.reason}</p>
                        <div class="advisory-monitoring-box">
                            <strong>Actionable Scouting:</strong> ${dis.monitoring_guidance}
                        </div>
                        <p class="advisory-disclaimer">
                            ⚠️ ${dis.disclaimer}
                        </p>
                        ${whyDiseaseHtml}
                    </div>

                    <!-- Advisory 3: Heat Stress & Spray Window -->
                    <div class="advisory-card advisory-${hs.spray_advisory.severity}">
                        <div class="advisory-card-header">
                            <span class="advisory-type-tag">☀️ Heat & Spray Windows</span>
                            <span class="advisory-badge-pill pill-${hs.spray_advisory.severity}">
                                ${hs.spray_advisory.status}
                            </span>
                        </div>
                        <div class="heat-spray-body">
                            <!-- Spray Window Section -->
                            <div class="sub-advisory-item spray-section">
                                <span class="sub-advisory-title">💨 Spray Window Suitability (${hs.spray_advisory.wind_speed_kmh} km/h wind):</span>
                                <p>${hs.spray_advisory.description}</p>
                                <span class="sub-advisory-disclaimer">ℹ️ ${hs.spray_advisory.disclaimer}</span>
                            </div>

                            <!-- Heat Conditions Section -->
                            <div class="sub-advisory-item heat-section">
                                <span class="sub-advisory-title">🌡️ Crop Heat Conditions (${hs.heat_advisory.temperature_c}°C):</span>
                                <p>${hs.heat_advisory.description}</p>
                            </div>
                        </div>
                        ${whyHeatSprayHtml}
                    </div>

                </div>

                <!-- 4. 7-Day Multi-Day Forecast Grid -->
                <div class="forecast-section">
                    <div class="forecast-header-row">
                        <h2 class="forecast-section-title">📅 7-Day Weather Forecast & Precipitation Outlook</h2>
                        ${extremesStripHtml}
                    </div>
                    <div class="forecast-grid">
                        ${forecastCardsHtml}
                    </div>
                </div>

                <!-- 5. Transparency & Heuristic Methodology Notice -->
                <div class="methodology-card">
                    <div class="methodology-icon">ℹ️</div>
                    <div class="methodology-body">
                        <h4 class="methodology-title">Agronomic Advisory Transparency & Threshold Notice</h4>
                        <p>
                            <strong>Rule-Based Heuristics (Not Machine Learning):</strong>
                            This weather intelligence module operates strictly on deterministic agrometeorological rules.
                            Configurable heuristic thresholds (e.g. rain probability &ge; 60%, heavy precipitation &ge; 10 mm, relative humidity &ge; 75%, heat &ge; 35°C, wind &ge; 20 km/h)
                            are provided for demonstration and advisory purposes. Actual agronomic thresholds vary significantly by specific crop species, phenological growth stage, soil type, irrigation setup, and microclimate.
                        </p>
                        <p>
                            <strong>Zero-IoT Architecture:</strong>
                            Soil moisture percentages and field parameters are entered manually by the farmer. This system does not require physical hardware sensors, ESP32 microcontrollers, or telemetry devices.
                        </p>
                        <p>
                            <strong>Data Provider & Refresh:</strong>
                            Meteorological data is provided by <a href="https://open-meteo.com/en/docs" target="_blank" rel="noopener">Open-Meteo</a>. Queries are cached for 15 minutes to respect rate limits. Clicking "Refresh Weather" triggers a single on-demand live update.
                        </p>
                    </div>
                </div>

            </div>
        `;

        resultsContainer.innerHTML = html;
        const navElem = document.getElementById("navbar") || document.querySelector(".navbar");
        const navHeight = navElem ? navElem.offsetHeight : 75;
        const rect = resultsContainer.getBoundingClientRect();
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const targetY = Math.max(0, rect.top + scrollTop - navHeight - 20);
        window.scrollTo({ top: targetY, behavior: "smooth" });
    }
});
