/**
 * AgriSmart AI — Farmer Assistant Frontend Controller
 * ====================================================
 * Manages conversational interactions, quick-prompt dispatching,
 * multilingual parameter routing, dynamic context updating, safe markdown
 * rendering, and duplicate submission prevention.
 */

document.addEventListener("DOMContentLoaded", () => {
    const inputElem = document.getElementById("assistant-input");
    const btnSend = document.getElementById("btn-send-message");
    const langSelect = document.getElementById("assistant-language");
    const messagesStream = document.getElementById("chat-messages-stream");
    const thinkingElem = document.getElementById("thinking-indicator");
    const errorBanner = document.getElementById("assistant-error-banner");
    const errorText = document.getElementById("assistant-error-text");
    const btnRefresh = document.getElementById("btn-refresh-context");

    // Context card elements
    const ctxCrop = document.getElementById("ctx-crop");
    const ctxStage = document.getElementById("ctx-stage");
    const ctxMoisture = document.getElementById("ctx-moisture");
    const ctxLocation = document.getElementById("ctx-location");
    const ctxIrrigation = document.getElementById("ctx-irrigation");
    const ctxWaterAvailability = document.getElementById("ctx-water-availability");
    const ctxNutrientPractice = document.getElementById("ctx-nutrient-practice");
    const ctxSoilCover = document.getElementById("ctx-soil-cover");
    const ctxSustainability = document.getElementById("ctx-sustainability");
    const ctxDiseasePrediction = document.getElementById("ctx-disease-prediction");
    const ctxRecommendedCrop = document.getElementById("ctx-recommended-crop");
    const chipWeather = document.getElementById("chip-weather");
    const chipDisease = document.getElementById("chip-disease");
    const chipCrop = document.getElementById("chip-crop");
    const modeBadge = document.getElementById("assistant-mode-badge");

    let isSubmitting = false;

    function showError(msg) {
        if (errorText) errorText.textContent = msg;
        if (errorBanner) errorBanner.style.display = "flex";
    }

    function clearError() {
        if (errorBanner) errorBanner.style.display = "none";
    }

    function scrollToBottom() {
        if (messagesStream) {
            messagesStream.scrollTop = messagesStream.scrollHeight;
        }
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text == null ? "" : String(text);
        return div.innerHTML;
    }

    function renderSafeMarkdown(rawText) {
        if (!rawText) return "";
        // 1. First escape all raw HTML to eliminate XSS / script injection
        const escaped = escapeHtml(rawText);

        // 2. Format paragraphs split by double newline
        const paragraphs = escaped.split(/\n\s*\n/);
        return paragraphs.map(para => {
            const lines = para.split("\n");
            const isList = lines.every(line => {
                const tr = line.trim();
                return tr.startsWith("• ") || tr.startsWith("- ") || tr.startsWith("* ") || /^\d+\.\s/.test(tr);
            });

            if (isList) {
                const listItems = lines.map(line => {
                    const itemText = line.trim().replace(/^(?:[•\-\*]|\d+\.)\s*/, "");
                    const boldFormatted = itemText.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
                    return `<li>${boldFormatted}</li>`;
                }).join("");
                return `<ul class="msg-bullet-list">${listItems}</ul>`;
            } else {
                const formattedLines = lines.map(line => {
                    return line.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
                }).join("<br>");
                return `<p>${formattedLines}</p>`;
            }
        }).join("");
    }

    function appendUserMessage(text) {
        if (!messagesStream) return;
        const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        const html = `
            <div class="chat-message message-user">
                <div class="message-body">
                    <div class="message-header-line">
                        <strong class="sender-name">Farmer</strong>
                        <span class="message-time">${escapeHtml(now)}</span>
                    </div>
                    <div class="message-text">
                        <p>${escapeHtml(text)}</p>
                    </div>
                </div>
                <div class="message-avatar" aria-hidden="true">👨‍🌾</div>
            </div>
        `;
        messagesStream.insertAdjacentHTML("beforeend", html);
        scrollToBottom();
    }

    function appendAssistantMessage(data) {
        if (!messagesStream) return;
        const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

        const sourcesHtml = (data.sources || []).map(s => `
            <span class="source-tag">📌 ${escapeHtml(s)}</span>
        `).join("");

        const isFallback = Boolean(data.fallback_used);
        const isAI = (data.provider === "gemini" || data.provider === "openai" || data.provider_mode === "GENAI") && !isFallback;
        const modeBadgeClass = isAI ? "badge-success" : (isFallback ? "badge-warning" : "badge-neutral");
        const modeTitle = isFallback ? "AI Assistant temporarily unavailable — Local Rule Mode active" : "";
        const statusClass = (data.data_status === "LIVE") ? "badge-success" : ((data.data_status === "DEMO") ? "badge-warning" : "badge-neutral");
        const formattedAnswer = renderSafeMarkdown(data.answer || "");

        const html = `
            <div class="chat-message message-assistant">
                <div class="message-avatar" aria-hidden="true">🌿</div>
                <div class="message-body">
                    <div class="message-header-line">
                        <strong class="sender-name">AgriSmart Assistant</strong>
                        <span class="message-time">${escapeHtml(now)}</span>
                        <span class="badge-pill ${statusClass}">${escapeHtml(data.data_status || "GROUNDED")}</span>
                        <span class="badge-pill ${modeBadgeClass}" title="${escapeHtml(modeTitle)}">${escapeHtml(data.mode_notice || "Grounded Mode")}</span>
                    </div>
                    <div class="message-text">
                        ${formattedAnswer}
                    </div>
                    ${isFallback ? `
                        <div class="message-fallback-note" style="margin-top: 8px; font-size: 0.82rem; color: var(--text-muted, #666);">
                            ⚠️ <em>AI Assistant temporarily unavailable. Providing advisory using AgriSmart's deterministic rules.</em>
                        </div>
                    ` : ""}
                    ${sourcesHtml ? `
                        <div class="message-sources-strip">
                            <span class="sources-label">Sources:</span>
                            ${sourcesHtml}
                        </div>
                    ` : ""}
                </div>
            </div>
        `;
        messagesStream.insertAdjacentHTML("beforeend", html);
        scrollToBottom();
    }

    function updateContextCard(summary) {
        if (!summary) return;
        if (ctxCrop) ctxCrop.textContent = summary.crop || "Not provided";
        if (ctxStage) ctxStage.textContent = summary.growth_stage || "Not provided";
        if (ctxMoisture) ctxMoisture.textContent = summary.soil_moisture || "Not provided";
        if (ctxLocation) ctxLocation.textContent = summary.location || "Not provided";
        if (ctxIrrigation) ctxIrrigation.textContent = summary.irrigation_method || "Not provided";
        if (ctxWaterAvailability) ctxWaterAvailability.textContent = summary.water_availability || "Not provided";
        if (ctxNutrientPractice) ctxNutrientPractice.textContent = summary.nutrient_practice || "Not provided";
        if (ctxSoilCover) ctxSoilCover.textContent = summary.soil_cover || "Not provided";
        if (ctxSustainability) ctxSustainability.textContent = summary.sustainability_score || "Not provided";
        if (ctxDiseasePrediction) ctxDiseasePrediction.textContent = summary.disease_prediction || "Not assessed";
        if (ctxRecommendedCrop) ctxRecommendedCrop.textContent = summary.recommended_crop || "Not computed";

        if (chipWeather && summary.weather_status) {
            chipWeather.textContent = summary.weather_status;
            chipWeather.className = "status-chip " + (summary.weather_status === "LIVE" ? "chip-live" : "chip-mock");
        }
        if (chipDisease && summary.disease_status) {
            chipDisease.textContent = summary.disease_status;
            chipDisease.className = "status-chip " + (summary.disease_status === "LIVE" ? "chip-live" : "chip-mock");
        }
        if (chipCrop && summary.crop_model_status) {
            chipCrop.textContent = summary.crop_model_status;
            chipCrop.className = "status-chip " + (summary.crop_model_status === "LIVE" ? "chip-live" : "chip-mock");
        }
        if (modeBadge && summary.mode_notice) {
            modeBadge.textContent = summary.mode_notice;
        }
    }

    async function fetchCurrentContext() {
        try {
            const res = await fetch("/api/farmer-assistant/context", {
                headers: { "X-Requested-With": "XMLHttpRequest" }
            });
            if (res.ok) {
                const json = await res.json();
                if (json && json.context_summary) {
                    updateContextCard(json.context_summary);
                }
            }
        } catch (err) {
            console.debug("Context fetch error:", err);
        }
    }

    async function handleSendMessage(queryOverride) {
        if (isSubmitting) return;

        clearError();
        const text = (queryOverride || (inputElem ? inputElem.value : "")).trim();
        if (!text) {
            showError("Please type a question before sending.");
            if (inputElem) inputElem.focus();
            return;
        }

        const lang = langSelect ? langSelect.value : "en";

        isSubmitting = true;
        appendUserMessage(text);
        if (inputElem) {
            inputElem.value = "";
            inputElem.disabled = true;
        }

        if (thinkingElem) thinkingElem.style.display = "flex";
        if (btnSend) btnSend.disabled = true;

        let resp = null;
        try {
            resp = await fetch("/api/farmer-assistant", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Requested-With": "XMLHttpRequest",
                },
                body: JSON.stringify({
                    message: text,
                    language: lang,
                }),
            });
        } catch (netErr) {
            console.error("Farmer Assistant fetch network failure:", netErr);
            showError("Network connection error reaching Farmer Assistant. Please verify your connection.");
            isSubmitting = false;
            if (thinkingElem) thinkingElem.style.display = "none";
            if (btnSend) btnSend.disabled = false;
            if (inputElem) {
                inputElem.disabled = false;
                inputElem.focus();
            }
            return;
        }

        let data = null;
        try {
            data = await resp.json();
        } catch (parseErr) {
            data = null;
        }

        if (!resp.ok || !data || data.status === "error" || data.ok === false) {
            const safeMsg = (data && data.message) || `Server returned status ${resp.status}. Please retry.`;
            showError(safeMsg);
            isSubmitting = false;
            if (thinkingElem) thinkingElem.style.display = "none";
            if (btnSend) btnSend.disabled = false;
            if (inputElem) {
                inputElem.disabled = false;
                inputElem.focus();
            }
            return;
        }

        try {
            appendAssistantMessage(data);

            if (data.context_summary) {
                updateContextCard(data.context_summary);
            }
        } catch (renderErr) {
            console.error("Farmer Assistant UI render error:", renderErr);
            showError("Unable to display response. Please retry.");
        } finally {
            isSubmitting = false;
            if (thinkingElem) thinkingElem.style.display = "none";
            if (btnSend) btnSend.disabled = false;
            if (inputElem) {
                inputElem.disabled = false;
                inputElem.focus();
            }
        }
    }

    if (btnSend) {
        btnSend.addEventListener("click", () => handleSendMessage());
    }

    if (inputElem) {
        inputElem.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });
    }

    if (btnRefresh) {
        btnRefresh.addEventListener("click", async () => {
            btnRefresh.disabled = true;
            btnRefresh.classList.add("refreshing");
            await fetchCurrentContext();
            setTimeout(() => {
                btnRefresh.disabled = false;
                btnRefresh.classList.remove("refreshing");
            }, 300);
        });
    }

    // Refresh context when user switches back to this tab
    window.addEventListener("focus", () => {
        fetchCurrentContext();
    });

    // Initial context load on page ready
    fetchCurrentContext();

    // Attach quick-prompt handler to window
    window.sendQuickPrompt = function (promptText) {
        if (inputElem) inputElem.value = promptText;
        handleSendMessage(promptText);
    };
});
