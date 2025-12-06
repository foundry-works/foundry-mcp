/**
 * foundry-mcp Dashboard - Main JavaScript
 */

// Configuration
let config = {
    refreshIntervalMs: 5000,
    autoRefresh: true,
};

let refreshInterval = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', async () => {
    // Load config from server
    await loadConfig();

    // Setup tab navigation
    setupTabs();

    // Setup auto-refresh toggle
    setupAutoRefresh();

    // Setup modal
    setupModal();

    // Initial data load
    await refreshData();

    // Start auto-refresh
    startAutoRefresh();
});

/**
 * Load configuration from server
 */
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        if (response.ok) {
            const data = await response.json();
            config.refreshIntervalMs = data.refresh_interval_ms || 5000;
        }
    } catch (error) {
        console.warn('Failed to load config:', error);
    }
}

/**
 * Setup tab navigation
 */
function setupTabs() {
    const tabs = document.querySelectorAll('.tab');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.dataset.tab;

            // Update active states
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(targetId).classList.add('active');

            // Refresh data for the active tab
            refreshTabData(targetId);
        });
    });
}

/**
 * Setup auto-refresh toggle
 */
function setupAutoRefresh() {
    const checkbox = document.getElementById('autoRefresh');
    checkbox.checked = config.autoRefresh;

    checkbox.addEventListener('change', (e) => {
        config.autoRefresh = e.target.checked;
        if (config.autoRefresh) {
            startAutoRefresh();
        } else {
            stopAutoRefresh();
        }
    });
}

/**
 * Start auto-refresh interval
 */
function startAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }

    if (config.autoRefresh) {
        refreshInterval = setInterval(refreshData, config.refreshIntervalMs);
    }
}

/**
 * Stop auto-refresh interval
 */
function stopAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    }
}

/**
 * Refresh all data
 */
async function refreshData() {
    const activeTab = document.querySelector('.tab.active')?.dataset.tab || 'overview';
    await refreshTabData(activeTab);
    updateLastUpdate();
}

/**
 * Refresh data for a specific tab
 */
async function refreshTabData(tabId) {
    switch (tabId) {
        case 'overview':
            await Promise.all([
                loadOverviewMetrics(),
                loadTopPatterns(),
                loadRecentErrors(),
            ]);
            break;
        case 'errors':
            await loadErrorsList();
            break;
        case 'metrics':
            await loadMetricsList();
            break;
        case 'providers':
            await loadProvidersList();
            break;
    }
}

/**
 * Update last update timestamp
 */
function updateLastUpdate() {
    const el = document.getElementById('lastUpdate');
    const now = new Date();
    el.textContent = `Last updated: ${now.toLocaleTimeString()}`;
}

// =============================================================================
// Overview Tab
// =============================================================================

/**
 * Load overview metrics from aggregated endpoint
 */
async function loadOverviewMetrics() {
    try {
        const response = await fetch('/api/overview/summary');
        if (!response.ok) {
            console.warn('Failed to load overview summary');
            return;
        }

        const data = await response.json();

        // Tool Invocations card
        document.getElementById('totalInvocations').textContent =
            formatNumber(data.invocations?.total || 0);
        document.getElementById('invocationsLastHour').textContent =
            `${data.invocations?.last_hour || 0} last hour`;

        // Active Tools card
        document.getElementById('activeTools').textContent =
            data.active_tools?.count || 0;

        // Health Status card
        const healthEl = document.getElementById('healthStatus');
        const healthStatus = data.health?.status || 'unknown';
        healthEl.textContent = healthStatus;
        healthEl.className = 'metric-value ' + healthStatus;
        document.getElementById('healthDeps').textContent =
            `${data.health?.deps_ok || 0}/${data.health?.deps_total || 0} deps ok`;

        // Avg Latency card
        const avgMs = data.latency?.avg_ms;
        document.getElementById('avgLatency').textContent =
            avgMs !== null ? `${avgMs}ms` : '-';

        // Error Rate card
        document.getElementById('errorRate').textContent =
            `${data.errors?.last_hour || 0}/hr`;
        document.getElementById('failureRate').textContent =
            `${data.errors?.failure_rate_pct || 0}% failure rate`;

        // Providers card
        document.getElementById('providersStatus').textContent =
            `${data.providers?.available || 0}/${data.providers?.total || 0}`;
        const providerNames = data.providers?.names || [];
        document.getElementById('providerNames').textContent =
            providerNames.length > 0 ? providerNames.join(', ') : 'none available';

    } catch (error) {
        console.error('Error loading overview metrics:', error);
    }
}

/**
 * Format large numbers with commas
 */
function formatNumber(num) {
    return num.toLocaleString();
}

/**
 * Load top error patterns
 */
async function loadTopPatterns() {
    const container = document.getElementById('topPatterns');

    try {
        const response = await fetch('/api/errors/patterns?min_count=2');
        if (!response.ok) {
            container.innerHTML = '<div class="empty-state">No patterns found</div>';
            return;
        }

        const data = await response.json();
        const patterns = data.patterns || [];

        if (patterns.length === 0) {
            container.innerHTML = '<div class="empty-state">No recurring patterns</div>';
            return;
        }

        container.innerHTML = patterns.slice(0, 5).map(p => `
            <div class="pattern-item">
                <span class="pattern-name">${escapeHtml(p.error_code || p.fingerprint || 'Unknown')}</span>
                <span class="pattern-count">${p.count}</span>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading patterns:', error);
        container.innerHTML = '<div class="empty-state">Failed to load patterns</div>';
    }
}

/**
 * Load recent errors
 */
async function loadRecentErrors() {
    const container = document.getElementById('recentErrors');

    try {
        const response = await fetch('/api/errors?limit=10');
        if (!response.ok) {
            container.innerHTML = '<div class="empty-state">No errors found</div>';
            return;
        }

        const data = await response.json();
        const errors = data.errors || [];

        if (errors.length === 0) {
            container.innerHTML = '<div class="empty-state">No recent errors</div>';
            return;
        }

        container.innerHTML = errors.map(e => `
            <div class="error-item" onclick="showErrorDetail('${e.id}')">
                <span class="error-timestamp">${formatTimestamp(e.timestamp)}</span>
                <span class="error-tool">${escapeHtml(e.tool_name || '-')}</span>
                <span class="error-message">${escapeHtml(e.message || e.error_type || '-')}</span>
                <span class="error-code">${escapeHtml(e.error_code || 'ERROR')}</span>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading recent errors:', error);
        container.innerHTML = '<div class="empty-state">Failed to load errors</div>';
    }
}

// =============================================================================
// Errors Tab
// =============================================================================

/**
 * Load errors list with filters
 */
async function loadErrorsList() {
    const container = document.getElementById('errorsList');
    const toolFilter = document.getElementById('errorToolFilter')?.value || '';
    const codeFilter = document.getElementById('errorCodeFilter')?.value || '';

    try {
        let url = '/api/errors?limit=50';
        if (toolFilter) url += `&tool_name=${encodeURIComponent(toolFilter)}`;
        if (codeFilter) url += `&error_code=${encodeURIComponent(codeFilter)}`;

        const response = await fetch(url);
        if (!response.ok) {
            container.innerHTML = '<div class="empty-state">No errors found</div>';
            return;
        }

        const data = await response.json();
        const errors = data.errors || [];

        if (errors.length === 0) {
            container.innerHTML = '<div class="empty-state">No errors match filters</div>';
            return;
        }

        container.innerHTML = `
            <div class="errors-list">
                ${errors.map(e => `
                    <div class="error-item" onclick="showErrorDetail('${e.id}')">
                        <span class="error-timestamp">${formatTimestamp(e.timestamp)}</span>
                        <span class="error-tool">${escapeHtml(e.tool_name || '-')}</span>
                        <span class="error-message">${escapeHtml(e.message || e.error_type || '-')}</span>
                        <span class="error-code">${escapeHtml(e.error_code || 'ERROR')}</span>
                    </div>
                `).join('')}
            </div>
        `;

        // Populate error code filter dropdown
        populateErrorCodeFilter(data.errors);
    } catch (error) {
        console.error('Error loading errors list:', error);
        container.innerHTML = '<div class="empty-state">Failed to load errors</div>';
    }
}

/**
 * Populate error code filter dropdown
 */
function populateErrorCodeFilter(errors) {
    const select = document.getElementById('errorCodeFilter');
    const currentValue = select.value;

    // Get unique error codes
    const codes = [...new Set(errors.map(e => e.error_code).filter(Boolean))];

    // Only update if codes changed
    const existingCodes = Array.from(select.options).map(o => o.value).filter(Boolean);
    if (JSON.stringify(codes.sort()) !== JSON.stringify(existingCodes.sort())) {
        select.innerHTML = '<option value="">All error codes</option>' +
            codes.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('');
        select.value = currentValue;
    }
}

// Setup error filter handlers
document.getElementById('errorToolFilter')?.addEventListener('input', debounce(loadErrorsList, 300));
document.getElementById('errorCodeFilter')?.addEventListener('change', loadErrorsList);

// =============================================================================
// Metrics Tab
// =============================================================================

/**
 * Load metrics list
 */
async function loadMetricsList() {
    const container = document.getElementById('metricsList');

    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) {
            container.innerHTML = '<div class="empty-state">No metrics found</div>';
            return;
        }

        const data = await response.json();
        const metrics = data.metrics || [];

        if (metrics.length === 0) {
            container.innerHTML = '<div class="empty-state">No metrics recorded yet</div>';
            return;
        }

        container.innerHTML = metrics.map(m => `
            <div class="metric-item">
                <div class="metric-item-name">${escapeHtml(m.metric_name || m.name || m)}</div>
                <div class="metric-item-info">${m.count ? `${m.count} records` : ''}</div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading metrics:', error);
        container.innerHTML = '<div class="empty-state">Failed to load metrics</div>';
    }
}

// =============================================================================
// Providers Tab
// =============================================================================

/**
 * Load providers list
 */
async function loadProvidersList() {
    const container = document.getElementById('providersList');

    try {
        const response = await fetch('/api/providers');
        if (!response.ok) {
            container.innerHTML = '<div class="empty-state">No providers found</div>';
            return;
        }

        const data = await response.json();
        const providers = data.providers || [];

        if (providers.length === 0) {
            container.innerHTML = '<div class="empty-state">No AI providers configured</div>';
            return;
        }

        container.innerHTML = providers.map(p => `
            <div class="provider-card">
                <div class="provider-header">
                    <span class="provider-name">${escapeHtml(p.id)}</span>
                    <span class="provider-status ${p.available ? 'available' : 'unavailable'}">
                        ${p.available ? 'Available' : 'Unavailable'}
                    </span>
                </div>
                ${p.description ? `<div class="provider-description">${escapeHtml(p.description)}</div>` : ''}
                ${p.models && p.models.length > 0 ? `
                    <div class="provider-models">
                        ${p.models.map(m => `<span class="provider-model">${escapeHtml(m)}</span>`).join('')}
                    </div>
                ` : ''}
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading providers:', error);
        container.innerHTML = '<div class="empty-state">Failed to load providers</div>';
    }
}

// =============================================================================
// Modal
// =============================================================================

/**
 * Setup modal handlers
 */
function setupModal() {
    const modal = document.getElementById('errorModal');
    const closeBtn = modal.querySelector('.modal-close');

    closeBtn.addEventListener('click', () => {
        modal.classList.remove('show');
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('show');
        }
    });

    // Close on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            modal.classList.remove('show');
        }
    });
}

/**
 * Show error detail modal
 */
async function showErrorDetail(errorId) {
    const modal = document.getElementById('errorModal');
    const detailContainer = document.getElementById('errorDetail');

    detailContainer.innerHTML = '<div class="loading">Loading...</div>';
    modal.classList.add('show');

    try {
        const response = await fetch(`/api/errors/${errorId}`);
        if (!response.ok) {
            detailContainer.innerHTML = '<div class="empty-state">Error not found</div>';
            return;
        }

        const data = await response.json();
        const error = data.error || {};

        detailContainer.innerHTML = `
            <div class="error-detail-row">
                <span class="error-detail-label">ID</span>
                <span class="error-detail-value">${escapeHtml(error.id || '-')}</span>
            </div>
            <div class="error-detail-row">
                <span class="error-detail-label">Timestamp</span>
                <span class="error-detail-value">${error.timestamp || '-'}</span>
            </div>
            <div class="error-detail-row">
                <span class="error-detail-label">Tool</span>
                <span class="error-detail-value">${escapeHtml(error.tool_name || '-')}</span>
            </div>
            <div class="error-detail-row">
                <span class="error-detail-label">Error Code</span>
                <span class="error-detail-value">${escapeHtml(error.error_code || '-')}</span>
            </div>
            <div class="error-detail-row">
                <span class="error-detail-label">Error Type</span>
                <span class="error-detail-value">${escapeHtml(error.error_type || '-')}</span>
            </div>
            <div class="error-detail-row">
                <span class="error-detail-label">Message</span>
                <span class="error-detail-value">${escapeHtml(error.message || '-')}</span>
            </div>
            <div class="error-detail-row">
                <span class="error-detail-label">Fingerprint</span>
                <span class="error-detail-value">${escapeHtml(error.fingerprint || '-')}</span>
            </div>
            ${error.stack_trace ? `
                <div class="error-stack">${escapeHtml(error.stack_trace)}</div>
            ` : ''}
            ${error.context ? `
                <div class="error-detail-row">
                    <span class="error-detail-label">Context</span>
                    <span class="error-detail-value"><pre>${escapeHtml(JSON.stringify(error.context, null, 2))}</pre></span>
                </div>
            ` : ''}
        `;
    } catch (error) {
        console.error('Error loading error detail:', error);
        detailContainer.innerHTML = '<div class="empty-state">Failed to load error details</div>';
    }
}

// =============================================================================
// Utilities
// =============================================================================

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

/**
 * Format timestamp for display
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return '-';
    try {
        const date = new Date(timestamp);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
        return timestamp;
    }
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
