/**
 * foundry-mcp Dashboard - Chart.js Configurations
 */

// Chart.js global defaults for dark theme
if (typeof Chart !== 'undefined') {
    Chart.defaults.color = '#a0a0a0';
    Chart.defaults.borderColor = '#2a2a4a';
    Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
}

// Chart instances
let errorRateChart = null;
let invocationsChart = null;
let overviewInvocationsChart = null;

// Chart colors
const chartColors = {
    primary: '#00d9ff',
    secondary: '#7c3aed',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    grid: '#2a2a4a',
    text: '#a0a0a0',
};

/**
 * Initialize charts after DOM is ready
 */
document.addEventListener('DOMContentLoaded', () => {
    initErrorRateChart();
    initInvocationsChart();
    initOverviewInvocationsChart();

    // Load initial chart data
    loadErrorRateData();
    loadInvocationsData();
    loadOverviewInvocationsData();
});

/**
 * Initialize error rate chart
 */
function initErrorRateChart() {
    const ctx = document.getElementById('errorRateChart');
    if (!ctx) return;

    errorRateChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Errors',
                data: [],
                borderColor: chartColors.error,
                backgroundColor: hexToRgba(chartColors.error, 0.1),
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                pointHoverRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    backgroundColor: '#0f3460',
                    titleColor: '#e8e8e8',
                    bodyColor: '#a0a0a0',
                    borderColor: '#2a2a4a',
                    borderWidth: 1,
                    padding: 12,
                },
            },
            scales: {
                x: {
                    grid: {
                        color: chartColors.grid,
                        drawBorder: false,
                    },
                    ticks: {
                        color: chartColors.text,
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 8,
                    },
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: chartColors.grid,
                        drawBorder: false,
                    },
                    ticks: {
                        color: chartColors.text,
                        stepSize: 1,
                    },
                },
            },
        },
    });
}

/**
 * Initialize invocations chart
 */
function initInvocationsChart() {
    const ctx = document.getElementById('invocationsChart');
    if (!ctx) return;

    invocationsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Invocations',
                data: [],
                backgroundColor: chartColors.primary,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    backgroundColor: '#0f3460',
                    titleColor: '#e8e8e8',
                    bodyColor: '#a0a0a0',
                    borderColor: '#2a2a4a',
                    borderWidth: 1,
                    padding: 12,
                },
            },
            scales: {
                x: {
                    grid: {
                        display: false,
                    },
                    ticks: {
                        color: chartColors.text,
                        maxRotation: 45,
                        minRotation: 45,
                    },
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: chartColors.grid,
                        drawBorder: false,
                    },
                    ticks: {
                        color: chartColors.text,
                    },
                },
            },
        },
    });
}

/**
 * Initialize overview invocations chart (bar chart for Overview tab)
 */
function initOverviewInvocationsChart() {
    const ctx = document.getElementById('overviewInvocationsChart');
    if (!ctx) return;

    overviewInvocationsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Invocations',
                data: [],
                backgroundColor: chartColors.primary,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    backgroundColor: '#0f3460',
                    titleColor: '#e8e8e8',
                    bodyColor: '#a0a0a0',
                    borderColor: '#2a2a4a',
                    borderWidth: 1,
                    padding: 12,
                },
            },
            scales: {
                x: {
                    grid: {
                        display: false,
                    },
                    ticks: {
                        color: chartColors.text,
                        maxRotation: 45,
                        minRotation: 45,
                    },
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: chartColors.grid,
                        drawBorder: false,
                    },
                    ticks: {
                        color: chartColors.text,
                    },
                },
            },
        },
    });
}

/**
 * Load overview invocations data (same data as metrics tab, for overview)
 */
async function loadOverviewInvocationsData() {
    if (!overviewInvocationsChart) return;

    try {
        const response = await fetch('/api/metrics/timeseries/tool_invocations_total?limit=500');
        if (!response.ok) {
            overviewInvocationsChart.data.labels = ['No data'];
            overviewInvocationsChart.data.datasets[0].data = [0];
            overviewInvocationsChart.update('none');
            return;
        }

        const data = await response.json();
        const datapoints = data.datapoints || [];

        if (datapoints.length === 0) {
            overviewInvocationsChart.data.labels = ['No data'];
            overviewInvocationsChart.data.datasets[0].data = [0];
            overviewInvocationsChart.update('none');
            return;
        }

        // Group by tool name
        const toolCounts = {};
        for (const dp of datapoints) {
            const toolName = dp.labels?.tool || 'unknown';
            toolCounts[toolName] = (toolCounts[toolName] || 0) + dp.value;
        }

        // Sort by count descending and take top 10 for overview
        const sorted = Object.entries(toolCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10);

        const labels = sorted.map(([name]) => formatToolName(name));
        const counts = sorted.map(([, count]) => count);

        overviewInvocationsChart.data.labels = labels;
        overviewInvocationsChart.data.datasets[0].data = counts;
        overviewInvocationsChart.update('none');
    } catch (error) {
        console.error('Error loading overview invocations data:', error);
    }
}

/**
 * Load error rate data for chart
 */
async function loadErrorRateData() {
    if (!errorRateChart) return;

    try {
        // Get errors from last hour, bucketed by 5-minute intervals
        const response = await fetch('/api/errors?limit=100');
        if (!response.ok) return;

        const data = await response.json();
        const errors = data.errors || [];

        // Bucket errors by 5-minute intervals
        const buckets = bucketByTime(errors, 5 * 60 * 1000, 12); // 12 buckets = 1 hour

        errorRateChart.data.labels = buckets.labels;
        errorRateChart.data.datasets[0].data = buckets.counts;
        errorRateChart.update('none');
    } catch (error) {
        console.error('Error loading error rate data:', error);
    }
}

/**
 * Load invocations data for chart - grouped by tool name
 */
async function loadInvocationsData() {
    if (!invocationsChart) return;

    try {
        // Get tool invocation timeseries data with labels
        const response = await fetch('/api/metrics/timeseries/tool_invocations_total?limit=500');
        if (!response.ok) {
            invocationsChart.data.labels = ['No data'];
            invocationsChart.data.datasets[0].data = [0];
            invocationsChart.update('none');
            return;
        }

        const data = await response.json();
        const datapoints = data.datapoints || [];

        if (datapoints.length === 0) {
            invocationsChart.data.labels = ['No data'];
            invocationsChart.data.datasets[0].data = [0];
            invocationsChart.update('none');
            return;
        }

        // Group by tool name
        const toolCounts = {};
        for (const dp of datapoints) {
            const toolName = dp.labels?.tool || 'unknown';
            toolCounts[toolName] = (toolCounts[toolName] || 0) + dp.value;
        }

        // Sort by count descending and take top 15
        const sorted = Object.entries(toolCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 15);

        const labels = sorted.map(([name]) => formatToolName(name));
        const counts = sorted.map(([, count]) => count);

        invocationsChart.data.labels = labels;
        invocationsChart.data.datasets[0].data = counts;
        invocationsChart.update('none');
    } catch (error) {
        console.error('Error loading invocations data:', error);
    }
}

/**
 * Format tool name for display
 */
function formatToolName(name) {
    // Shorten long tool names for display
    if (name.length > 25) {
        return name.substring(0, 22) + '...';
    }
    return name;
}

/**
 * Bucket errors by time intervals
 */
function bucketByTime(errors, intervalMs, numBuckets) {
    const now = Date.now();
    const buckets = new Array(numBuckets).fill(0);
    const labels = [];

    // Generate labels (going backwards in time)
    for (let i = numBuckets - 1; i >= 0; i--) {
        const time = new Date(now - (i * intervalMs));
        labels.push(time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    }

    // Count errors in each bucket
    for (const error of errors) {
        if (!error.timestamp) continue;

        const errorTime = new Date(error.timestamp).getTime();
        const age = now - errorTime;
        const bucketIndex = numBuckets - 1 - Math.floor(age / intervalMs);

        if (bucketIndex >= 0 && bucketIndex < numBuckets) {
            buckets[bucketIndex]++;
        }
    }

    return { labels, counts: buckets };
}

/**
 * Format metric name for display
 */
function formatMetricName(name) {
    return name
        .replace(/^(tool_|metric_)/, '')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase())
        .substring(0, 20);
}

/**
 * Convert hex color to rgba
 */
function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * Refresh charts (called from dashboard.js)
 */
function refreshCharts() {
    loadErrorRateData();
    loadInvocationsData();
    loadOverviewInvocationsData();
}

// Refresh charts when tabs are shown
document.addEventListener('DOMContentLoaded', () => {
    const overviewTab = document.querySelector('.tab[data-tab="overview"]');
    const metricsTab = document.querySelector('.tab[data-tab="metrics"]');

    if (overviewTab) {
        overviewTab.addEventListener('click', () => {
            setTimeout(() => {
                loadErrorRateData();
                loadOverviewInvocationsData();
            }, 100);
        });
    }

    if (metricsTab) {
        metricsTab.addEventListener('click', () => {
            setTimeout(loadInvocationsData, 100);
        });
    }
});
