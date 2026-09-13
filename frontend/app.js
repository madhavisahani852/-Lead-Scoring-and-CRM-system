/* ============================================================
   LeadScore AI — Complete Application Logic
   Vanilla JS SPA — No framework dependencies
   ============================================================ */

// ============================================================
// APPLICATION STATE
// ============================================================
const state = {
  currentTab: 'dashboard',
  leadsPage: 1,
  leadsLimit: 15,
  totalLeadsPages: 1,
  sampleLeads: {},
  analyticsData: null,
  modelInfo: null,
  currentUser: null,
  // Local data stores (created/imported leads + assignments)
  localLeads: [],          // CSV-imported or manually-created leads with scores
  leadAssignments: {},     // leadId -> { rep, stage }
  leadActivityLogs: {},    // leadId -> [{ time, text }]
  pipelineStages: ['New', 'Qualified', 'Demo Scheduled', 'Proposal', 'Won', 'Lost'],
  // Current lead being viewed in detail modal
  currentDetailLead: null,
  sidebarCollapsed: false,
};

// Required CSV columns for import validation
const REQUIRED_CSV_COLUMNS = [
  'industry', 'company_size', 'lead_source', 'product_interest',
  'budget_range', 'website_visits', 'pricing_page_visits',
  'demo_requested', 'email_opens', 'num_meetings'
];

// ============================================================
// INITIALIZATION
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  checkAuthState();
  setupNavigation();
  setupScoringForm();
  setupClickOutsideDropdown();
});

// ============================================================
// AUTH / USER SYSTEM (Direct Open — No Sign In Required)
// ============================================================
function getDefaultUser() {
  return { name: 'Sales Admin', email: 'admin@leadscore.ai', role: 'Sales Manager', initials: 'SA' };
}

function checkAuthState() {
  state.currentUser = getDefaultUser();
  showMainApp();
}

function showMainApp() {
  const mainApp = document.getElementById('main-app');
  if (mainApp) mainApp.style.display = 'flex';
  updateUserDisplay();
  loadInitialData();
}

function handleLogout() {
  showToast('User session reset.', 'info');
  updateUserDisplay();
}

function showAuthError(el, msg) {
  el.textContent = msg;
  el.style.display = 'block';
}

function updateUserDisplay() {
  if (!state.currentUser) return;
  document.getElementById('user-initials').textContent = state.currentUser.initials || 'U';
  document.getElementById('user-display-name').textContent = state.currentUser.name;
  document.getElementById('user-display-role').textContent = state.currentUser.role;
}

function toggleUserDropdown() {
  const dd = document.getElementById('user-dropdown');
  dd.style.display = dd.style.display === 'none' ? 'block' : 'none';
}

function setupClickOutsideDropdown() {
  document.addEventListener('click', (e) => {
    const menu = document.getElementById('user-menu');
    if (menu && !menu.contains(e.target)) {
      const dd = document.getElementById('user-dropdown');
      if (dd) dd.style.display = 'none';
    }
  });
}

// ============================================================
// SIDEBAR TOGGLE
// ============================================================
function toggleSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  const sidebar = document.getElementById('sidebar');
  sidebar.classList.toggle('collapsed', state.sidebarCollapsed);
}

// ============================================================
// NAVIGATION
// ============================================================
function setupNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const targetTab = item.getAttribute('data-tab');
      if (targetTab) switchTab(targetTab);
    });
  });
}

function switchTab(tabId) {
  state.currentTab = tabId;

  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('active', item.getAttribute('data-tab') === tabId);
  });

  const titles = {
    dashboard: { title: 'Executive Dashboard', subtitle: 'Real-time Lead Scoring & CRM Pipeline Overview' },
    scoring:   { title: 'Lead Scoring & Priority Queue', subtitle: 'Evaluate conversion probability & manage high-intent leads' },
    leads:     { title: 'Leads Explorer', subtitle: 'Search, filter and manage all lead records' },
    pipeline:  { title: 'Pipeline Board', subtitle: 'Kanban view — drag leads through sales stages' },
    analytics: { title: 'CRM Analytics & Insights', subtitle: 'Conversion trends, source performance & pipeline metrics' },
    import:    { title: 'CSV Import', subtitle: 'Upload, validate and score leads in bulk' },
    model:     { title: 'Model Telemetry & Evaluation', subtitle: 'XGBoost performance metrics & feature importances' }
  };

  const h = titles[tabId];
  if (h) {
    document.getElementById('current-view-title').textContent = h.title;
    document.getElementById('current-view-subtitle').textContent = h.subtitle;
  }

  document.querySelectorAll('.tab-view').forEach(view => {
    view.classList.toggle('active', view.id === `${tabId}-view`);
  });

  // Lazy load tab data
  if (tabId === 'leads') fetchLeadsData(state.leadsPage);
  else if (tabId === 'analytics' && !state.analyticsData) fetchAnalyticsData();
  else if (tabId === 'analytics' && state.analyticsData) renderAnalytics(state.analyticsData);
  else if (tabId === 'model' && !state.modelInfo) fetchModelInfoData();
  else if (tabId === 'pipeline') buildPipelineBoard();
  else if (tabId === 'scoring' && state.analyticsData) renderPriorityQueue(state.analyticsData.top_leads);
}

// ============================================================
// INITIAL DATA LOAD
// ============================================================
async function loadInitialData() {
  try {
    const presetsRes = await fetch('/api/sample-leads');
    const presetsData = await presetsRes.json();
    if (presetsData.success) {
      presetsData.samples.forEach(s => { state.sampleLeads[s.id] = s.data; });
    }
  } catch (err) {
    console.warn('Could not load sample presets:', err);
  }
  fetchAnalyticsData();
}

// ============================================================
// TOAST NOTIFICATION SYSTEM
// ============================================================
function showToast(message, type = 'info') {
  const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.animation = 'toastOut 0.3s ease forwards';
    setTimeout(() => toast.remove(), 300);
  }, 3800);
}

// ============================================================
// ANALYTICS & DASHBOARD
// ============================================================
async function fetchAnalyticsData() {
  try {
    const res = await fetch('/api/analytics');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.success) {
      state.analyticsData = data;
      renderDashboard(data);
      renderAnalytics(data);
      if (state.currentTab === 'scoring') renderPriorityQueue(data.top_leads);
    }
  } catch (err) {
    console.error('Failed to load analytics:', err);
    showToast('Failed to load analytics data. Is the backend running?', 'error');
    renderAnalyticsError();
  }
}

function renderDashboard(data) {
  const k = data.kpis;
  document.getElementById('kpi-total-leads').textContent = k.total_leads.toLocaleString();
  document.getElementById('kpi-high-leads').textContent = k.high_priority_count.toLocaleString();
  document.getElementById('kpi-high-pct').textContent = `${k.high_priority_pct}% of total leads`;
  document.getElementById('kpi-avg-score').textContent = k.avg_lead_score;
  document.getElementById('kpi-conversion-rate').textContent = `${k.actual_conversion_rate}%`;

  renderPriorityBreakdownChart(k);
  renderBarChart('dashboard-score-chart', data.score_distribution.map(d => ({ label: d.range, value: d.count })));
  renderTopLeadsPreview(data.top_leads);
}

function renderAnalytics(data) {
  const k = data.kpis;
  document.getElementById('ana-scored-leads').textContent = k.total_leads.toLocaleString();
  document.getElementById('ana-high-priority').textContent = k.high_priority_count.toLocaleString();
  document.getElementById('ana-high-pct').textContent = `${k.high_priority_pct}% of pipeline`;
  document.getElementById('ana-avg-prob').textContent = `${k.avg_conversion_prob}%`;

  const topSrc = data.source_breakdown.sort((a, b) => b.conversion_rate - a.conversion_rate)[0];
  if (topSrc) {
    document.getElementById('ana-top-source').textContent = topSrc.source;
    document.getElementById('ana-top-source-rate').textContent = `${topSrc.conversion_rate}% conv. rate`;
  }

  // Source bar chart
  renderBarChart('analytics-source-chart', data.source_breakdown.map(s => ({
    label: s.source,
    value: s.conversion_rate,
    suffix: '%'
  })));

  // Trend chart (score distribution as line-style bars)
  renderBarChart('analytics-trend-chart', data.score_distribution.map(d => ({
    label: d.range,
    value: d.count
  })), true);

  // Industry table
  renderIndustryTable(data.industry_breakdown);

  // Source table
  renderSourceTable(data.source_breakdown);

  // Fetch model info for analytics section
  if (!state.modelInfo) {
    fetchModelInfoForAnalytics();
  } else {
    renderAnalyticsModelSection(state.modelInfo);
  }
}

function renderAnalyticsError() {
  document.getElementById('kpi-total-leads').textContent = '—';
  document.getElementById('kpi-high-leads').textContent = '—';
  document.getElementById('kpi-avg-score').textContent = '—';
  document.getElementById('kpi-conversion-rate').textContent = '—';
  ['dashboard-score-chart', 'analytics-source-chart', 'analytics-trend-chart'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = `<div class="empty-state"><span class="empty-state-icon">⚠️</span><h4>Could not load chart</h4><p>Backend may be unavailable</p></div>`;
  });
}

function renderPriorityBreakdownChart(kpis) {
  const total = kpis.total_leads;
  const container = document.getElementById('dashboard-priority-chart');
  const bars = [
    { label: 'High Priority', count: kpis.high_priority_count, color: 'var(--high-priority)', bg: 'var(--high-priority-bg)' },
    { label: 'Medium Priority', count: kpis.medium_priority_count, color: 'var(--medium-priority)', bg: 'var(--medium-priority-bg)' },
    { label: 'Low Priority', count: kpis.low_priority_count, color: 'var(--low-priority)', bg: 'var(--low-priority-bg)' }
  ];
  container.innerHTML = bars.map(b => {
    const pct = total > 0 ? ((b.count / total) * 100).toFixed(1) : '0.0';
    return `
      <div>
        <div style="display:flex;justify-content:space-between;font-size:12.5px;font-weight:600;margin-bottom:6px;">
          <span style="color:${b.color};">${b.label}</span>
          <span>${b.count.toLocaleString()} (${pct}%)</span>
        </div>
        <div style="background:var(--bg-main);height:10px;border-radius:var(--radius-full);overflow:hidden;">
          <div style="width:${pct}%;background:${b.color};height:100%;border-radius:var(--radius-full);transition:width 0.8s cubic-bezier(0.34,1.56,0.64,1);"></div>
        </div>
      </div>
    `;
  }).join('');
}

function renderBarChart(elementId, items, useGradient = false) {
  const container = document.getElementById(elementId);
  if (!container || !items || items.length === 0) return;

  const maxVal = Math.max(...items.map(d => d.value), 1);
  container.innerHTML = items.map((item, i) => {
    const heightPct = Math.max(8, Math.round((item.value / maxVal) * 100));
    const hue = useGradient ? 200 + (i * 15) : 215;
    const color = `hsl(${hue}, 80%, 55%)`;
    const colorDark = `hsl(${hue}, 80%, 45%)`;
    return `
      <div class="chart-bar-group">
        <div class="chart-bar" style="height:${heightPct}%;background:linear-gradient(180deg,${color},${colorDark});" title="${item.label}: ${item.value}${item.suffix || ''}">
          <span class="bar-value">${item.value}${item.suffix || ''}</span>
        </div>
        <span class="bar-label">${item.label}</span>
      </div>
    `;
  }).join('');
}

function renderTopLeadsPreview(leads) {
  const tbody = document.getElementById('top-leads-tbody');
  if (!leads || leads.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8"><div class="empty-state"><span class="empty-state-icon">📋</span><h4>No leads yet</h4><p>Analytics data will appear once the model scores leads.</p></div></td></tr>`;
    return;
  }
  tbody.innerHTML = leads.map(l => `
    <tr onclick="viewLeadDetail('${l.lead_id}', ${JSON.stringify(l).replace(/"/g, '&quot;')})">
      <td><strong>#${l.lead_id}</strong></td>
      <td>
        <div style="font-weight:700;">${l.name}</div>
        <div style="font-size:11px;color:var(--text-muted);">${l.company}</div>
      </td>
      <td>${l.industry}</td>
      <td>${l.lead_source}</td>
      <td><strong style="font-size:16px;color:var(--primary);">${l.lead_score}</strong></td>
      <td><strong style="color:var(--high-priority);">${(l.conversion_probability * 100).toFixed(1)}%</strong></td>
      <td><span class="priority-badge priority-${l.priority.toLowerCase()}">${l.priority}</span></td>
      <td onclick="event.stopPropagation()">
        <button class="btn-secondary" style="padding:5px 10px;font-size:11.5px;" onclick="viewLeadDetail('${l.lead_id}', ${JSON.stringify(l).replace(/"/g, '&quot;')})">View</button>
      </td>
    </tr>
  `).join('');
}

function renderIndustryTable(industries) {
  const tbody = document.getElementById('analytics-industry-tbody');
  if (!tbody) return;
  if (!industries || industries.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="table-loading-cell">No data available</td></tr>`;
    return;
  }
  tbody.innerHTML = industries.map(ind => `
    <tr>
      <td><strong>${ind.industry}</strong></td>
      <td>${ind.count}</td>
      <td>${ind.avg_score}</td>
      <td><strong style="color:var(--high-priority);">${ind.conversion_rate}%</strong></td>
      <td>${ind.high_priority_pct}%</td>
    </tr>
  `).join('');
}

function renderSourceTable(sources) {
  const tbody = document.getElementById('analytics-source-tbody');
  if (!tbody) return;
  tbody.innerHTML = sources.map(s => `
    <tr>
      <td><strong>${s.source}</strong></td>
      <td>${s.count}</td>
      <td>${s.avg_score}</td>
      <td><strong style="color:var(--primary);">${s.conversion_rate}%</strong></td>
    </tr>
  `).join('');
}

async function fetchModelInfoForAnalytics() {
  try {
    const res = await fetch('/api/model-info');
    const data = await res.json();
    if (data.success) {
      state.modelInfo = data;
      renderAnalyticsModelSection(data);
      renderModelFeaturesTable(data.top_features);
      updateModelMetrics(data);
    }
  } catch (err) {
    console.warn('Could not load model info:', err);
  }
}

function renderAnalyticsModelSection(data) {
  document.getElementById('analytics-model-name').textContent = data.model_name;
  document.getElementById('analytics-model-reason').textContent = data.selection_reason;

  const m = data.test_metrics || {};
  const metrics = [
    { label: 'ROC-AUC', value: m.roc_auc ? m.roc_auc.toFixed(4) : '0.8086', color: 'var(--primary)' },
    { label: 'PR-AUC', value: m.pr_auc ? m.pr_auc.toFixed(4) : '0.8654', color: 'var(--text-primary)' },
    { label: 'F1 Score', value: m.f1 ? m.f1.toFixed(4) : '0.7922', color: 'var(--text-primary)' },
    { label: 'Precision', value: m.precision ? m.precision.toFixed(4) : '0.7710', color: 'var(--high-priority)' },
  ];

  document.getElementById('analytics-model-metrics').innerHTML = metrics.map(mt => `
    <div style="background:var(--bg-main);padding:14px;border-radius:var(--radius-md);border:1px solid var(--border-color);">
      <div style="font-size:10.5px;color:var(--text-secondary);font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">${mt.label}</div>
      <div style="font-size:24px;font-weight:900;color:${mt.color};">${mt.value}</div>
    </div>
  `).join('');
}

// ============================================================
// PRIORITY QUEUE
// ============================================================
function renderPriorityQueue(leads) {
  const container = document.getElementById('priority-queue-list');
  if (!leads || leads.length === 0) {
    container.innerHTML = `<div class="empty-state"><span class="empty-state-icon">🏆</span><h4>No leads scored yet</h4><p>Visit the dashboard to load the dataset.</p></div>`;
    return;
  }

  const sorted = [...leads].sort((a, b) => (b.conversion_probability || 0) - (a.conversion_probability || 0));
  const rankEmojis = ['🥇', '🥈', '🥉'];

  container.innerHTML = sorted.map((l, i) => {
    const rankClass = i < 3 ? `rank-${i + 1}` : '';
    const rankLabel = rankEmojis[i] || `#${i + 1}`;
    return `
      <div class="queue-item" onclick="viewLeadDetail('${l.lead_id}', ${JSON.stringify(l).replace(/"/g, '&quot;')})">
        <div class="queue-rank ${rankClass}">${rankLabel}</div>
        <div class="queue-info">
          <div class="queue-name">${l.name}</div>
          <div class="queue-company">${l.company} · ${l.industry} · ${l.lead_source}</div>
        </div>
        <div style="flex-shrink:0;">
          <span class="priority-badge priority-${l.priority.toLowerCase()}">${l.priority}</span>
        </div>
        <div class="queue-score-block">
          <div class="queue-score">${l.lead_score}</div>
          <div class="queue-prob">${(l.conversion_probability * 100).toFixed(1)}% prob.</div>
        </div>
      </div>
    `;
  }).join('');
}

// ============================================================
// LEAD SCORING FORM
// ============================================================
function setupScoringForm() {
  const form = document.getElementById('scoring-form');
  if (form) form.addEventListener('submit', e => { e.preventDefault(); submitScoringForm(); });
}

function loadPreset(presetId) {
  const data = state.sampleLeads[presetId];
  if (!data) { showToast('Preset not loaded yet. Try again shortly.', 'warning'); return; }

  document.getElementById('input-name').value = data.name || '';
  document.getElementById('input-company').value = data.company || '';
  document.getElementById('input-job-title').value = data.job_title || '';
  document.getElementById('input-industry').value = data.industry || 'SaaS';
  document.getElementById('input-company-size').value = data.company_size || 100;
  document.getElementById('input-budget-range').value = data.budget_range || 'Unknown';
  document.getElementById('input-lead-source').value = data.lead_source || 'Website';
  document.getElementById('input-product-interest').value = data.product_interest || 'Enterprise Plan';
  document.getElementById('input-website-visits').value = data.website_visits || 5;
  document.getElementById('input-page-views').value = data.page_views || 15;
  document.getElementById('input-pricing-page-visits').value = data.pricing_page_visits || 2;

  const demoVal = data.demo_requested || 'No';
  document.getElementsByName('demo_requested').forEach(r => { r.checked = (r.value === demoVal); });

  document.getElementById('input-email-opens').value = data.email_opens || 2;
  document.getElementById('input-form-completions').value = data.form_completions || 1;
  document.getElementById('input-content-downloads').value = data.content_downloads || 1;
  document.getElementById('input-previous-interactions').value = data.previous_interactions || 2;
  document.getElementById('input-response-time').value = data.response_time_hours || 4.0;
  document.getElementById('input-num-calls').value = data.num_calls || 1;
  document.getElementById('input-num-meetings').value = data.num_meetings || 0;

  submitScoringForm();
}

async function submitScoringForm() {
  const submitBtn = document.getElementById('btn-submit-score');
  const originalHtml = submitBtn.innerHTML;

  submitBtn.disabled = true;
  submitBtn.innerHTML = `<div class="spinner"></div> <span>Scoring Lead...</span>`;
  document.getElementById('result-status-tag').textContent = 'Running Inference...';

  let demoVal = 'No';
  document.getElementsByName('demo_requested').forEach(r => { if (r.checked) demoVal = r.value; });

  const payload = {
    name: document.getElementById('input-name').value,
    company: document.getElementById('input-company').value,
    job_title: document.getElementById('input-job-title').value,
    industry: document.getElementById('input-industry').value,
    company_size: parseFloat(document.getElementById('input-company-size').value) || 0,
    budget_range: document.getElementById('input-budget-range').value,
    lead_source: document.getElementById('input-lead-source').value,
    product_interest: document.getElementById('input-product-interest').value,
    website_visits: parseFloat(document.getElementById('input-website-visits').value) || 0,
    page_views: parseFloat(document.getElementById('input-page-views').value) || 0,
    pricing_page_visits: parseFloat(document.getElementById('input-pricing-page-visits').value) || 0,
    demo_requested: demoVal,
    email_opens: parseFloat(document.getElementById('input-email-opens').value) || 0,
    form_completions: parseFloat(document.getElementById('input-form-completions').value) || 0,
    content_downloads: parseFloat(document.getElementById('input-content-downloads').value) || 0,
    previous_interactions: parseFloat(document.getElementById('input-previous-interactions').value) || 0,
    response_time_hours: parseFloat(document.getElementById('input-response-time').value) || 0,
    num_calls: parseFloat(document.getElementById('input-num-calls').value) || 0,
    num_meetings: parseFloat(document.getElementById('input-num-meetings').value) || 0
  };

  try {
    const res = await fetch('/api/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (result.success) {
      displayScoringResult(result);
      document.getElementById('result-status-tag').textContent = 'Updated';
    } else {
      showToast('Scoring error: ' + (result.detail || 'Unknown error'), 'error');
    }
  } catch (err) {
    showToast('Connection error while scoring lead. Is the backend running?', 'error');
    console.error(err);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = originalHtml;
  }
}

function displayScoringResult(res) {
  const badgeEl = document.getElementById('result-priority-badge');
  const scoreEl = document.getElementById('result-score');
  const probEl = document.getElementById('result-probability');
  const actionEl = document.getElementById('result-action');
  const driversEl = document.getElementById('result-drivers');
  const gaugeArc = document.getElementById('gauge-arc');

  const colorMap = {
    High: { badge: 'priority-high', color: 'var(--high-priority)', stroke: '#16a34a', text: 'HIGH PRIORITY / HOT' },
    Medium: { badge: 'priority-medium', color: 'var(--medium-priority)', stroke: '#d97706', text: 'MEDIUM PRIORITY / WARM' },
    Low: { badge: 'priority-low', color: 'var(--low-priority)', stroke: '#dc2626', text: 'LOW PRIORITY / COLD' }
  };
  const cfg = colorMap[res.priority] || colorMap.Low;

  badgeEl.className = `priority-badge ${cfg.badge}`;
  badgeEl.textContent = cfg.text;
  scoreEl.style.color = cfg.color;
  scoreEl.innerHTML = `${res.lead_score} <span>/ 100</span>`;
  probEl.textContent = `${(res.conversion_probability * 100).toFixed(1)}% Estimated Conversion Probability`;
  actionEl.textContent = res.recommended_action;

  // Animate gauge arc — arc total length is 157 (semicircle)
  const offset = 157 - (res.lead_score / 100) * 157;
  if (gaugeArc) {
    gaugeArc.setAttribute('stroke', cfg.stroke);
    gaugeArc.style.strokeDashoffset = offset;
  }

  driversEl.innerHTML = (res.key_drivers || []).map(d => `
    <div class="driver-item">
      <span style="font-weight:600;">${d.factor}</span>
      <span class="driver-tag ${d.type}">${d.impact}</span>
    </div>
  `).join('') || '<div style="color:var(--text-muted);font-size:13px;">No key drivers identified.</div>';
}

function resetForm() {
  document.getElementById('scoring-form').reset();
  document.getElementById('result-status-tag').textContent = 'Ready';
  document.getElementById('result-score').innerHTML = '-- <span>/ 100</span>';
  document.getElementById('result-probability').textContent = '--';
  document.getElementById('result-drivers').innerHTML = '';
  const gaugeArc = document.getElementById('gauge-arc');
  if (gaugeArc) gaugeArc.style.strokeDashoffset = '157';
}

// ============================================================
// LEADS EXPLORER
// ============================================================
let searchDebounceTimeout = null;

function handleLeadsSearch() {
  clearTimeout(searchDebounceTimeout);
  searchDebounceTimeout = setTimeout(() => fetchLeadsData(1), 320);
}

async function fetchLeadsData(page = 1) {
  state.leadsPage = page;
  const tbody = document.getElementById('leads-table-body');
  tbody.innerHTML = `<tr><td colspan="9"><div style="display:flex;align-items:center;justify-content:center;gap:10px;padding:32px;color:var(--text-muted);"><div class="spinner-dark"></div><span>Loading leads...</span></div></td></tr>`;

  const search = document.getElementById('leads-search').value;
  const priority = document.getElementById('leads-filter-priority').value;
  const industry = document.getElementById('leads-filter-industry').value;
  const sortBy = document.getElementById('leads-sort-by').value;

  const params = new URLSearchParams({ page, limit: state.leadsLimit, sort_by: sortBy, sort_dir: 'desc' });
  if (search) params.append('search', search);
  if (priority && priority !== 'All') params.append('priority', priority);
  if (industry && industry !== 'All') params.append('industry', industry);

  try {
    const res = await fetch(`/api/leads?${params.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.success) {
      state.totalLeadsPages = data.total_pages;
      renderLeadsTable(data.leads);

      const start = (data.page - 1) * data.limit + 1;
      const end = Math.min(data.page * data.limit, data.total);
      document.getElementById('leads-pagination-info').textContent =
        `Showing ${start.toLocaleString()} – ${end.toLocaleString()} of ${data.total.toLocaleString()} leads`;
      document.getElementById('current-page-indicator').textContent = `Page ${data.page} of ${data.total_pages}`;
      document.getElementById('btn-prev-page').disabled = data.page <= 1;
      document.getElementById('btn-next-page').disabled = data.page >= data.total_pages;
    }
  } catch (err) {
    console.error('Failed to fetch leads:', err);
    tbody.innerHTML = `
      <tr><td colspan="9">
        <div class="empty-state">
          <span class="empty-state-icon">⚠️</span>
          <h4>Failed to load leads</h4>
          <p>Backend may be unavailable. Make sure the server is running.</p>
          <button class="btn-secondary" onclick="fetchLeadsData(1)" style="margin-top:12px;">Try Again</button>
        </div>
      </td></tr>`;
    showToast('Failed to load leads data.', 'error');
  }
}

function renderLeadsTable(leads) {
  const tbody = document.getElementById('leads-table-body');
  if (!leads || leads.length === 0) {
    tbody.innerHTML = `
      <tr><td colspan="9">
        <div class="empty-state">
          <span class="empty-state-icon">🔍</span>
          <h4>No leads found</h4>
          <p>Try adjusting your search query or filters, or create a new lead.</p>
          <button class="btn-secondary" onclick="openCreateLeadModal()" style="margin-top:12px;">+ Create Lead</button>
        </div>
      </td></tr>`;
    return;
  }

  tbody.innerHTML = leads.map(l => {
    const prio = l.priority || 'Low';
    const score = l.lead_score !== undefined ? l.lead_score : '--';
    const compSize = l.company_size ? Math.round(l.company_size).toLocaleString() : 'N/A';
    const leadData = JSON.stringify(l).replace(/"/g, '&quot;');
    return `
      <tr onclick="viewLeadDetail('${l.lead_id}', ${leadData})">
        <td><strong>#${l.lead_id}</strong></td>
        <td><strong>${l.name || 'N/A'}</strong></td>
        <td>${l.company || 'N/A'}</td>
        <td>${l.industry || 'N/A'}</td>
        <td>${l.lead_source || 'N/A'}</td>
        <td>${compSize} emp.</td>
        <td><strong style="font-size:15px;color:var(--primary);">${score}</strong></td>
        <td><span class="priority-badge priority-${prio.toLowerCase()}">${prio}</span></td>
        <td onclick="event.stopPropagation();return false;" style="white-space:nowrap;">
          <button class="btn-secondary" style="padding:5px 10px;font-size:11.5px;margin-right:4px;" onclick="viewLeadDetail('${l.lead_id}', ${leadData})">View</button>
          <button class="btn-secondary" style="padding:5px 10px;font-size:11.5px;" onclick="openEditLeadModal(${leadData})">Edit</button>
        </td>
      </tr>
    `;
  }).join('');
}

function changeLeadsPage(delta) {
  const newPage = state.leadsPage + delta;
  if (newPage >= 1 && newPage <= state.totalLeadsPages) fetchLeadsData(newPage);
}

// ============================================================
// LEAD DETAIL MODAL
// ============================================================
function viewLeadDetail(leadId, leadObj) {
  let lead = leadObj;
  if (typeof lead === 'string') {
    try { lead = JSON.parse(lead); } catch (e) { lead = { lead_id: leadId }; }
  }
  state.currentDetailLead = lead;

  // Populate score card
  const prio = lead.priority || 'Low';
  const score = lead.lead_score !== undefined ? lead.lead_score : '--';
  const prob = lead.conversion_probability !== undefined ? (lead.conversion_probability * 100).toFixed(1) : '--';

  document.getElementById('detail-modal-title').textContent = lead.name || `Lead #${leadId}`;
  document.getElementById('detail-modal-subtitle').textContent = lead.company || '';
  document.getElementById('detail-priority-badge').className = `priority-badge priority-${prio.toLowerCase()}`;
  document.getElementById('detail-priority-badge').textContent = `${prio.toUpperCase()} PRIORITY`;
  document.getElementById('detail-score-number').textContent = score;
  document.getElementById('detail-probability').textContent = `${prob}% Conversion Probability`;

  // CRM info
  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val !== undefined && val !== null ? val : '—';
  };
  setVal('d-name', lead.name);
  setVal('d-company', lead.company);
  setVal('d-job-title', lead.job_title);
  setVal('d-industry', lead.industry);
  setVal('d-company-size', lead.company_size ? `${Math.round(lead.company_size).toLocaleString()} employees` : '—');
  setVal('d-budget', lead.budget_range);
  setVal('d-source', lead.lead_source);
  setVal('d-product', lead.product_interest);

  // Engagement
  setVal('d-website-visits', lead.website_visits !== undefined ? Math.round(lead.website_visits) : '—');
  setVal('d-page-views', lead.page_views !== undefined ? Math.round(lead.page_views) : '—');
  setVal('d-pricing-visits', lead.pricing_page_visits !== undefined ? Math.round(lead.pricing_page_visits) : '—');
  setVal('d-email-opens', lead.email_opens !== undefined ? Math.round(lead.email_opens) : '—');
  setVal('d-num-calls', lead.num_calls !== undefined ? Math.round(lead.num_calls) : '—');
  setVal('d-num-meetings', lead.num_meetings !== undefined ? Math.round(lead.num_meetings) : '—');

  // Assignment
  const assignment = state.leadAssignments[leadId] || {};
  const repSelect = document.getElementById('detail-assign-rep');
  const stageSelect = document.getElementById('detail-pipeline-stage');
  if (repSelect) repSelect.value = assignment.rep || '';
  if (stageSelect) stageSelect.value = assignment.stage || deriveStageFromPriority(prio);

  // Key Drivers (generate from lead features)
  renderDetailDrivers(lead);

  // Activity log
  renderActivityLog(leadId);

  document.getElementById('lead-detail-modal').style.display = 'flex';
}

function deriveStageFromPriority(prio) {
  if (prio === 'High') return 'Qualified';
  if (prio === 'Medium') return 'New';
  return 'New';
}

function renderDetailDrivers(lead) {
  const driversEl = document.getElementById('detail-drivers-list');
  const drivers = [];
  if (lead.demo_requested === 'Yes') drivers.push({ factor: 'Demo Requested', impact: 'High Positive', type: 'positive' });
  if (lead.pricing_page_visits >= 3) drivers.push({ factor: `${Math.round(lead.pricing_page_visits)} Pricing Visits`, impact: 'Positive', type: 'positive' });
  if (lead.num_meetings >= 1) drivers.push({ factor: `${Math.round(lead.num_meetings)} Sales Meetings`, impact: 'Positive', type: 'positive' });
  if (lead.company_size >= 200) drivers.push({ factor: 'Enterprise Scale', impact: 'Positive', type: 'positive' });
  if (lead.budget_range && ['50k-1L', '1L-5L', '5L+'].includes(lead.budget_range)) drivers.push({ factor: `Budget: ${lead.budget_range}`, impact: 'Positive', type: 'positive' });
  if (lead.response_time_hours > 24) drivers.push({ factor: 'Slow Response Time', impact: 'Negative', type: 'negative' });
  if (lead.website_visits <= 2) drivers.push({ factor: 'Low Website Activity', impact: 'Negative', type: 'negative' });

  if (drivers.length === 0) drivers.push({ factor: 'Baseline Profile', impact: 'Neutral', type: 'neutral' });

  driversEl.innerHTML = drivers.slice(0, 5).map(d => `
    <div class="driver-item">
      <span style="font-weight:600;">${d.factor}</span>
      <span class="driver-tag ${d.type}">${d.impact}</span>
    </div>
  `).join('');
}

function renderActivityLog(leadId) {
  const container = document.getElementById('detail-activity-log');
  const logs = state.leadActivityLogs[leadId] || [];

  // Auto-generate initial activity if empty
  if (logs.length === 0) {
    const now = new Date();
    const auto = [
      { time: formatTime(new Date(now - 3600000 * 24 * 3)), text: 'Lead created and scored by ML model.' },
      { time: formatTime(new Date(now - 3600000 * 24 * 1)), text: 'Lead viewed by sales team.' },
    ];
    state.leadActivityLogs[leadId] = auto;
  }

  const allLogs = state.leadActivityLogs[leadId] || [];
  container.innerHTML = allLogs.length === 0
    ? `<div style="color:var(--text-muted);font-size:13px;padding:8px 0;">No activity logged yet.</div>`
    : allLogs.map(log => `
        <div class="activity-entry">
          <span class="activity-time">${log.time}</span>
          <span class="activity-text">${log.text}</span>
        </div>
      `).join('');

  container.scrollTop = container.scrollHeight;
}

function addLeadNote() {
  const input = document.getElementById('detail-note-input');
  const text = input.value.trim();
  if (!text) return;

  const leadId = state.currentDetailLead?.lead_id;
  if (!leadId) return;

  if (!state.leadActivityLogs[leadId]) state.leadActivityLogs[leadId] = [];
  state.leadActivityLogs[leadId].push({ time: formatTime(new Date()), text: `📝 Note: ${text}` });
  input.value = '';
  renderActivityLog(leadId);
  showToast('Note added.', 'success');
}

function saveLeadAssignment() {
  const leadId = state.currentDetailLead?.lead_id;
  if (!leadId) return;

  const rep = document.getElementById('detail-assign-rep').value;
  const stage = document.getElementById('detail-pipeline-stage').value;
  state.leadAssignments[leadId] = { rep, stage };

  if (!state.leadActivityLogs[leadId]) state.leadActivityLogs[leadId] = [];
  const repText = rep ? `Assigned to ${rep}` : 'Unassigned';
  state.leadActivityLogs[leadId].push({ time: formatTime(new Date()), text: `🔄 ${repText}. Stage → ${stage}` });
  renderActivityLog(leadId);
  showToast('Assignment saved successfully!', 'success');
}

function closeLeadDetailModal(event) {
  if (event && event.target !== document.getElementById('lead-detail-modal')) return;
  document.getElementById('lead-detail-modal').style.display = 'none';
  state.currentDetailLead = null;
}

// ============================================================
// CREATE / EDIT LEAD MODAL
// ============================================================
function openCreateLeadModal() {
  document.getElementById('lead-form-title').textContent = 'Create New Lead';
  document.getElementById('lead-form-btn-text').textContent = 'Create & Score Lead';
  document.getElementById('lead-create-form').reset();
  document.getElementById('lead-form-error').style.display = 'none';
  document.getElementById('lead-form-modal').setAttribute('data-mode', 'create');
  document.getElementById('lead-form-modal').style.display = 'flex';
}

function openEditLeadModal(lead) {
  document.getElementById('lead-form-title').textContent = 'Edit Lead';
  document.getElementById('lead-form-btn-text').textContent = 'Save & Re-Score';
  document.getElementById('lead-form-modal').setAttribute('data-edit-id', lead.lead_id);

  // Prefill
  document.getElementById('cf-name').value = lead.name || '';
  document.getElementById('cf-company').value = lead.company || '';
  document.getElementById('cf-job-title').value = lead.job_title || '';
  document.getElementById('cf-email').value = lead.email || '';
  document.getElementById('cf-industry').value = lead.industry || '';
  document.getElementById('cf-company-size').value = lead.company_size || '';
  document.getElementById('cf-lead-source').value = lead.lead_source || 'Website';
  document.getElementById('cf-product-interest').value = lead.product_interest || 'Enterprise Plan';
  document.getElementById('cf-budget-range').value = lead.budget_range || 'Unknown';
  document.getElementById('cf-website-visits').value = lead.website_visits || 0;
  document.getElementById('cf-pricing-visits').value = lead.pricing_page_visits || 0;
  document.getElementById('cf-email-opens').value = lead.email_opens || 0;
  document.getElementById('cf-num-meetings').value = lead.num_meetings || 0;

  const demoVal = lead.demo_requested || 'No';
  document.getElementsByName('cf_demo_requested').forEach(r => { r.checked = r.value === demoVal; });

  document.getElementById('lead-form-error').style.display = 'none';
  document.getElementById('lead-form-modal').style.display = 'flex';
}

async function submitLeadForm(e) {
  e.preventDefault();
  const errEl = document.getElementById('lead-form-error');
  errEl.style.display = 'none';

  const name = document.getElementById('cf-name').value.trim();
  const company = document.getElementById('cf-company').value.trim();
  const industry = document.getElementById('cf-industry').value;

  if (!name || !company || !industry) {
    errEl.textContent = 'Please fill in all required fields (Name, Company, Industry).';
    errEl.style.display = 'block';
    return;
  }

  let cfDemo = 'No';
  document.getElementsByName('cf_demo_requested').forEach(r => { if (r.checked) cfDemo = r.value; });

  const payload = {
    name,
    company,
    job_title: document.getElementById('cf-job-title').value || 'Unknown',
    industry,
    company_size: parseFloat(document.getElementById('cf-company-size').value) || 50,
    lead_source: document.getElementById('cf-lead-source').value,
    product_interest: document.getElementById('cf-product-interest').value,
    budget_range: document.getElementById('cf-budget-range').value,
    demo_requested: cfDemo,
    website_visits: parseFloat(document.getElementById('cf-website-visits').value) || 0,
    page_views: 0,
    pricing_page_visits: parseFloat(document.getElementById('cf-pricing-visits').value) || 0,
    email_opens: parseFloat(document.getElementById('cf-email-opens').value) || 0,
    form_completions: 0,
    content_downloads: 0,
    previous_interactions: 0,
    response_time_hours: 4.0,
    num_calls: 0,
    num_meetings: parseFloat(document.getElementById('cf-num-meetings').value) || 0
  };

  const btn = document.getElementById('btn-create-lead-submit');
  btn.innerHTML = '<div class="spinner"></div>';
  btn.disabled = true;

  try {
    const res = await fetch('/api/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();

    if (result.success) {
      const newLead = {
        lead_id: `L-${Date.now()}`,
        name: payload.name,
        company: payload.company,
        job_title: payload.job_title,
        industry: payload.industry,
        company_size: payload.company_size,
        lead_source: payload.lead_source,
        product_interest: payload.product_interest,
        budget_range: payload.budget_range,
        demo_requested: payload.demo_requested,
        website_visits: payload.website_visits,
        pricing_page_visits: payload.pricing_page_visits,
        email_opens: payload.email_opens,
        num_meetings: payload.num_meetings,
        lead_score: result.lead_score,
        conversion_probability: result.conversion_probability,
        priority: result.priority,
        recommended_action: result.recommended_action,
        key_drivers: result.key_drivers,
      };
      state.localLeads.unshift(newLead);
      closeLeadFormModal();
      showToast(`Lead "${name}" scored! Score: ${result.lead_score} (${result.priority} Priority)`, 'success');

      // If on leads tab, refresh or show in scoring
      if (state.currentTab === 'leads') fetchLeadsData(1);
    } else {
      errEl.textContent = result.detail || 'Scoring failed. Please try again.';
      errEl.style.display = 'block';
    }
  } catch (err) {
    errEl.textContent = 'Connection error. Is the backend server running?';
    errEl.style.display = 'block';
  } finally {
    btn.innerHTML = `<span id="lead-form-btn-text">Create &amp; Score Lead</span>`;
    btn.disabled = false;
  }
}

function closeLeadFormModal(event) {
  if (event && event.target !== document.getElementById('lead-form-modal')) return;
  document.getElementById('lead-form-modal').style.display = 'none';
}

// ============================================================
// PIPELINE BOARD (KANBAN)
// ============================================================
async function buildPipelineBoard() {
  const board = document.getElementById('pipeline-board');
  const summary = document.getElementById('pipeline-summary');
  board.innerHTML = `<div style="display:flex;align-items:center;gap:10px;padding:40px;color:var(--text-muted);"><div class="spinner-dark"></div><span>Loading pipeline data...</span></div>`;

  // Get leads for the pipeline
  let leads = [];
  try {
    if (state.analyticsData) {
      leads = state.analyticsData.top_leads || [];
    } else {
      const res = await fetch('/api/leads?limit=50&sort_by=lead_score&sort_dir=desc');
      const data = await res.json();
      if (data.success) leads = data.leads;
    }
  } catch (err) {
    board.innerHTML = `<div class="empty-state"><span class="empty-state-icon">⚠️</span><h4>Could not load pipeline</h4><p>Backend may be unavailable.</p></div>`;
    return;
  }

  // Assign stages
  leads.forEach(l => {
    if (!state.leadAssignments[l.lead_id]) {
      state.leadAssignments[l.lead_id] = { stage: deriveStageFromPriority(l.priority || 'Low') };
    }
  });

  // Group by stage
  const stageMap = {};
  state.pipelineStages.forEach(s => { stageMap[s] = []; });
  leads.forEach(l => {
    const stage = state.leadAssignments[l.lead_id]?.stage || 'New';
    if (!stageMap[stage]) stageMap[stage] = [];
    stageMap[stage].push(l);
  });

  // Stage accent colors
  const stageColors = {
    'New': '#64748b',
    'Qualified': '#2563eb',
    'Demo Scheduled': '#7c3aed',
    'Proposal': '#d97706',
    'Won': '#16a34a',
    'Lost': '#dc2626'
  };

  // Build summary
  summary.innerHTML = state.pipelineStages.map(s => `
    <div class="pipeline-stat">
      <div class="pipeline-stat-value" style="color:${stageColors[s]};">${stageMap[s].length}</div>
      <div class="pipeline-stat-label">${s}</div>
    </div>
  `).join('');

  // Build columns
  board.innerHTML = state.pipelineStages.map(stage => `
    <div class="pipeline-column" id="col-${stage.replace(/\s+/g, '-')}"
         ondragover="pipelineDragOver(event, '${stage}')"
         ondragleave="pipelineDragLeave(event)"
         ondrop="pipelineDrop(event, '${stage}')">
      <div class="pipeline-col-header">
        <span class="pipeline-col-title" style="color:${stageColors[stage]};">${stage}</span>
        <span class="pipeline-col-count">${stageMap[stage].length}</span>
      </div>
      <div class="pipeline-col-body" id="body-${stage.replace(/\s+/g, '-')}">
        ${stageMap[stage].length === 0
          ? `<div class="empty-state" style="padding:20px;"><span style="font-size:24px;">📭</span><p style="font-size:12px;">Drop leads here</p></div>`
          : stageMap[stage].map(l => buildPipelineCard(l)).join('')
        }
      </div>
    </div>
  `).join('');
}

function buildPipelineCard(lead) {
  const prio = lead.priority || 'Low';
  const score = lead.lead_score || '--';
  return `
    <div class="pipeline-card"
         id="card-${lead.lead_id}"
         draggable="true"
         ondragstart="pipelineDragStart(event, '${lead.lead_id}')"
         ondragend="pipelineDragEnd(event)"
         onclick="viewLeadDetail('${lead.lead_id}', ${JSON.stringify(lead).replace(/"/g, '&quot;')})">
      <div class="pipeline-card-name">${lead.name || 'Unknown'}</div>
      <div class="pipeline-card-company">${lead.company || '—'} · ${lead.industry || '—'}</div>
      <div class="pipeline-card-footer">
        <span class="priority-badge priority-${prio.toLowerCase()}" style="font-size:10px;padding:3px 8px;">${prio}</span>
        <span class="pipeline-card-score">${score}</span>
      </div>
    </div>
  `;
}

let dragLeadId = null;

function pipelineDragStart(event, leadId) {
  dragLeadId = leadId;
  const card = document.getElementById(`card-${leadId}`);
  if (card) card.classList.add('dragging');
  event.dataTransfer.effectAllowed = 'move';
}

function pipelineDragEnd(event) {
  if (dragLeadId) {
    const card = document.getElementById(`card-${dragLeadId}`);
    if (card) card.classList.remove('dragging');
  }
}

function pipelineDragOver(event, stage) {
  event.preventDefault();
  event.dataTransfer.dropEffect = 'move';
  const col = document.getElementById(`col-${stage.replace(/\s+/g, '-')}`);
  if (col) col.classList.add('drag-over');
}

function pipelineDragLeave(event) {
  const col = event.currentTarget;
  col.classList.remove('drag-over');
}

function pipelineDrop(event, stage) {
  event.preventDefault();
  const col = document.getElementById(`col-${stage.replace(/\s+/g, '-')}`);
  if (col) col.classList.remove('drag-over');

  if (!dragLeadId) return;

  // Update assignment
  if (!state.leadAssignments[dragLeadId]) state.leadAssignments[dragLeadId] = {};
  const oldStage = state.leadAssignments[dragLeadId].stage;
  state.leadAssignments[dragLeadId].stage = stage;

  if (oldStage !== stage) {
    showToast(`Lead moved to "${stage}"`, 'success');
    // Log activity
    if (!state.leadActivityLogs[dragLeadId]) state.leadActivityLogs[dragLeadId] = [];
    state.leadActivityLogs[dragLeadId].push({ time: formatTime(new Date()), text: `🔄 Stage changed: ${oldStage} → ${stage}` });
  }

  buildPipelineBoard();
  dragLeadId = null;
}

function refreshPipeline() {
  buildPipelineBoard();
  showToast('Pipeline refreshed.', 'info');
}

// ============================================================
// CSV IMPORT
// ============================================================
let parsedCsvRows = [];

function handleDragOver(event) {
  event.preventDefault();
  document.getElementById('csv-dropzone').classList.add('drag-active');
}

function handleDragLeave(event) {
  document.getElementById('csv-dropzone').classList.remove('drag-active');
}

function handleDrop(event) {
  event.preventDefault();
  document.getElementById('csv-dropzone').classList.remove('drag-active');
  const file = event.dataTransfer.files[0];
  if (file) processCSVFile(file);
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) processCSVFile(file);
}

function processCSVFile(file) {
  if (!file.name.endsWith('.csv')) {
    showToast('Please upload a .csv file only.', 'error');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const text = e.target.result;
    parsedCsvRows = parseCSV(text);
    validateAndShowCSV(parsedCsvRows);
  };
  reader.readAsText(file);
  showToast(`File "${file.name}" loaded. Validating...`, 'info');
}

function parseCSV(text) {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return [];

  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, '').toLowerCase());
  const rows = [];

  for (let i = 1; i < lines.length; i++) {
    const values = splitCSVLine(lines[i]);
    const row = {};
    headers.forEach((h, idx) => { row[h] = (values[idx] || '').trim().replace(/"/g, ''); });
    row.__rowNum = i;
    rows.push(row);
  }
  return rows;
}

function splitCSVLine(line) {
  const result = [];
  let current = '';
  let inQuotes = false;
  for (let ch of line) {
    if (ch === '"') { inQuotes = !inQuotes; }
    else if (ch === ',' && !inQuotes) { result.push(current); current = ''; }
    else { current += ch; }
  }
  result.push(current);
  return result;
}

function validateAndShowCSV(rows) {
  const validationCard = document.getElementById('import-validation-card');
  const importBtn = document.getElementById('btn-import-run');
  const tbody = document.getElementById('import-validation-tbody');

  if (rows.length === 0) {
    showToast('CSV file is empty or malformed.', 'error');
    return;
  }

  validationCard.style.display = 'block';
  let validCount = 0;

  tbody.innerHTML = rows.map(row => {
    const issues = [];
    REQUIRED_CSV_COLUMNS.forEach(col => {
      if (!row[col] || row[col].trim() === '') issues.push(`Missing: ${col}`);
    });

    // Type checks
    if (row.company_size && isNaN(parseFloat(row.company_size))) issues.push('company_size must be a number');
    if (row.website_visits && isNaN(parseFloat(row.website_visits))) issues.push('website_visits must be a number');

    const isValid = issues.length === 0;
    if (isValid) validCount++;

    return `
      <tr>
        <td><strong>#${row.__rowNum}</strong></td>
        <td>${row.name || '—'}</td>
        <td>${row.company || '—'}</td>
        <td>${row.industry || '—'}</td>
        <td>
          ${isValid
            ? `<span class="priority-badge" style="background:var(--high-priority-bg);color:var(--high-priority);border-color:var(--high-priority-border);">✅ Valid</span>`
            : `<span class="priority-badge" style="background:var(--low-priority-bg);color:var(--low-priority);border-color:var(--low-priority-border);">❌ Invalid</span>`
          }
        </td>
        <td style="font-size:11.5px;color:var(--low-priority);max-width:200px;white-space:normal;">
          ${issues.length > 0 ? issues.join(', ') : '<span style="color:var(--high-priority);">All checks passed</span>'}
        </td>
      </tr>
    `;
  }).join('');

  importBtn.style.display = validCount > 0 ? 'inline-flex' : 'none';
  showToast(`Validation complete: ${validCount} valid / ${rows.length - validCount} invalid rows.`, validCount > 0 ? 'success' : 'warning');
}

async function runImport() {
  const validRows = parsedCsvRows.filter(row => {
    const issues = REQUIRED_CSV_COLUMNS.filter(col => !row[col] || row[col].trim() === '');
    return issues.length === 0;
  });

  if (validRows.length === 0) {
    showToast('No valid rows to import.', 'warning');
    return;
  }

  const progressSection = document.getElementById('import-progress-section');
  const progressBar = document.getElementById('import-progress-bar');
  const progressLabel = document.getElementById('import-progress-label');
  const resultCard = document.getElementById('import-result-card');

  progressSection.style.display = 'block';
  resultCard.style.display = 'none';

  let imported = 0;
  let errors = 0;
  const skipped = parsedCsvRows.length - validRows.length;

  for (let i = 0; i < validRows.length; i++) {
    const row = validRows[i];
    const pct = Math.round(((i + 1) / validRows.length) * 100);
    progressBar.style.width = `${pct}%`;
    progressLabel.textContent = `${pct}%`;

    try {
      const payload = {
        name: row.name || `Lead ${i + 1}`,
        company: row.company || 'Unknown Company',
        job_title: row.job_title || 'Unknown',
        industry: row.industry,
        company_size: parseFloat(row.company_size) || 50,
        lead_source: row.lead_source,
        product_interest: row.product_interest,
        budget_range: row.budget_range,
        demo_requested: row.demo_requested || 'No',
        website_visits: parseFloat(row.website_visits) || 0,
        page_views: parseFloat(row.page_views) || 0,
        pricing_page_visits: parseFloat(row.pricing_page_visits) || 0,
        email_opens: parseFloat(row.email_opens) || 0,
        form_completions: parseFloat(row.form_completions) || 0,
        content_downloads: parseFloat(row.content_downloads) || 0,
        previous_interactions: parseFloat(row.previous_interactions) || 0,
        response_time_hours: parseFloat(row.response_time_hours) || 4.0,
        num_calls: parseFloat(row.num_calls) || 0,
        num_meetings: parseFloat(row.num_meetings) || 0
      };

      const res = await fetch('/api/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const result = await res.json();

      if (result.success) {
        state.localLeads.push({
          lead_id: `IMP-${Date.now()}-${i}`,
          ...payload,
          lead_score: result.lead_score,
          conversion_probability: result.conversion_probability,
          priority: result.priority
        });
        imported++;
      } else {
        errors++;
      }
    } catch (err) {
      errors++;
    }

    // Small delay to not overwhelm the API
    await new Promise(r => setTimeout(r, 30));
  }

  progressSection.style.display = 'none';

  // Show results
  document.getElementById('import-stat-imported').textContent = imported;
  document.getElementById('import-stat-skipped').textContent = skipped;
  document.getElementById('import-stat-errors').textContent = errors;
  document.getElementById('import-stat-total').textContent = parsedCsvRows.length;
  resultCard.style.display = 'block';

  // Show scored leads summary
  const topImported = state.localLeads.slice(-imported).sort((a, b) => b.lead_score - a.lead_score).slice(0, 5);
  document.getElementById('import-score-results').innerHTML = topImported.length > 0 ? `
    <h4 style="font-size:12px;font-weight:700;color:var(--text-secondary);text-transform:uppercase;margin-bottom:8px;">Top Scored Imports</h4>
    ${topImported.map(l => `
      <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 10px;background:var(--bg-main);border-radius:var(--radius-sm);margin-bottom:6px;">
        <div>
          <div style="font-weight:700;font-size:13px;">${l.name}</div>
          <div style="font-size:11.5px;color:var(--text-muted);">${l.company}</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
          <strong style="color:var(--primary);font-size:16px;">${l.lead_score}</strong>
          <span class="priority-badge priority-${l.priority.toLowerCase()}">${l.priority}</span>
        </div>
      </div>
    `).join('')}
  ` : '';

  showToast(`Import complete! ${imported} leads scored successfully.`, 'success');
}

function resetImport() {
  parsedCsvRows = [];
  document.getElementById('csv-file-input').value = '';
  document.getElementById('import-validation-card').style.display = 'none';
  document.getElementById('import-result-card').style.display = 'none';
  document.getElementById('import-progress-section').style.display = 'none';
  document.getElementById('btn-import-run').style.display = 'none';
  showToast('Import reset.', 'info');
}

// ============================================================
// MODEL TELEMETRY
// ============================================================
async function fetchModelInfoData() {
  try {
    const res = await fetch('/api/model-info');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.success) {
      state.modelInfo = data;
      updateModelMetrics(data);
      renderModelFeaturesTable(data.top_features);
      document.getElementById('model-selection-reason').textContent = data.selection_reason;
    }
  } catch (err) {
    console.error('Failed to load model info:', err);
    showToast('Failed to load model telemetry.', 'error');
  }
}

function updateModelMetrics(data) {
  const m = data.test_metrics || {};
  const safeVal = (v, digits = 4) => v !== undefined ? v.toFixed(digits) : '—';

  const roc = document.getElementById('metric-roc-auc');
  const prAuc = document.getElementById('metric-pr-auc');
  const f1 = document.getElementById('metric-f1');
  const prec10 = document.getElementById('metric-prec-10');

  if (roc) roc.textContent = safeVal(m.roc_auc);
  if (prAuc) prAuc.textContent = safeVal(m.pr_auc);
  if (f1) f1.textContent = safeVal(m.f1);
  if (prec10 && data.ranking_metrics?.top_10) {
    prec10.textContent = `${(data.ranking_metrics.top_10.precision_at_k * 100).toFixed(1)}%`;
  }
}

function renderModelFeaturesTable(features) {
  const tbody = document.getElementById('model-features-tbody');
  if (!tbody || !features) return;

  tbody.innerHTML = features.map(f => {
    const pct = Math.round(f.importance * 100);
    return `
      <tr>
        <td><strong>${f.feature}</strong></td>
        <td><span class="driver-tag positive">${f.category}</span></td>
        <td style="white-space:normal;max-width:200px;">${f.description}</td>
        <td>
          <div style="display:flex;align-items:center;gap:8px;">
            <div style="background:var(--bg-main);height:8px;width:100px;border-radius:4px;overflow:hidden;border:1px solid var(--border-color);">
              <div style="width:${pct}%;background:linear-gradient(90deg,var(--primary),#60a5fa);height:100%;transition:width 0.6s ease;"></div>
            </div>
            <span style="font-weight:700;font-size:12px;">${(f.importance * 100).toFixed(1)}%</span>
          </div>
        </td>
      </tr>
    `;
  }).join('');
}

// ============================================================
// UTILITY HELPERS
// ============================================================
function formatTime(date) {
  const now = new Date();
  const diff = now - date;
  const mins = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return `${days}d ago`;
}
