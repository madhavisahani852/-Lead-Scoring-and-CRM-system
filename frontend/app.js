/**
 * frontend/app.js
 * Lead CRM — Single Page Application View Controller
 * Enforces strict single-view visibility via hash routing:
 * #/dashboard | #/leads | #/score | #/pipeline | #/analytics
 */

// ── Application State ──────────────────────────────────────────────────────────
function getApiBase() {
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return '';
  }
  if (window.location.origin && window.location.origin.includes('onrender.com')) {
    return '';
  }
  return 'https://lead-scoring-and-crm-system.onrender.com';
}
const API = getApiBase();
let currentView = 'dashboard';
let currentActiveLeadId = null;
let allLeadsCache = [];
let workspaceMembersCache = [];
let currentUser = null;
let currentPage = 1;
const PAGE_LIMIT = 50;
let totalLeadsCount = 0;
let csvFileSelected = null;
let lastScoredResult = null;

function authHeaders(headers = {}) {
  const token = localStorage.getItem('crm_token');
  const h = { ...headers };
  if (token) {
    h['Authorization'] = `Bearer ${token}`;
  }
  return h;
}

// ── Navigation Configuration ──────────────────────────────────────────────────
const VIEWS = {
  dashboard: {
    title: 'Dashboard',
    subtitle: 'Overview of leads, scores, and pipeline',
    nav: 'nav-dashboard',
  },
  leads: {
    title: 'Leads',
    subtitle: 'Browse, filter, and manage your leads',
    nav: 'nav-leads',
  },
  score: {
    title: 'Score Lead',
    subtitle: 'Compute real ML prediction for a new lead',
    nav: 'nav-scoring',
  },
  pipeline: {
    title: 'Pipeline',
    subtitle: 'Visual kanban board for your lead pipeline',
    nav: 'nav-pipeline',
  },
  'csv-uploads': {
    title: 'CSV Uploads',
    subtitle: 'Import leads and view upload history',
    nav: 'nav-csv-uploads',
  },
  analytics: {
    title: 'Analytics',
    subtitle: 'Source performance, conversion rates, and score distribution',
    nav: 'nav-analytics',
  },
};

function getRouteFromHash() {
  const raw = window.location.hash || '';
  const cleaned = raw.replace(/^[#/]+/, '').split('?')[0].split('/')[0].trim().toLowerCase();
  return VIEWS[cleaned] ? cleaned : 'dashboard';
}

function navigateTo(view) {
  const target = VIEWS[view] ? view : 'dashboard';
  const targetHash = `#/${target}`;
  if (window.location.hash === targetHash) {
    handleRouteChange();
  } else {
    window.location.hash = targetHash;
  }
}

function showAuthGate() {
  document.querySelectorAll('.page-view').forEach(el => el.classList.remove('active'));
  const gate = document.getElementById('auth-gate-view');
  if (gate) {
    gate.classList.add('active');
  }
  const titleEl = document.getElementById('current-view-title');
  const subEl = document.getElementById('current-view-subtitle');
  if (titleEl) titleEl.textContent = 'Sign In';
  if (subEl) subEl.textContent = 'Please authenticate to access your CRM workspace';
  const topAction = document.getElementById('top-bar-action');
  if (topAction) topAction.style.display = 'none';
}

function hideAuthGate() {
  const gate = document.getElementById('auth-gate-view');
  if (gate) {
    gate.classList.remove('active');
  }
  const topAction = document.getElementById('top-bar-action');
  if (topAction) topAction.style.display = 'flex';
}

function handleRouteChange() {
  if (!currentUser && !localStorage.getItem('crm_token')) {
    showAuthGate();
    return;
  }
  hideAuthGate();

  const view = getRouteFromHash();
  currentView = view;

  // 1. STRICT VIEW SWITCHING: Hide ALL views first
  document.querySelectorAll('.page-view').forEach(el => el.classList.remove('active'));

  // 2. Activate ONLY the target view
  const targetSection = document.getElementById(`${view}-view`);
  if (targetSection) {
    targetSection.classList.add('active');
  }

  // 3. Update sidebar active state
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  const navId = VIEWS[view]?.nav;
  if (navId) {
    const navEl = document.getElementById(navId);
    if (navEl) navEl.classList.add('active');
  }

  // 4. Update top bar text
  const titleEl = document.getElementById('current-view-title');
  const subEl = document.getElementById('current-view-subtitle');
  if (titleEl) titleEl.textContent = VIEWS[view].title;
  if (subEl) subEl.textContent = VIEWS[view].subtitle;

  // 5. Always scroll to top of viewport
  window.scrollTo(0, 0);

  // 6. Fetch data for active view
  switch (view) {
    case 'dashboard':
      loadDashboard();
      break;
    case 'leads':
      loadLeads();
      break;
    case 'pipeline':
      loadPipeline();
      break;
    case 'analytics':
      loadAnalytics();
      break;
    case 'score':
      // Score form does not require initial fetch
      break;
    case 'csv-uploads':
      loadCsvHistory();
      break;
  }
}

window.addEventListener('hashchange', handleRouteChange);

// ── Initialization ────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  checkCurrentUser();
  loadWorkspaceMembers();

  if (!window.location.hash) {
    window.location.hash = '#/dashboard';
  }
  handleRouteChange();

  // CSV Drag and Drop
  const dropZone = document.getElementById('csv-import-drop-zone');
  if (dropZone) {
    dropZone.addEventListener('dragover', e => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file) setImportFile(file);
    });
  }
});


// ── DASHBOARD VIEW ─────────────────────────────────────────────────────────────
async function loadDashboard() {
  loadAlerts();
  try {
    const res = await fetch(`${API}/api/dashboard`, { headers: authHeaders() });
    const data = await res.json();
    if (!data.success) throw new Error('Dashboard load failed');
    const d = data.dashboard;

    // Stat Cards
    document.getElementById('dash-total-leads').textContent = fmt(d.total_leads);
    document.getElementById('dash-hot-leads').textContent = fmt(d.hot_leads);
    document.getElementById('dash-warm-leads').textContent = fmt(d.warm_leads);
    document.getElementById('dash-cold-leads').textContent = fmt(d.cold_leads);
    document.getElementById('dash-conv-rate').textContent = `${d.conversion_rate}%`;

    // Pipeline Overview Strip
    renderDashPipelineStrip(d.pipeline_counts || {});

    // Priority Queue
    renderPriorityQueue(d.priority_leads || []);

    // Recent Activity
    renderRecentActivity(d.recent_activities || []);

    // Recent Leads Table
    renderDashRecentLeads(d.recent_leads || []);
  } catch (err) {
    console.error('Dashboard error:', err);
    showToast('Failed to load dashboard data', 'error');
  }
}

function renderDashPipelineStrip(counts) {
  const el = document.getElementById('dash-pipeline-strip');
  if (!el) return;
  const stages = [
    'New', 'Contacted', 'Qualified', 'Demo Scheduled',
    'Proposal', 'Negotiation', 'Won', 'Lost'
  ];
  el.innerHTML = stages.map(s => `
    <div class="pipeline-overview-item" onclick="navigateTo('pipeline')" style="cursor:pointer;" title="View in Pipeline">
      <div class="pipeline-overview-label">${s}</div>
      <div class="pipeline-overview-count">${counts[s] ?? 0}</div>
    </div>
  `).join('');
}

function renderPriorityQueue(leads) {
  const el = document.getElementById('dash-priority-queue');
  if (!leads.length) {
    el.innerHTML = '<div class="table-empty">No priority leads</div>';
    return;
  }
  el.innerHTML = leads.slice(0, 8).map(l => `
    <div class="priority-item" onclick="openDetailsModal('${esc(l.id)}')">
      <div class="priority-item-left">
        <div class="priority-avatar ${catClass(l.category)}">${initials(l.name)}</div>
        <div>
          <div class="priority-name">${esc(l.name)}</div>
          <div class="priority-sub">${esc(l.company || '')} · ${esc(l.industry || '')}</div>
        </div>
      </div>
      <div class="priority-item-right">
        <div class="score-pill ${catClass(l.category)}">${l.score ?? '—'}</div>
        <span class="badge ${badgeClass(l.category)}">${l.category || '—'}</span>
      </div>
    </div>
  `).join('');
}

function renderRecentActivity(activities) {
  const el = document.getElementById('dash-recent-activity');
  if (!activities.length) {
    el.innerHTML = '<div class="table-empty">No recent activity</div>';
    return;
  }
  el.innerHTML = activities.slice(0, 8).map(a => `
    <div class="activity-item">
      <div class="activity-icon">${activityIcon(a.activity_type)}</div>
      <div>
        <div class="activity-desc">${esc(a.description)}</div>
        <div class="activity-time">${relativeTime(a.created_at)}</div>
      </div>
    </div>
  `).join('');
}

function renderDashRecentLeads(leads) {
  const tbody = document.getElementById('dash-recent-leads-tbody');
  if (!leads.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="table-empty">No recent leads</td></tr>';
    return;
  }
  tbody.innerHTML = leads.slice(0, 8).map(l => `
    <tr>
      <td><span style="font-weight:600; color:var(--primary-brown); cursor:pointer;" onclick="openDetailsModal('${esc(l.id)}')">${esc(l.name)}</span></td>
      <td>${esc(l.company || '—')}</td>
      <td><span class="score-pill ${catClass(l.category)}">${l.score ?? '—'}</span></td>
      <td><span class="badge ${badgeClass(l.category)}">${l.category || '—'}</span></td>
      <td><span class="badge-stage stage-${esc(l.pipeline_stage || 'New').replace(/\s+/g, '-')}">${esc(l.pipeline_stage || 'New')}</span></td>
      <td>${esc(l.lead_source || '—')}</td>
      <td><button class="btn btn-secondary btn-sm" onclick="openDetailsModal('${esc(l.id)}')">View</button></td>
    </tr>
  `).join('');
}


// ── LEADS VIEW ─────────────────────────────────────────────────────────────────
function handleLeadsFilterChange() {
  currentPage = 1;
  loadLeads();
}

async function loadLeads() {
  const search = document.getElementById('leads-search-input')?.value || '';
  const category = document.getElementById('leads-category-filter')?.value || 'All';
  const source = document.getElementById('leads-source-filter')?.value || 'All';
  const industry = document.getElementById('leads-industry-filter')?.value || 'All';
  const assigned = document.getElementById('leads-assigned-filter')?.value || 'All';
  const sortBy = document.getElementById('leads-sort-by')?.value || 'score';
  const sortDir = 'desc';

  const params = new URLSearchParams({
    page: currentPage,
    limit: PAGE_LIMIT,
    sort_by: sortBy,
    sort_dir: sortDir,
  });
  if (search) params.set('search', search);
  if (category !== 'All') params.set('category', category);
  if (source !== 'All') params.set('lead_source', source);
  if (industry !== 'All') params.set('industry', industry);
  if (assigned !== 'All') params.set('assigned_to', assigned);

  const tbody = document.getElementById('leads-table-tbody');
  tbody.innerHTML = '<tr><td colspan="9" class="table-empty">Loading leads...</td></tr>';

  try {
    const res = await fetch(`${API}/api/leads?${params}`);
    const data = await res.json();
    if (!data.success) throw new Error('Leads load failed');

    allLeadsCache = data.leads || [];
    totalLeadsCount = data.total || 0;

    const countLabel = document.getElementById('leads-count-label');
    if (countLabel) countLabel.textContent = `${totalLeadsCount.toLocaleString()} leads`;

    renderLeadsTable(allLeadsCache);
    renderPagination(data.page, data.total, data.limit);
  } catch (err) {
    console.error('Leads error:', err);
    tbody.innerHTML = '<tr><td colspan="9" class="table-empty" style="color:var(--status-hot);">Failed to load leads.</td></tr>';
  }
}

function renderLeadsTable(leads) {
  const tbody = document.getElementById('leads-table-tbody');
  if (!leads.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="table-empty">No leads match your filters.</td></tr>';
    return;
  }
  tbody.innerHTML = leads.map(l => `
    <tr>
      <td>
        <span style="font-weight:600; color:var(--primary-brown); cursor:pointer;" onclick="openDetailsModal('${esc(l.id)}')">${esc(l.name)}</span>
      </td>
      <td style="font-size: 12px; color: var(--text-muted);">${esc(l.email || '—')}</td>
      <td>${esc(l.company || '—')}</td>
      <td>${esc(l.industry || '—')}</td>
      <td><span class="score-pill ${catClass(l.category)}">${l.score ?? '—'}</span></td>
      <td><span class="badge ${badgeClass(l.category)}">${l.category || '—'}</span></td>
      <td>
        <select class="form-control" style="font-size:12px; padding:3px 6px; width:130px;" onchange="quickPipelineChange('${esc(l.id)}', this.value)">
          ${pipelineOptions(l.pipeline_stage || l.status || 'New')}
        </select>
      </td>
      <td style="font-size: 12px;">${esc(l.lead_source || '—')}</td>
      <td>
        <div style="display: flex; gap: 4px;">
          <button class="btn btn-secondary btn-sm" onclick="openDetailsModal('${esc(l.id)}')">View</button>
          <button class="btn btn-secondary btn-sm" onclick="openEditModal('${esc(l.id)}')">Edit</button>
          <button class="btn btn-danger btn-sm" onclick="deleteLead('${esc(l.id)}', '${esc(l.name)}')">Del</button>
        </div>
      </td>
    </tr>
  `).join('');
}

function pipelineOptions(current) {
  const stages = [
    'New', 'Contacted', 'Qualified', 'Demo Scheduled',
    'Proposal', 'Negotiation', 'Won', 'Lost'
  ];
  return stages.map(s => `<option value="${s}" ${s === current ? 'selected' : ''}>${s}</option>`).join('');
}

async function quickPipelineChange(leadId, stage) {
  try {
    const res = await fetch(`${API}/api/leads/${leadId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pipeline_stage: stage, status: stage }),
    });
    const data = await res.json();
    if (!data.success) throw new Error('Stage change failed');
    showToast(`Stage updated to ${stage}`, 'success');
  } catch (err) {
    showToast('Failed to update stage', 'error');
  }
}

function renderPagination(page, total, limit) {
  const container = document.getElementById('leads-pagination');
  if (!container) return;
  const totalPages = Math.ceil(total / limit);
  if (totalPages <= 1) {
    container.style.display = 'none';
    return;
  }
  container.style.display = 'flex';
  document.getElementById('page-info').textContent = `Page ${page} of ${totalPages} (${total.toLocaleString()} leads)`;
  document.getElementById('btn-prev-page').disabled = page <= 1;
  document.getElementById('btn-next-page').disabled = page >= totalPages;
}

function changePage(delta) {
  currentPage += delta;
  if (currentPage < 1) currentPage = 1;
  loadLeads();
}


// ── SCORE LEAD VIEW ────────────────────────────────────────────────────────────
async function handleCalculateScore(event) {
  event.preventDefault();
  const btn = document.getElementById('score-submit-btn');
  btn.textContent = 'Calculating...';
  btn.disabled = true;

  const leadData = {
    name: document.getElementById('score-input-name').value,
    email: document.getElementById('score-input-email').value,
    company: document.getElementById('score-input-company').value,
    phone: document.getElementById('score-input-phone').value,
    job_title: document.getElementById('score-input-job').value,
    location: document.getElementById('score-input-location').value,
    industry: document.getElementById('score-input-industry').value,
    company_size: parseFloat(document.getElementById('score-input-company-size').value) || 250,
    lead_source: document.getElementById('score-input-source').value,
    product_interest: document.getElementById('score-input-product').value,
    budget_range: document.getElementById('score-input-budget').value,
    demo_requested: document.getElementById('score-input-demo').value,
    website_visits: parseFloat(document.getElementById('score-input-visits').value) || 0,
    page_views: parseFloat(document.getElementById('score-input-views').value) || 0,
    pricing_page_visits: parseFloat(document.getElementById('score-input-pricing').value) || 0,
    email_opens: parseFloat(document.getElementById('score-input-email-opens').value) || 0,
    form_completions: parseFloat(document.getElementById('score-input-form').value) || 0,
    content_downloads: parseFloat(document.getElementById('score-input-downloads').value) || 0,
    previous_interactions: parseFloat(document.getElementById('score-input-interactions').value) || 0,
    response_time_hours: parseFloat(document.getElementById('score-input-response').value) || 0,
    num_calls: parseFloat(document.getElementById('score-input-calls').value) || 0,
    num_meetings: parseFloat(document.getElementById('score-input-meetings').value) || 0,
  };

  try {
    const res = await fetch(`${API}/api/score`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(leadData),
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.detail || 'Prediction failed');

    lastScoredResult = { ...data, ...leadData };
    renderScoreResult(data);
    showToast('Score calculated via production XGBoost model!', 'success');
  } catch (err) {
    showToast('Scoring error: ' + err.message, 'error');
  } finally {
    btn.textContent = 'Calculate Score';
    btn.disabled = false;
  }
}

function renderScoreResult(data) {
  const container = document.getElementById('score-result-container');
  container.style.display = 'block';

  document.getElementById('result-score-value').textContent = data.score;
  const catEl = document.getElementById('result-category-badge');
  catEl.textContent = data.category;
  catEl.className = `badge ${badgeClass(data.category)}`;

  document.getElementById('result-prob-value').textContent = `${data.conversion_probability_pct}%`;
  document.getElementById('result-priority-badge').textContent = `Priority: ${data.priority}`;

  const expl = data.explanation;
  if (expl) {
    const pos = expl.positive_signals || [];
    const neg = expl.negative_signals || [];
    document.getElementById('score-explanation').innerHTML = `
      <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">${esc(expl.summary || '')}</p>
      <div class="signals-container">
        <div class="signal-box">
          <div class="signal-box-title positive">✅ Positive Signals</div>
          ${pos.map(s => `
            <div class="signal-item">
              <strong>${esc(s.label)}</strong>
              <div style="color: var(--text-muted); font-size: 11px;">${esc(s.reason)}</div>
            </div>
          `).join('') || '<div style="color: var(--text-muted); font-size: 12px;">No strong positive signals</div>'}
        </div>
        <div class="signal-box">
          <div class="signal-box-title negative">⚠️ Negative Signals</div>
          ${neg.map(s => `
            <div class="signal-item">
              <strong>${esc(s.label)}</strong>
              <div style="color: var(--text-muted); font-size: 11px;">${esc(s.reason)}</div>
            </div>
          `).join('') || '<div style="color: var(--text-muted); font-size: 12px;">No negative signals detected</div>'}
        </div>
      </div>
    `;
  }
}

function resetScoreForm() {
  document.getElementById('score-lead-form').reset();
  document.getElementById('score-result-container').style.display = 'none';
  lastScoredResult = null;
}

async function handleSaveScoredLead() {
  if (!lastScoredResult) {
    showToast('Please calculate a score first', 'error');
    return;
  }
  const r = lastScoredResult;
  const payload = {
    name: r.name || 'New Lead',
    email: r.email || '',
    phone: r.phone || '',
    company: r.company || '',
    job_title: r.job_title || '',
    location: r.location || '',
    industry: r.industry || 'SaaS',
    company_size: r.company_size || 250,
    lead_source: r.lead_source || 'Website',
    product_interest: r.product_interest || '',
    budget_range: r.budget_range || 'Unknown',
    website_visits: r.website_visits || 0,
    page_views: r.page_views || 0,
    pricing_page_visits: r.pricing_page_visits || 0,
    demo_requested: r.demo_requested || 'No',
    email_opens: r.email_opens || 0,
    form_completions: r.form_completions || 0,
    content_downloads: r.content_downloads || 0,
    previous_interactions: r.previous_interactions || 0,
    response_time_hours: r.response_time_hours || 0,
    num_calls: r.num_calls || 0,
    num_meetings: r.num_meetings || 0,
    score: r.score,
    conversion_probability: r.conversion_probability,
    category: r.category,
    priority: r.priority,
    status: 'New',
    pipeline_stage: 'New',
  };

  try {
    const res = await fetch(`${API}/api/leads`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.detail || 'Save failed');
    showToast('Lead successfully saved to CRM!', 'success');
    lastScoredResult = null;
    document.getElementById('score-result-container').style.display = 'none';
    navigateTo('leads');
  } catch (err) {
    showToast('Save failed: ' + err.message, 'error');
  }
}


// ── PIPELINE VIEW ──────────────────────────────────────────────────────────────
async function loadPipeline() {
  const board = document.getElementById('pipeline-board');
  board.innerHTML = '<div class="table-empty">Loading pipeline...</div>';

  try {
    const res = await fetch(`${API}/api/leads?sort_by=score&sort_dir=desc&limit=200`);
    const data = await res.json();
    if (!data.success) throw new Error('Pipeline load failed');

    const stages = [
      'New', 'Contacted', 'Qualified', 'Demo Scheduled',
      'Proposal', 'Negotiation', 'Won', 'Lost'
    ];
    const byStage = {};
    stages.forEach(s => (byStage[s] = []));

    (data.leads || []).forEach(l => {
      const stage = l.pipeline_stage || l.status || 'New';
      if (!byStage[stage]) byStage[stage] = [];
      byStage[stage].push(l);
    });

    board.innerHTML = stages.map(stage => `
      <div class="pipeline-column">
        <div class="pipeline-header">
          <span class="pipeline-title">${stage}</span>
          <span class="pipeline-count">${byStage[stage].length}</span>
        </div>
        <div class="pipeline-cards">
          ${byStage[stage].length === 0
        ? '<div style="color: var(--text-muted); font-size: 11px; text-align: center; padding: 16px 0;">No leads</div>'
        : byStage[stage].slice(0, 25).map(l => renderPipelineCard(l)).join('')
      }
        </div>
      </div>
    `).join('');
  } catch (err) {
    board.innerHTML = '<div class="table-empty" style="color:var(--status-hot);">Failed to load pipeline</div>';
    showToast('Pipeline load failed', 'error');
  }
}

function renderPipelineCard(l) {
  const prob = l.conversion_probability != null ? `${Math.round(l.conversion_probability * 100)}%` : (l.score != null ? `${l.score}%` : '—');
  return `
    <div class="pipeline-card">
      <div class="pipeline-card-name" style="cursor:pointer;" onclick="openDetailsModal('${esc(l.id)}')">${esc(l.name)}</div>
      <div class="pipeline-card-company">${esc(l.company || '—')}</div>
      <div class="pipeline-card-footer">
        <span class="score-pill ${catClass(l.category)}">${l.score ?? '—'} <small style="font-size:10px;opacity:0.8;">(${prob})</small></span>
        <span class="badge ${badgeClass(l.category)}">${l.category || ''}</span>
      </div>
      <div style="margin-top: 8px; border-top: 1px solid var(--border-color); padding-top: 6px;">
        <select class="form-control" style="font-size: 11px; padding: 2px 4px; height: 26px; width: 100%; cursor: pointer;" onchange="handlePipelineCardStageChange('${esc(l.id)}', this.value)" onclick="event.stopPropagation();">
          ${pipelineOptions(l.pipeline_stage || l.status || 'New')}
        </select>
      </div>
    </div>
  `;
}

async function handlePipelineCardStageChange(leadId, newStage) {
  await quickPipelineChange(leadId, newStage);
  loadPipeline();
}


// ── ANALYTICS VIEW ─────────────────────────────────────────────────────────────
async function loadAnalytics() {
  loadModelInfo();
  try {
    const res = await fetch(`${API}/api/analytics`, { headers: authHeaders() });
    const data = await res.json();
    if (!data.success) throw new Error('Analytics load failed');

    const a = data.analytics;
    const ov = a.overview;

    document.getElementById('ana-total').textContent = fmt(ov.total_leads);
    document.getElementById('ana-won').textContent = fmt(ov.won_leads);
    document.getElementById('ana-lost').textContent = fmt(ov.lost_leads);
    document.getElementById('ana-conv-rate').textContent = `${ov.conversion_rate}%`;

    // Source breakdown
    const srcTbody = document.getElementById('ana-source-tbody');
    srcTbody.innerHTML = (a.source_breakdown || []).map(s => `
      <tr>
        <td>${esc(s.source)}</td>
        <td>${fmt(s.total)}</td>
        <td>${fmt(s.won)}</td>
        <td>${s.conversion_rate}%</td>
        <td><span class="score-pill">${s.avg_score}</span></td>
      </tr>
    `).join('') || '<tr><td colspan="5" class="table-empty">No data</td></tr>';

    // Industry breakdown
    const indTbody = document.getElementById('ana-industry-tbody');
    indTbody.innerHTML = (a.industry_breakdown || []).map(i => `
      <tr>
        <td>${esc(i.industry)}</td>
        <td>${fmt(i.total)}</td>
        <td>${fmt(i.won)}</td>
        <td>${i.conversion_rate}%</td>
        <td><span class="score-pill">${i.avg_score}</span></td>
      </tr>
    `).join('') || '<tr><td colspan="5" class="table-empty">No data</td></tr>';

    // Score distribution bars
    renderScoreDist(a.score_distribution || []);
  } catch (err) {
    showToast('Analytics load failed', 'error');
  }
}

function renderScoreDist(dist) {
  const container = document.getElementById('ana-score-dist');
  if (!dist.length) {
    container.innerHTML = '<div class="table-empty">No score data</div>';
    return;
  }
  const maxCount = Math.max(...dist.map(d => d.count), 1);
  container.innerHTML = dist.map(d => {
    const pct = Math.round((d.count / maxCount) * 100);
    return `
      <div class="dist-bar-row">
        <div class="dist-label">${esc(d.range)}</div>
        <div class="dist-bar-track">
          <div class="dist-bar-fill" style="width: ${pct}%"></div>
        </div>
        <div class="dist-count">${fmt(d.count)}</div>
      </div>
    `;
  }).join('');
}


// ── LEAD DETAILS MODAL ─────────────────────────────────────────────────────────
async function openDetailsModal(leadId) {
  currentActiveLeadId = leadId;
  switchModalTab('tab-overview');
  openModal('lead-details-modal');

  try {
    const res = await fetch(`${API}/api/leads/${leadId}`);
    const data = await res.json();
    if (!data.success) throw new Error('Lead not found');
    const l = data.lead;

    document.getElementById('detail-modal-name').textContent = l.name || '—';
    document.getElementById('detail-modal-subtitle').textContent = `${l.company || ''} · ${l.industry || ''}`;
    document.getElementById('detail-modal-email').textContent = l.email || '—';
    document.getElementById('detail-modal-phone').textContent = l.phone || '—';
    document.getElementById('detail-modal-company').textContent = l.company || '—';
    document.getElementById('detail-modal-job').textContent = l.job_title || '—';
    document.getElementById('detail-modal-industry').textContent = l.industry || '—';
    document.getElementById('detail-modal-source').textContent = l.lead_source || '—';
    document.getElementById('detail-modal-location').textContent = l.location || '—';
    document.getElementById('detail-modal-budget').textContent = l.budget_range || '—';
    document.getElementById('detail-modal-product').textContent = l.product_interest || '—';
    const assignSelect = document.getElementById('detail-modal-assigned-select');
    if (assignSelect) assignSelect.value = l.assigned_to || '';
    document.getElementById('detail-modal-score').textContent = l.score ?? '—';
    const probPct = l.conversion_probability != null ? `${Math.round(l.conversion_probability * 100)}%` : '—';
    document.getElementById('detail-modal-prob').textContent = probPct;
    document.getElementById('detail-modal-category').innerHTML = `<span class="badge ${badgeClass(l.category)}">${l.category || '—'}</span>`;

    const pipeSelect = document.getElementById('detail-modal-pipeline-select');
    if (pipeSelect) pipeSelect.value = l.pipeline_stage || l.status || 'New';

    // Engagement grid
    renderEngagementGrid(l);

    // Notes and Activity
    loadModalNotes(leadId);
    loadModalActivities(leadId);
  } catch (err) {
    showToast('Failed to load lead details', 'error');
    closeDetailsModal();
  }
}

function renderEngagementGrid(l) {
  const grid = document.getElementById('detail-engagement-grid');
  if (!grid) return;
  const fields = [
    ['Website Visits', l.website_visits],
    ['Page Views', l.page_views],
    ['Pricing Page Visits', l.pricing_page_visits],
    ['Email Opens', l.email_opens],
    ['Form Completions', l.form_completions],
    ['Content Downloads', l.content_downloads],
    ['Calls', l.num_calls],
    ['Meetings', l.num_meetings],
    ['Demo Requested', l.demo_requested],
  ];
  grid.innerHTML = fields.map(([label, val]) => `
    <div style="padding: 8px; background: var(--bg-beige); border-radius: var(--radius-sm); font-size: 12px; border: 1px solid var(--border-color);">
      <div style="color: var(--text-muted); margin-bottom: 2px;">${label}</div>
      <div style="font-weight: 700; color: var(--dark-espresso);">${val ?? '—'}</div>
    </div>
  `).join('');
}

async function loadModalNotes(leadId) {
  try {
    const res = await fetch(`${API}/api/leads/${leadId}/notes`);
    const data = await res.json();
    renderNotesList(data.notes || []);
  } catch (e) {
    document.getElementById('modal-notes-list').innerHTML = '<div style="color: var(--text-muted); font-size: 12px;">Could not load notes.</div>';
  }
}

async function loadModalActivities(leadId) {
  try {
    const res = await fetch(`${API}/api/leads/${leadId}/activities`);
    const data = await res.json();
    renderActivitiesList(data.activities || []);
  } catch (e) {
    document.getElementById('modal-activity-list').innerHTML = '<div style="color: var(--text-muted); font-size: 12px;">Could not load activity.</div>';
  }
}

function renderNotesList(notes) {
  const el = document.getElementById('modal-notes-list');
  if (!notes.length) {
    el.innerHTML = '<div style="color: var(--text-muted); font-size: 12px;">No notes recorded yet.</div>';
    return;
  }
  el.innerHTML = notes.map(n => `
    <div class="note-item">
      <div class="note-text">${esc(n.note)}</div>
      <div class="note-meta">${esc(n.author || 'User')} · ${relativeTime(n.created_at)}</div>
    </div>
  `).join('');
}

function renderActivitiesList(activities) {
  const el = document.getElementById('modal-activity-list');
  if (!activities.length) {
    el.innerHTML = '<div style="color: var(--text-muted); font-size: 12px;">No activity recorded yet.</div>';
    return;
  }
  el.innerHTML = activities.map(a => `
    <div class="activity-item">
      <div class="activity-icon">${activityIcon(a.activity_type)}</div>
      <div>
        <div class="activity-desc">${esc(a.description)}</div>
        <div class="activity-time">${relativeTime(a.created_at)}</div>
      </div>
    </div>
  `).join('');
}

function switchModalTab(tabId) {
  document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.modal-tab-content').forEach(c => c.classList.remove('active'));
  const targetContent = document.getElementById(tabId);
  if (targetContent) targetContent.classList.add('active');

  const tabBtns = document.querySelectorAll('.modal-tab');
  const tabMap = { 'tab-overview': 0, 'tab-notes': 1, 'tab-activity': 2 };
  const idx = tabMap[tabId];
  if (idx !== undefined && tabBtns[idx]) tabBtns[idx].classList.add('active');
}

async function handleAddNote() {
  if (!currentActiveLeadId) return;
  const input = document.getElementById('new-note-input');
  const note = input.value.trim();
  if (!note) return;
  try {
    const res = await fetch(`${API}/api/leads/${currentActiveLeadId}/notes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ note, author: 'CRM User' }),
    });
    const data = await res.json();
    if (!data.success) throw new Error('Note save failed');
    input.value = '';
    showToast('Note added!', 'success');
    loadModalNotes(currentActiveLeadId);
    loadModalActivities(currentActiveLeadId);
  } catch (err) {
    showToast('Failed to add note', 'error');
  }
}

async function handleModalPipelineChange() {
  if (!currentActiveLeadId) return;
  const stage = document.getElementById('detail-modal-pipeline-select').value;
  try {
    const res = await fetch(`${API}/api/leads/${currentActiveLeadId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pipeline_stage: stage, status: stage }),
    });
    const data = await res.json();
    if (!data.success) throw new Error('Update failed');
    showToast(`Stage updated to ${stage}`, 'success');
    loadModalActivities(currentActiveLeadId);
  } catch (err) {
    showToast('Failed to update stage', 'error');
  }
}

async function handleDeleteCurrentLead() {
  if (!currentActiveLeadId) return;
  if (!confirm('Delete this lead? This action cannot be undone.')) return;
  await deleteLead(currentActiveLeadId);
  closeDetailsModal();
}

async function deleteLead(leadId, name) {
  if (name && !confirm(`Delete lead "${name}"? This cannot be undone.`)) return;
  try {
    const res = await fetch(`${API}/api/leads/${leadId}`, { method: 'DELETE' });
    const data = await res.json();
    if (!data.success) throw new Error('Delete failed');
    showToast('Lead deleted', 'success');
    if (currentView === 'leads') loadLeads();
    if (currentView === 'pipeline') loadPipeline();
    if (currentView === 'dashboard') loadDashboard();
  } catch (err) {
    showToast('Delete failed', 'error');
  }
}

function closeDetailsModal() {
  closeModal('lead-details-modal');
  currentActiveLeadId = null;
}


// ── EDIT LEAD MODAL ───────────────────────────────────────────────────────────
async function openEditModal(leadId) {
  closeDetailsModal();
  openModal('edit-lead-modal');

  const lead = allLeadsCache.find(l => String(l.id) === String(leadId));
  if (lead) {
    populateEditForm(lead);
  } else {
    try {
      const res = await fetch(`${API}/api/leads/${leadId}`);
      const data = await res.json();
      if (data.success) {
        populateEditForm(data.lead);
        currentActiveLeadId = leadId;
      }
    } catch (e) {
      /* ignore */
    }
  }
}

function populateEditForm(l) {
  currentActiveLeadId = l.id;
  document.getElementById('edit-input-name').value = l.name || '';
  document.getElementById('edit-input-email').value = l.email || '';
  document.getElementById('edit-input-company').value = l.company || '';
  document.getElementById('edit-input-phone').value = l.phone || '';
  document.getElementById('edit-input-job').value = l.job_title || '';
  document.getElementById('edit-input-industry').value = l.industry || '';
  const pipeEl = document.getElementById('edit-input-pipeline');
  if (pipeEl) pipeEl.value = l.pipeline_stage || l.status || 'New';
  document.getElementById('edit-input-assigned').value = l.assigned_to || '';
}

async function handleSaveEditLead(event) {
  event.preventDefault();
  if (!currentActiveLeadId) return;
  const updates = {
    name: document.getElementById('edit-input-name').value,
    email: document.getElementById('edit-input-email').value,
    company: document.getElementById('edit-input-company').value,
    phone: document.getElementById('edit-input-phone').value,
    job_title: document.getElementById('edit-input-job').value,
    industry: document.getElementById('edit-input-industry').value,
    pipeline_stage: document.getElementById('edit-input-pipeline')?.value || 'New',
    assigned_to: document.getElementById('edit-input-assigned').value || null,
  };
  try {
    const res = await fetch(`${API}/api/leads/${currentActiveLeadId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.detail || 'Update failed');
    showToast('Lead updated successfully!', 'success');
    closeEditModal();
    if (currentView === 'leads') loadLeads();
    if (currentView === 'pipeline') loadPipeline();
    if (currentView === 'dashboard') loadDashboard();
  } catch (err) {
    showToast('Update failed: ' + err.message, 'error');
  }
}

function closeEditModal() {
  closeModal('edit-lead-modal');
}


// ── CSV IMPORT MODAL ───────────────────────────────────────────────────────────
function openImportModal() {
  openModal('import-modal');
  csvFileSelected = null;
  document.getElementById('btn-run-import').disabled = true;
  document.getElementById('import-file-name').style.display = 'none';
  document.getElementById('import-result').style.display = 'none';
}

function closeImportModal() {
  closeModal('import-modal');
  csvFileSelected = null;
}

function handleFileSelected(input) {
  const file = input.files[0];
  if (file) setImportFile(file);
}

function setImportFile(file) {
  csvFileSelected = file;
  document.getElementById('import-file-name').style.display = 'block';
  document.getElementById('import-selected-name').textContent = file.name;
  document.getElementById('btn-run-import').disabled = false;
  document.getElementById('import-result').style.display = 'none';
}

async function runImport() {
  if (!csvFileSelected) {
    showToast('No CSV file selected', 'error');
    return;
  }
  const btn = document.getElementById('btn-run-import');
  btn.textContent = 'Importing...';
  btn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('file', csvFileSelected);
    const res = await fetch(`${API}/api/import`, { method: 'POST', body: formData });
    const data = await res.json();

    const resultEl = document.getElementById('import-result');
    resultEl.style.display = 'block';

    if (res.ok && data.success) {
      resultEl.innerHTML = `
        <div style="padding: 12px; background: var(--stage-won-bg); border: 1px solid #C8DDC2; border-radius: var(--radius-sm); font-size: 13px; color: var(--stage-won-text);">
          <strong>✅ Import Successful</strong><br>
          File: ${esc(data.filename)}<br>
          Total Rows: ${data.total_rows} | Valid: ${data.valid_rows} | Invalid: ${data.invalid_rows}<br>
          <strong>Imported: ${data.imported}</strong> | Duplicates skipped: ${data.duplicates} | Failed: ${data.failed}
        </div>
      `;
      showToast(`Imported ${data.imported} leads!`, 'success');
      csvFileSelected = null;
      if (currentView === 'leads') loadLeads();
      if (currentView === 'dashboard') loadDashboard();
    } else {
      resultEl.innerHTML = `<div style="padding: 12px; background: var(--status-hot-bg); border: 1px solid #F3CFBE; border-radius: var(--radius-sm); font-size: 13px; color: var(--status-hot);"><strong>Error:</strong> ${esc(data.detail || 'Import failed')}</div>`;
      showToast('Import failed', 'error');
    }
  } catch (err) {
    showToast('Import error: ' + err.message, 'error');
  } finally {
    btn.textContent = 'Upload & Import';
    btn.disabled = false;
  }
}


// ── MODAL HELPERS ─────────────────────────────────────────────────────────────
function openModal(modalId) {
  const el = document.getElementById(modalId);
  if (el) el.classList.add('active');
}

function closeModal(modalId) {
  const el = document.getElementById(modalId);
  if (el) el.classList.remove('active');
}

function handleOverlayClick(event, modalId) {
  if (event.target.id === modalId) closeModal(modalId);
}


// ── TOAST NOTIFICATIONS ───────────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('toast-show'));
  setTimeout(() => {
    toast.classList.remove('toast-show');
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}


// ── UTILITIES ─────────────────────────────────────────────────────────────────
function esc(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function fmt(n) {
  if (n == null) return '—';
  return Number(n).toLocaleString();
}

function initials(name) {
  if (!name) return '?';
  const parts = String(name).trim().split(/\s+/);
  return parts.slice(0, 2).map(p => p[0]).join('').toUpperCase();
}

function catClass(cat) {
  const c = String(cat || '').toUpperCase();
  if (c === 'HOT') return 'cat-hot';
  if (c === 'WARM') return 'cat-warm';
  return 'cat-cold';
}

function badgeClass(cat) {
  const c = String(cat || '').toUpperCase();
  if (c === 'HOT') return 'badge-hot';
  if (c === 'WARM') return 'badge-warm';
  return 'badge-cold';
}

function activityIcon(type) {
  const icons = {
    lead_created: '🆕',
    status_changed: '🔄',
    pipeline_changed: '📌',
    assigned: '👤',
    score_calculated: '🎯',
    note_added: '📝',
    csv_import: '📂',
  };
  return icons[type] || '📌';
}

function relativeTime(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  if (isNaN(d)) return String(ts).substring(0, 10);
  const diff = Date.now() - d.getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return d.toLocaleDateString();
}


// ── AUTHENTICATION & SESSION MANAGEMENT ──────────────────────────────────────
async function checkCurrentUser() {
  const token = localStorage.getItem('crm_token');
  if (!token) {
    currentUser = null;
    renderUserSession(null);
    showAuthGate();
    return;
  }
  try {
    const res = await fetch(`${API}/api/auth/me`, {
      headers: authHeaders()
    });
    if (res.ok) {
      const data = await res.json();
      if (data.authenticated && data.user) {
        currentUser = data.user;
        renderUserSession(currentUser);
        hideAuthGate();
        handleRouteChange();
        return;
      }
    }
  } catch (e) {}
  localStorage.removeItem('crm_token');
  currentUser = null;
  renderUserSession(null);
  showAuthGate();
}

function renderUserSession(user) {
  const infoEl = document.getElementById('sidebar-user-info');
  const loginBtn = document.getElementById('btn-sidebar-login');
  const logoutBtn = document.getElementById('btn-sidebar-logout');
  if (!infoEl || !loginBtn || !logoutBtn) return;

  if (user) {
    infoEl.style.display = 'flex';
    loginBtn.style.display = 'none';
    logoutBtn.style.display = 'block';
    const avatarEl = document.getElementById('sidebar-user-avatar');
    const nameEl = document.getElementById('sidebar-user-name');
    const roleEl = document.getElementById('sidebar-user-role');
    if (avatarEl) avatarEl.textContent = initials(user.name || user.email);
    if (nameEl) nameEl.textContent = user.name || user.email;
    if (roleEl) roleEl.textContent = `${user.role || 'Member'} · ${user.workspace_id || 'ws-main'}`;
  } else {
    infoEl.style.display = 'none';
    loginBtn.style.display = 'block';
    logoutBtn.style.display = 'none';
  }
}

function openAuthModal(mode = 'login') {
  openModal('auth-modal');
  switchAuthTab(mode);
}

function closeAuthModal() {
  closeModal('auth-modal');
  const errL = document.getElementById('auth-login-error');
  const errS = document.getElementById('auth-signup-error');
  if (errL) { errL.style.display = 'none'; errL.textContent = ''; }
  if (errS) { errS.style.display = 'none'; errS.textContent = ''; }
}

function switchAuthTab(mode) {
  const loginTab = document.getElementById('auth-tab-login');
  const signupTab = document.getElementById('auth-tab-signup');
  const loginForm = document.getElementById('auth-login-form');
  const signupForm = document.getElementById('auth-signup-form');
  const title = document.getElementById('auth-modal-title');

  if (mode === 'signup') {
    if (signupTab) signupTab.classList.add('active');
    if (loginTab) loginTab.classList.remove('active');
    if (signupForm) signupForm.style.display = 'block';
    if (loginForm) loginForm.style.display = 'none';
    if (title) title.textContent = 'Create CRM Account';
  } else {
    if (loginTab) loginTab.classList.add('active');
    if (signupTab) signupTab.classList.remove('active');
    if (loginForm) loginForm.style.display = 'block';
    if (signupForm) signupForm.style.display = 'none';
    if (title) title.textContent = 'Sign In to Lead CRM';
  }
}

function switchGateTab(mode) {
  const loginTab = document.getElementById('gate-tab-login');
  const signupTab = document.getElementById('gate-tab-signup');
  const loginForm = document.getElementById('gate-login-form');
  const signupForm = document.getElementById('gate-signup-form');

  if (mode === 'signup') {
    if (signupTab) signupTab.classList.add('active');
    if (loginTab) loginTab.classList.remove('active');
    if (signupForm) signupForm.style.display = 'block';
    if (loginForm) loginForm.style.display = 'none';
  } else {
    if (loginTab) loginTab.classList.add('active');
    if (signupTab) signupTab.classList.remove('active');
    if (loginForm) loginForm.style.display = 'block';
    if (signupForm) signupForm.style.display = 'none';
  }
}

async function fillDemoAndLogin() {
  const emailInput = document.getElementById('gate-login-email');
  const pwdInput = document.getElementById('gate-login-password');
  if (emailInput) emailInput.value = 'demo@leadcrm.com';
  if (pwdInput) pwdInput.value = 'demo123';
  await submitGateLogin('demo@leadcrm.com', 'demo123');
}

async function handleGateLogin(e) {
  e.preventDefault();
  const email = document.getElementById('gate-login-email').value.trim();
  const password = document.getElementById('gate-login-password').value;
  await submitGateLogin(email, password);
}

async function submitGateLogin(email, password) {
  const errEl = document.getElementById('gate-login-error');
  if (errEl) errEl.style.display = 'none';
  try {
    const res = await fetch(`${API}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.detail || data.error || 'Invalid login credentials');
    }
    const token = data.token || data.access_token;
    localStorage.setItem('crm_token', token);
    currentUser = data.user;
    renderUserSession(currentUser);
    hideAuthGate();
    showToast(`Welcome back, ${currentUser.name || currentUser.email}!`, 'success');
    loadWorkspaceMembers();
    navigateTo('dashboard');
  } catch (err) {
    if (errEl) {
      errEl.textContent = err.message;
      errEl.style.display = 'block';
    }
  }
}

async function handleGateSignup(e) {
  e.preventDefault();
  const name = document.getElementById('gate-signup-name').value.trim();
  const email = document.getElementById('gate-signup-email').value.trim();
  const password = document.getElementById('gate-signup-password').value;
  const role = document.getElementById('gate-signup-role').value;
  const errEl = document.getElementById('gate-signup-error');
  if (errEl) errEl.style.display = 'none';

  try {
    const res = await fetch(`${API}/api/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password, role }),
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.detail || data.error || 'Signup failed');
    }
    const token = data.token || data.access_token;
    localStorage.setItem('crm_token', token);
    currentUser = data.user;
    renderUserSession(currentUser);
    hideAuthGate();
    showToast(`Account created! Welcome, ${currentUser.name}!`, 'success');
    loadWorkspaceMembers();
    navigateTo('dashboard');
  } catch (err) {
    if (errEl) {
      errEl.textContent = err.message;
      errEl.style.display = 'block';
    }
  }
}

async function handleAuthLogin(e) {
  e.preventDefault();
  const email = document.getElementById('login-email').value.trim();
  const password = document.getElementById('login-password').value;
  const errEl = document.getElementById('auth-login-error');
  errEl.style.display = 'none';

  try {
    const res = await fetch(`${API}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.detail || data.error || 'Invalid login credentials');
    }
    const token = data.token || data.access_token;
    localStorage.setItem('crm_token', token);
    currentUser = data.user;
    renderUserSession(currentUser);
    hideAuthGate();
    closeAuthModal();
    showToast(`Welcome back, ${currentUser.name || currentUser.email}!`, 'success');
    loadWorkspaceMembers();
    navigateTo('dashboard');
  } catch (err) {
    errEl.textContent = err.message;
    errEl.style.display = 'block';
  }
}

async function handleAuthSignup(e) {
  e.preventDefault();
  const name = document.getElementById('signup-name').value.trim();
  const email = document.getElementById('signup-email').value.trim();
  const password = document.getElementById('signup-password').value;
  const role = document.getElementById('signup-role').value;
  const errEl = document.getElementById('auth-signup-error');
  errEl.style.display = 'none';

  try {
    const res = await fetch(`${API}/api/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password, role }),
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.detail || data.error || 'Signup failed');
    }
    const token = data.token || data.access_token;
    localStorage.setItem('crm_token', token);
    currentUser = data.user;
    renderUserSession(currentUser);
    hideAuthGate();
    closeAuthModal();
    showToast(`Account created! Welcome, ${currentUser.name}!`, 'success');
    loadWorkspaceMembers();
    navigateTo('dashboard');
  } catch (err) {
    errEl.textContent = err.message;
    errEl.style.display = 'block';
  }
}

async function handleLogout() {
  const token = localStorage.getItem('crm_token');
  if (token) {
    try {
      await fetch(`${API}/api/auth/logout`, {
        method: 'POST',
        headers: authHeaders()
      });
    } catch (e) {}
  }
  localStorage.removeItem('crm_token');
  currentUser = null;
  renderUserSession(null);
  showAuthGate();
  showToast('Signed out successfully', 'info');
}


// ── WORKSPACE MEMBERS ─────────────────────────────────────────────────────────
async function loadWorkspaceMembers() {
  try {
    const res = await fetch(`${API}/api/workspace/members`, {
      headers: authHeaders()
    });
    const data = await res.json();
    if (data.success && Array.isArray(data.members)) {
      workspaceMembersCache = data.members;
      populateMemberDropdowns(workspaceMembersCache);
    }
  } catch (e) {
    console.warn('Could not load workspace members:', e);
  }
}

function populateMemberDropdowns(members) {
  const optionsHtml = '<option value="">Unassigned</option>' + members.map(m => `
    <option value="${esc(m.name || m.email)}">${esc(m.name)} (${esc(m.role || 'Member')})</option>
  `).join('');

  const filterSelect = document.getElementById('leads-assigned-filter');
  if (filterSelect) {
    const prev = filterSelect.value;
    filterSelect.innerHTML = '<option value="All">All Assignees</option>' + members.map(m => `
      <option value="${esc(m.name || m.email)}">${esc(m.name)}</option>
    `).join('');
    if (prev) filterSelect.value = prev;
  }

  const editSelect = document.getElementById('edit-input-assigned');
  if (editSelect) {
    const prev = editSelect.value;
    editSelect.innerHTML = optionsHtml;
    if (prev) editSelect.value = prev;
  }

  const detailSelect = document.getElementById('detail-modal-assigned-select');
  if (detailSelect) {
    const prev = detailSelect.value;
    detailSelect.innerHTML = optionsHtml;
    if (prev) detailSelect.value = prev;
  }
}

async function handleModalAssignChange() {
  if (!currentActiveLeadId) return;
  const select = document.getElementById('detail-modal-assigned-select');
  const assignee = select.value || null;
  try {
    const res = await fetch(`${API}/api/leads/${currentActiveLeadId}`, {
      method: 'PUT',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ assigned_to: assignee }),
    });
    const data = await res.json();
    if (!data.success) throw new Error('Failed to update assignment');
    showToast(`Lead assigned to ${assignee || 'Unassigned'}`, 'success');
    loadModalActivities(currentActiveLeadId);
  } catch (e) {
    showToast('Assignment update failed', 'error');
  }
}


// ── SMART ALERTS WIDGET ───────────────────────────────────────────────────────
async function loadAlerts() {
  const grid = document.getElementById('dash-alerts-grid');
  const countBadge = document.getElementById('dash-alerts-count');
  if (!grid) return;

  try {
    const res = await fetch(`${API}/api/alerts`, { headers: authHeaders() });
    const data = await res.json();
    if (!data.success) throw new Error('Alerts fetch failed');

    const alerts = data.alerts || [];
    if (countBadge) countBadge.textContent = `${alerts.length} Active`;

    if (alerts.length === 0) {
      grid.innerHTML = '<div class="table-empty" style="padding: 16px;">✨ No urgent alerts right now. All high-priority leads are handled!</div>';
      return;
    }

    grid.innerHTML = alerts.slice(0, 4).map(a => {
      const sevClass = a.severity === 'high' ? 'alert-severity-high' : (a.severity === 'medium' ? 'alert-severity-medium' : 'alert-severity-info');
      const icon = a.type === 'HIGH_VALUE_LEAD' ? '🔥' : (a.type === 'FOLLOW_UP_REQUIRED' ? '⏰' : '⚡');
      return `
        <div class="alert-card">
          <div class="alert-card-header">
            <div class="alert-card-title">
              <span>${icon}</span>
              <span>${esc(a.title)}</span>
            </div>
            <span class="alert-severity-badge ${sevClass}">${esc(a.severity)}</span>
          </div>
          <div class="alert-card-body">${esc(a.message)}</div>
          <div class="alert-card-footer">
            <span class="alert-time">${relativeTime(a.created_at)}</span>
            ${a.lead_id ? `<button class="btn btn-secondary btn-sm" onclick="openDetailsModal('${esc(a.lead_id)}')">${esc(a.action_label || 'View Lead')} →</button>` : ''}
          </div>
        </div>
      `;
    }).join('');
  } catch (e) {
    grid.innerHTML = '<div class="table-empty" style="color: var(--text-muted);">Could not load alerts.</div>';
  }
}


// ── CSV EXPORT ────────────────────────────────────────────────────────────────
async function handleExportCSV() {
  const catFilter = document.getElementById('leads-category-filter')?.value || 'All';
  const srcFilter = document.getElementById('leads-source-filter')?.value || 'All';
  const indFilter = document.getElementById('leads-industry-filter')?.value || 'All';
  const assignFilter = document.getElementById('leads-assigned-filter')?.value || 'All';
  const search = document.getElementById('leads-search-input')?.value || '';

  const params = new URLSearchParams();
  if (catFilter !== 'All') params.set('category', catFilter);
  if (srcFilter !== 'All') params.set('lead_source', srcFilter);
  if (indFilter !== 'All') params.set('industry', indFilter);
  if (assignFilter !== 'All') params.set('assigned_to', assignFilter);
  if (search) params.set('search', search);

  const q = params.toString() ? `?${params.toString()}` : '';
  const url = `${API}/api/export${q}`;
  showToast('Preparing CSV download...', 'info');

  try {
    const res = await fetch(url, { headers: authHeaders() });
    if (!res.ok) throw new Error('Export request failed');
    const blob = await res.blob();
    const blobUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = blobUrl;
    a.download = `leads_export_${catFilter.toLowerCase()}_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(blobUrl);
    showToast('CSV export downloaded successfully!', 'success');
  } catch (err) {
    showToast('Export failed: ' + err.message, 'error');
  }
}


// ── MODEL GOVERNANCE & RETRAINING ─────────────────────────────────────────────
async function loadModelInfo() {
  try {
    const res = await fetch(`${API}/api/model-info`, { headers: authHeaders() });
    const data = await res.json();
    if (data.success || data.model_type) {
      const typeEl = document.getElementById('model-val-type');
      const verEl = document.getElementById('model-val-version');
      const rocEl = document.getElementById('model-val-roc');
      if (typeEl) typeEl.textContent = data.model_type || 'XGBoost';
      if (verEl) verEl.textContent = data.active_version || 'v1.0.0';
      if (rocEl) rocEl.textContent = data.current_model_roc_auc != null ? data.current_model_roc_auc.toFixed(4) : '0.7932';
    }
  } catch (e) {
    console.warn('Could not load model info:', e);
  }
}

async function handleRetrainEvaluation() {
  const btn = document.getElementById('btn-retrain-model');
  const box = document.getElementById('model-retrain-status-box');
  if (!btn || !box) return;

  btn.disabled = true;
  btn.textContent = '⏳ Evaluating Candidate...';
  box.style.display = 'block';
  box.className = 'model-retrain-result retrain-retained';
  box.innerHTML = '<strong>Running candidate retraining pipeline...</strong><br>Validating resolved leads dataset, tuning XGBoost candidate model, and calculating ROC-AUC against production baseline (0.7932)...';

  try {
    const res = await fetch(`${API}/api/model/retrain`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({}),
    });
    const data = await res.json();

    if (res.ok && data.status === 'success') {
      const candRoc = data.candidate_roc_auc != null ? data.candidate_roc_auc.toFixed(4) : '—';
      const prodRoc = data.current_roc_auc != null ? data.current_roc_auc.toFixed(4) : '0.7932';
      const isPromoted = data.promoted;

      if (isPromoted) {
        box.className = 'model-retrain-result retrain-promoted';
        box.innerHTML = `
          <strong>🎉 Candidate Model Promoted to Production!</strong><br>
          Candidate ROC-AUC: <strong>${candRoc}</strong> | Previous Production ROC-AUC: <strong>${prodRoc}</strong><br>
          <span style="font-size: 12px;">Candidate exceeded production baseline and has been promoted to active version.</span>
        `;
        showToast('New candidate model promoted!', 'success');
      } else {
        box.className = 'model-retrain-result retrain-retained';
        box.innerHTML = `
          <strong>🛡️ Production Baseline Retained</strong><br>
          Candidate ROC-AUC: <strong>${candRoc}</strong> | Production ROC-AUC Baseline: <strong>${prodRoc}</strong><br>
          <span style="font-size: 12px;">${esc(data.reason || 'Candidate model did not exceed production baseline ROC-AUC. Active production model preserved.')}</span>
        `;
        showToast('Production model preserved (ROC-AUC baseline retained)', 'info');
      }
      loadModelInfo();
    } else {
      box.className = 'model-retrain-result retrain-retained';
      box.innerHTML = `<strong>Notice:</strong> ${esc(data.detail || data.reason || 'Retraining could not be completed.')}`;
    }
  } catch (err) {
    box.className = 'model-retrain-result retrain-retained';
    box.innerHTML = `<strong>Error:</strong> ${esc(err.message)}`;
    showToast('Retraining evaluation failed', 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '⚡ Retrain & Evaluate Candidate';
  }
}


// ── CSV UPLOADS VIEW ────────────────────────────────────────────────────────
async function loadCsvHistory() {
  const tbody = document.getElementById('csv-history-tbody');
  if (!tbody) return;
  tbody.innerHTML = '<tr><td colspan="6" class="table-empty">Loading history...</td></tr>';

  try {
    const res = await fetch(`${API}/api/imports/history`, { headers: authHeaders() });
    const data = await res.json();
    if (!data.success) throw new Error('Failed to load history');

    const history = data.history || [];
    if (!history.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="table-empty">No CSV imports yet.</td></tr>';
      return;
    }

    tbody.innerHTML = history.map(h => `
      <tr>
        <td><strong>${esc(h.filename || 'import.csv')}</strong></td>
        <td style="font-size: 12px; color: var(--text-muted);">${new Date(h.created_at).toLocaleString()}</td>
        <td>${h.total_rows || 0}</td>
        <td style="color: var(--status-warm); font-weight: 500;">${h.imported || 0}</td>
        <td style="color: var(--text-muted);">${h.duplicates || 0}</td>
        <td style="color: var(--status-hot);">${h.failed || 0}</td>
      </tr>
    `).join('');
  } catch (err) {
    console.error(err);
    tbody.innerHTML = '<tr><td colspan="6" class="table-empty" style="color:var(--status-hot);">Failed to load upload history.</td></tr>';
  }
}

function handleCsvFileSelect(event) {
  const file = event.target.files[0];
  if (file) setImportFile(file);
}

function setImportFile(file) {
  if (!file.name.toLowerCase().endsWith('.csv')) {
    showToast('Only .csv files are supported', 'error');
    return;
  }
  csvFileSelected = file;
  const infoEl = document.getElementById('csv-selected-file-info');
  if (infoEl) {
    infoEl.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    infoEl.style.display = 'block';
  }
  document.getElementById('csv-upload-result').style.display = 'none';
}

async function submitCsvUpload() {
  if (!csvFileSelected) {
    showToast('Please select a CSV file first', 'error');
    return;
  }

  const btn = document.getElementById('btn-submit-csv-upload');
  const resultDiv = document.getElementById('csv-upload-result');
  btn.disabled = true;
  btn.textContent = 'Uploading...';
  resultDiv.style.display = 'none';

  const formData = new FormData();
  formData.append('file', csvFileSelected);
  
  if (currentUser && currentUser.workspace_id) {
    formData.append('workspace_id', currentUser.workspace_id);
  }

  try {
    const res = await fetch(`${API}/api/import`, {
      method: 'POST',
      headers: authHeaders(),
      body: formData
    });
    
    const contentType = res.headers.get("content-type");
    let data;
    if (contentType && contentType.indexOf("application/json") !== -1) {
      data = await res.json();
    } else {
      const textErr = await res.text();
      throw new Error(textErr || 'Upload failed with non-JSON response');
    }

    if (!data.success) throw new Error(data.detail || data.error || 'Import failed');

    resultDiv.className = 'upload-success';
    resultDiv.style.display = 'block';
    resultDiv.style.backgroundColor = 'rgba(76, 175, 80, 0.1)';
    resultDiv.style.border = '1px solid var(--status-warm)';
    resultDiv.style.color = 'var(--status-warm)';
    resultDiv.innerHTML = `
      <strong>✅ Import Complete</strong><br>
      Successfully imported and scored <strong>${data.imported}</strong> new leads using the XGBoost production model.<br>
      <span style="font-size:12px;">Valid Rows: ${data.valid_rows} | Duplicates skipped: ${data.valid_rows - data.imported} | Errors: ${data.invalid_rows}</span>
    `;

    csvFileSelected = null;
    document.getElementById('csv-upload-input').value = '';
    document.getElementById('csv-selected-file-info').style.display = 'none';
    
    showToast('Leads successfully imported and scored', 'success');
    
    loadCsvHistory();
    allLeadsCache = [];
  } catch (err) {
    console.error(err);
    resultDiv.className = 'upload-error';
    resultDiv.style.display = 'block';
    resultDiv.style.backgroundColor = 'rgba(244, 67, 54, 0.1)';
    resultDiv.style.border = '1px solid var(--status-hot)';
    resultDiv.style.color = 'var(--status-hot)';
    resultDiv.innerHTML = `<strong>❌ Import Failed</strong><br>${esc(err.message)}`;
    showToast('CSV Import Failed', 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Upload & Import';
  }
}
