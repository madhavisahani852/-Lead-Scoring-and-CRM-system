/**
 * frontend/app.js
 * Lead CRM — Single Page Application View Controller
 * Enforces strict single-view visibility via hash routing:
 * #/dashboard | #/leads | #/score | #/pipeline | #/analytics
 */

// ── Application State ──────────────────────────────────────────────────────────
const API = 'https://lead-scoring-and-crm-system.onrender.com';
let currentView = 'dashboard';
let currentActiveLeadId = null;
let allLeadsCache = [];
let currentPage = 1;
const PAGE_LIMIT = 50;
let totalLeadsCount = 0;
let csvFileSelected = null;
let lastScoredResult = null;

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

function handleRouteChange() {
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
  }
}

window.addEventListener('hashchange', handleRouteChange);

// ── Initialization ────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  if (!window.location.hash) {
    window.location.hash = '#/dashboard';
  }
  handleRouteChange();

  // CSV Drag and Drop
  const dropZone = document.getElementById('import-drop-zone');
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
  try {
    const res = await fetch(`${API}/api/dashboard`);
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
  return `
    <div class="pipeline-card">
      <div class="pipeline-card-name" style="cursor:pointer;" onclick="openDetailsModal('${esc(l.id)}')">${esc(l.name)}</div>
      <div class="pipeline-card-company">${esc(l.company || '—')}</div>
      <div class="pipeline-card-footer">
        <span class="score-pill ${catClass(l.category)}">${l.score ?? '—'}</span>
        <span class="badge ${badgeClass(l.category)}">${l.category || ''}</span>
      </div>
    </div>
  `;
}


// ── ANALYTICS VIEW ─────────────────────────────────────────────────────────────
async function loadAnalytics() {
  try {
    const res = await fetch(`${API}/api/analytics`);
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
    document.getElementById('detail-modal-assigned').textContent = l.assigned_to || 'Unassigned';
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
