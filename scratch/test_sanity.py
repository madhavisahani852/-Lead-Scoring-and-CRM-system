import requests

# 1. HTML check
r_html = requests.get('http://127.0.0.1:8000/')
assert '<section id="dashboard-view" class="page-view active">' in r_html.text, "Missing dashboard view"
assert '<section id="leads-view" class="page-view">' in r_html.text, "Missing leads view"
assert '<section id="score-view" class="page-view">' in r_html.text, "Missing score view"
assert '<section id="pipeline-view" class="page-view">' in r_html.text, "Missing pipeline view"
assert '<section id="analytics-view" class="page-view">' in r_html.text, "Missing analytics view"
assert 'id="lead-details-modal" class="modal-overlay"' in r_html.text, "Missing lead details modal"
assert 'id="edit-lead-modal" class="modal-overlay"' in r_html.text, "Missing edit lead modal"
assert 'id="import-modal" class="modal-overlay"' in r_html.text, "Missing import modal"
print('1. HTML structure: ALL 5 VIEWS AND 3 MODALS VERIFIED')

# 2. CSS check
r_css = requests.get('http://127.0.0.1:8000/static/styles.css')
assert '--bg-main: #FAF7F2;' in r_css.text
assert '--primary-brown: #6F4E37;' in r_css.text
assert '--dark-espresso: #3E2A1F;' in r_css.text
assert '--caramel: #B97850;' in r_css.text
assert '--terracotta: #A95C3B;' in r_css.text
assert '--border-color: #E5DCD0;' in r_css.text
assert '.page-view {\n  display: none !important;\n}' in r_css.text
assert '.page-view.active {\n  display: block !important;\n}' in r_css.text
assert '.modal-overlay {\n  display: none !important;' in r_css.text
assert '.modal-overlay.active {\n  display: flex !important;\n}' in r_css.text
print('2. CSS styling: EXACT BROWN/CREAM PALETTE & STRICT VIEW SWITCHING VERIFIED')

# 3. JS check
r_js = requests.get('http://127.0.0.1:8000/static/app.js')
assert 'getRouteFromHash' in r_js.text
assert 'renderDashPipelineStrip' in r_js.text
print('3. JavaScript router & functions VERIFIED')

# 4. API check
d = requests.get('http://127.0.0.1:8000/api/dashboard').json()
assert d['success'] is True
print(f'4. API Dashboard: {d["dashboard"]["total_leads"]} leads loaded')

print('ALL SANITY CHECKS PASSED!')
