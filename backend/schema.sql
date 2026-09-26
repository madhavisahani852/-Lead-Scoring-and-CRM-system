-- Lead CRM Supabase Schema
-- Run this in your Supabase SQL editor to create or update tables

-- 1. Pipeline Stages
CREATE TABLE IF NOT EXISTS pipeline_stages (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    position INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO pipeline_stages (id, name, position) VALUES
    ('new', 'New', 1),
    ('contacted', 'Contacted', 2),
    ('qualified', 'Qualified', 3),
    ('demo_scheduled', 'Demo Scheduled', 4),
    ('proposal', 'Proposal', 5),
    ('negotiation', 'Negotiation', 6),
    ('won', 'Won', 7),
    ('lost', 'Lost', 8)
ON CONFLICT (id) DO NOTHING;

-- 2. Workspaces Table
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Workspace Members Table
CREATE TABLE IF NOT EXISTS workspace_members (
    id TEXT PRIMARY KEY,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    email TEXT NOT NULL,
    name TEXT,
    role TEXT DEFAULT 'salesperson',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Leads Table (primary CRM table)
CREATE TABLE IF NOT EXISTS leads (
    id TEXT PRIMARY KEY,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    name TEXT,
    email TEXT,
    phone TEXT,
    company TEXT,
    job_title TEXT,
    industry TEXT,
    company_size FLOAT,
    location TEXT,
    lead_source TEXT,
    campaign TEXT,
    product_interest TEXT,
    budget_range TEXT,
    website_visits FLOAT,
    page_views FLOAT,
    pricing_page_visits FLOAT,
    demo_requested TEXT,
    email_opens FLOAT,
    form_completions FLOAT,
    content_downloads FLOAT,
    previous_interactions FLOAT,
    response_time_hours FLOAT,
    num_calls FLOAT,
    num_meetings FLOAT,
    score INTEGER,
    conversion_probability FLOAT,
    category TEXT,
    priority TEXT,
    status TEXT DEFAULT 'New',
    pipeline_stage TEXT DEFAULT 'New',
    assigned_to TEXT,
    assigned_user_id TEXT,
    tags TEXT[],
    converted BOOLEAN,
    outcome TEXT,
    converted_at TIMESTAMPTZ,
    model_version TEXT DEFAULT 'v1.0-production',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Notes Table
CREATE TABLE IF NOT EXISTS notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id TEXT REFERENCES leads(id) ON DELETE CASCADE,
    note TEXT NOT NULL,
    author TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6. Lead Activities Table
CREATE TABLE IF NOT EXISTS lead_activities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id TEXT REFERENCES leads(id) ON DELETE CASCADE,
    activity_type TEXT NOT NULL,
    description TEXT NOT NULL,
    actor TEXT,
    user_id TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 7. CSV Imports Log
CREATE TABLE IF NOT EXISTS csv_imports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    filename TEXT,
    total_rows INTEGER,
    imported INTEGER,
    duplicates INTEGER,
    failed INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 8. Model Versions Registry
CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY,
    algorithm TEXT NOT NULL,
    training_records INTEGER,
    roc_auc FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1 FLOAT,
    status TEXT DEFAULT 'Production', -- Production, Candidate, Archived
    selection_reason TEXT,
    metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert Default Model Version (v1.0-production)
INSERT INTO model_versions (version, algorithm, training_records, roc_auc, precision, recall, f1, status, selection_reason)
VALUES (
    'v1.0-production',
    'XGBoost Classifier',
    1017,
    0.7932,
    0.7578,
    0.7822,
    0.7698,
    'Production',
    'Best Tuned XGBoost Pipeline trained on cleaned historical lead dataset'
) ON CONFLICT (version) DO NOTHING;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_leads_workspace ON leads(workspace_id);
CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status);
CREATE INDEX IF NOT EXISTS idx_leads_category ON leads(category);
CREATE INDEX IF NOT EXISTS idx_leads_pipeline_stage ON leads(pipeline_stage);
CREATE INDEX IF NOT EXISTS idx_leads_assigned_to ON leads(assigned_to);
CREATE INDEX IF NOT EXISTS idx_activities_lead_id ON lead_activities(lead_id);
CREATE INDEX IF NOT EXISTS idx_notes_lead_id ON notes(lead_id);
CREATE INDEX IF NOT EXISTS idx_members_workspace ON workspace_members(workspace_id);
