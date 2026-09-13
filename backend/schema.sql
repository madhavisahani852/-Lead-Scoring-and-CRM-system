-- Lead CRM Supabase Schema
-- Run this in your Supabase SQL editor to create tables

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

-- 2. Leads Table (primary CRM table)
CREATE TABLE IF NOT EXISTS leads (
    id TEXT PRIMARY KEY,
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
    pipeline_stage TEXT DEFAULT 'new',
    assigned_to TEXT,
    tags TEXT[],
    converted BOOLEAN,
    outcome TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Notes Table
CREATE TABLE IF NOT EXISTS notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id TEXT REFERENCES leads(id) ON DELETE CASCADE,
    note TEXT NOT NULL,
    author TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Lead Activities Table
CREATE TABLE IF NOT EXISTS lead_activities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id TEXT REFERENCES leads(id) ON DELETE CASCADE,
    activity_type TEXT NOT NULL,
    description TEXT NOT NULL,
    actor TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. CSV Imports Log
CREATE TABLE IF NOT EXISTS csv_imports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT,
    total_rows INTEGER,
    imported INTEGER,
    duplicates INTEGER,
    failed INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status);
CREATE INDEX IF NOT EXISTS idx_leads_category ON leads(category);
CREATE INDEX IF NOT EXISTS idx_leads_pipeline_stage ON leads(pipeline_stage);
CREATE INDEX IF NOT EXISTS idx_leads_assigned_to ON leads(assigned_to);
CREATE INDEX IF NOT EXISTS idx_activities_lead_id ON lead_activities(lead_id);
CREATE INDEX IF NOT EXISTS idx_notes_lead_id ON notes(lead_id);
