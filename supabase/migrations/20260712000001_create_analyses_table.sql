-- ============================================================
-- Supabase Migration: Create analyses table
-- ============================================================

CREATE TABLE IF NOT EXISTS analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),

    -- Input
    image_url TEXT DEFAULT '',
    image_metadata JSONB,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,

    -- Pipeline results
    classification_result JSONB,
    segmentation_result JSONB,
    detection_results JSONB DEFAULT '[]'::jsonb,
    damage_assessment JSONB,

    -- Assessment
    severity_score DOUBLE PRECISION,
    damage_category TEXT
        CHECK (damage_category IS NULL OR damage_category IN ('minor', 'moderate', 'severe', 'critical')),

    -- Result images
    annotated_image_url TEXT DEFAULT '',
    mask_image_url TEXT DEFAULT '',

    -- Error tracking
    error_message TEXT DEFAULT '',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Index for user queries
CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_analyses_updated_at
    BEFORE UPDATE ON analyses
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
