-- ============================================================
-- Supabase Complete Backend Schema
-- Flood Damage Assessment Project
-- ============================================================

-- ------------------------------------------------------------
-- 1. TABLES
-- ------------------------------------------------------------

-- A. USERS TABLE
-- Explantion: Stores extended profile data for users. 
-- It references Supabase's built-in authentication table (auth.users).
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE,
    full_name TEXT,
    organization TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- B. UPLOADED_IMAGES TABLE
-- Explanation: Tracks raw images uploaded by users before they are processed.
-- Stores the path to the image in the Supabase Storage bucket.
CREATE TABLE IF NOT EXISTS public.uploaded_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    storage_path TEXT NOT NULL,
    original_filename TEXT,
    file_size_bytes BIGINT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- C. PREDICTIONS TABLE
-- Explanation: The core table. Stores the high-level AI analysis results 
-- for a specific uploaded image.
CREATE TABLE IF NOT EXISTS public.predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID NOT NULL REFERENCES public.uploaded_images(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    
    is_flood BOOLEAN NOT NULL,
    confidence_score DOUBLE PRECISION NOT NULL,
    flood_percentage DOUBLE PRECISION NOT NULL,
    
    severity TEXT CHECK (severity IN ('Low', 'Medium', 'High', 'Severe')),
    risk_level TEXT,
    
    processed_mask_url TEXT, -- URL to the overlay image
    
    created_at TIMESTAMPTZ DEFAULT now()
);

-- D. DETECTED_OBJECTS TABLE
-- Explanation: A normalized table to store the individual objects detected by YOLOv8 
-- (e.g., a specific building, a specific car) within a single prediction.
CREATE TABLE IF NOT EXISTS public.detected_objects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID NOT NULL REFERENCES public.predictions(id) ON DELETE CASCADE,
    
    object_class TEXT NOT NULL CHECK (object_class IN ('Buildings', 'Roads', 'Vehicles', 'Infrastructure')),
    confidence DOUBLE PRECISION NOT NULL,
    
    -- Bounding box coordinates [x1, y1, x2, y2]
    bbox_x1 INT NOT NULL,
    bbox_y1 INT NOT NULL,
    bbox_x2 INT NOT NULL,
    bbox_y2 INT NOT NULL,
    
    is_submerged BOOLEAN DEFAULT false,
    
    created_at TIMESTAMPTZ DEFAULT now()
);

-- E. REPORTS TABLE
-- Explanation: Stores links to generated PDF reports summarizing the damage assessment.
CREATE TABLE IF NOT EXISTS public.reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID NOT NULL REFERENCES public.predictions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    
    pdf_storage_path TEXT NOT NULL,
    generated_at TIMESTAMPTZ DEFAULT now()
);

-- F. HISTORY TABLE (Audit Log)
-- Explanation: An append-only audit log that tracks user activities for accountability.
CREATE TABLE IF NOT EXISTS public.history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    
    action_type TEXT NOT NULL CHECK (action_type IN ('UPLOAD', 'ANALYZE', 'DOWNLOAD_REPORT', 'DELETE_RECORD')),
    details JSONB, -- Flexible JSON for storing context (e.g., which image ID)
    
    timestamp TIMESTAMPTZ DEFAULT now()
);


-- ------------------------------------------------------------
-- 2. ROW LEVEL SECURITY (RLS) POLICIES
-- ------------------------------------------------------------
-- Explanation: RLS ensures that User A cannot see or delete User B's images, 
-- predictions, or reports. We restrict access strictly to `auth.uid() = user_id`.

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.uploaded_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.detected_objects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.history ENABLE ROW LEVEL SECURITY;

-- Users Policy
CREATE POLICY "Users can view their own profile" 
ON public.users FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile" 
ON public.users FOR UPDATE USING (auth.uid() = id);

-- Uploaded Images Policy
CREATE POLICY "Users can manage their own images" 
ON public.uploaded_images FOR ALL USING (auth.uid() = user_id);

-- Predictions Policy
CREATE POLICY "Users can manage their own predictions" 
ON public.predictions FOR ALL USING (auth.uid() = user_id);

-- Detected Objects Policy (Inherits ownership via prediction_id -> user_id)
CREATE POLICY "Users can view detected objects for their predictions" 
ON public.detected_objects FOR SELECT 
USING (
    prediction_id IN (
        SELECT id FROM public.predictions WHERE user_id = auth.uid()
    )
);
CREATE POLICY "Users can insert detected objects for their predictions" 
ON public.detected_objects FOR INSERT 
WITH CHECK (
    prediction_id IN (
        SELECT id FROM public.predictions WHERE user_id = auth.uid()
    )
);

-- Reports Policy
CREATE POLICY "Users can manage their own reports" 
ON public.reports FOR ALL USING (auth.uid() = user_id);

-- History Policy
CREATE POLICY "Users can view their own history" 
ON public.history FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own history" 
ON public.history FOR INSERT WITH CHECK (auth.uid() = user_id);


-- ------------------------------------------------------------
-- 3. STORAGE BUCKETS
-- ------------------------------------------------------------
-- Explanation: Creates physical storage buckets where the actual JPG/PNG and PDF files live.

-- Create buckets if they don't exist
INSERT INTO storage.buckets (id, name, public) 
VALUES ('images', 'images', false) 
ON CONFLICT (id) DO NOTHING;

INSERT INTO storage.buckets (id, name, public) 
VALUES ('reports', 'reports', false) 
ON CONFLICT (id) DO NOTHING;

-- Enable RLS on storage
CREATE POLICY "Users can manage their own storage images"
ON storage.objects FOR ALL
USING ( bucket_id = 'images' AND auth.uid()::text = (storage.foldername(name))[1] );

CREATE POLICY "Users can manage their own storage reports"
ON storage.objects FOR ALL
USING ( bucket_id = 'reports' AND auth.uid()::text = (storage.foldername(name))[1] );
