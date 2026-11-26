BEGIN;

ALTER TABLE cameras
    ADD COLUMN IF NOT EXISTS detection_width INTEGER NOT NULL DEFAULT 640,
    ADD COLUMN IF NOT EXISTS detection_height INTEGER NOT NULL DEFAULT 480,
    ADD COLUMN IF NOT EXISTS preprocessing_mode TEXT NOT NULL DEFAULT 'fit',
    ADD COLUMN IF NOT EXISTS roi_x DOUBLE PRECISION NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS roi_y DOUBLE PRECISION NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS roi_width DOUBLE PRECISION NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS roi_height DOUBLE PRECISION NOT NULL DEFAULT 1;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_cameras_detection_dimensions'
          AND conrelid = 'cameras'::regclass
    ) THEN
        ALTER TABLE cameras
            ADD CONSTRAINT ck_cameras_detection_dimensions
            CHECK (detection_width > 0 AND detection_height > 0);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_cameras_preprocessing_mode'
          AND conrelid = 'cameras'::regclass
    ) THEN
        ALTER TABLE cameras
            ADD CONSTRAINT ck_cameras_preprocessing_mode
            CHECK (preprocessing_mode IN ('fit', 'crop'));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_cameras_roi_values'
          AND conrelid = 'cameras'::regclass
    ) THEN
        ALTER TABLE cameras
            ADD CONSTRAINT ck_cameras_roi_values
            CHECK (
                roi_x >= 0 AND roi_x <= 1
                AND roi_y >= 0 AND roi_y <= 1
                AND roi_width > 0 AND roi_width <= 1
                AND roi_height > 0 AND roi_height <= 1
                AND roi_x + roi_width <= 1
                AND roi_y + roi_height <= 1
            );
    END IF;
END
$$;

COMMENT ON COLUMN cameras.detection_width IS
    'Per-camera detection input width; must not exceed the worker global frame cap.';
COMMENT ON COLUMN cameras.detection_height IS
    'Per-camera detection input height; must not exceed the worker global frame cap.';
COMMENT ON COLUMN cameras.preprocessing_mode IS
    'Use fit for the full frame or crop for the configured normalized ROI.';

COMMIT;
