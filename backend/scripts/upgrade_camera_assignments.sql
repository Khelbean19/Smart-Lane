BEGIN;

CREATE TABLE IF NOT EXISTS camera_assignments (
    road_end_id  INTEGER   PRIMARY KEY
                 REFERENCES road_ends (road_end_id) ON DELETE CASCADE,
    camera_id    INTEGER   NOT NULL UNIQUE
                 REFERENCES cameras (camera_id) ON DELETE RESTRICT,
    updated_by   UUID      REFERENCES users (user_id),
    updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE camera_assignments IS
    'Desired camera selection for each live camera slot; camera status separately records runtime health.';

UPDATE cameras
SET status = 'OFFLINE'
WHERE NOT EXISTS (
    SELECT 1
    FROM camera_assignments assignment
    WHERE assignment.camera_id = cameras.camera_id
);

COMMIT;
