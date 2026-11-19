-- =====================================================================
-- SmartLane (Vision Vanguard) - PostgreSQL schema
-- Laboratory #9: Software Design Modeling
-- Target: PostgreSQL 18 (uses built-in gen_random_uuid()), pgAdmin 4
--
-- How to run: create/select an empty database, open Query Tool,
-- paste or open this file, then press F5 (Execute).
-- =====================================================================

-- ---------------------------------------------------------------------
-- OPTIONAL clean start (uncomment ONLY to wipe and rebuild everything)
-- ---------------------------------------------------------------------
-- DROP TABLE IF EXISTS report_exports, audit_logs, system_settings,
--     override_logs, emergency_events, alert_logs, signal_logs,
--     pairing_rules, detection_events, cameras, vehicle_classes,
--     road_ends, users CASCADE;
-- DROP TYPE IF EXISTS user_role, user_status, camera_status, signal_mode,
--     override_action, alert_status, setting_value_type, audit_action CASCADE;

BEGIN;

-- =====================================================================
-- 1. ENUM TYPES
-- =====================================================================
CREATE TYPE user_role          AS ENUM ('ADMINISTRATOR', 'ENFORCER');
CREATE TYPE user_status        AS ENUM ('ACTIVE', 'DISABLED');
CREATE TYPE camera_status      AS ENUM ('ONLINE', 'OFFLINE', 'FAULT');
CREATE TYPE signal_mode        AS ENUM ('AUTO', 'MANUAL', 'EMERGENCY', 'FALLBACK');
CREATE TYPE override_action    AS ENUM ('FORCE_A', 'FORCE_B', 'ALL_RED', 'EMERGENCY_CLEAR');
CREATE TYPE alert_status       AS ENUM ('SENT', 'FAILED');
CREATE TYPE setting_value_type AS ENUM ('INT', 'FLOAT', 'BOOL', 'TEXT');
CREATE TYPE audit_action       AS ENUM ('LOGIN', 'LOGOUT', 'AUTO_LOGOUT', 'AUTH_LOCK',
                                        'EDIT_SETTINGS', 'EXPORT', 'OVERRIDE');

-- =====================================================================
-- 2. TABLES
-- =====================================================================

-- Table 1. Users ------------------------------------------------------
CREATE TABLE users (
    user_id        UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name      TEXT         NOT NULL,
    username       TEXT         NOT NULL UNIQUE,
    email          TEXT         UNIQUE,                 -- nullable
    phone          VARCHAR(12)  NOT NULL UNIQUE,
    password_hash  TEXT         NOT NULL,               -- hashed, never plain text
    role           user_role    NOT NULL,
    status         user_status  NOT NULL,
    last_login     TIMESTAMP,                           -- NULL until first login
    created_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP                            -- NULL if never updated
);
COMMENT ON TABLE users IS 'Accounts that can log in to the web system.';

-- Table 3. RoadEnds (created before Cameras, which references it) -----
CREATE TABLE road_ends (
    road_end_id  INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    label        TEXT    NOT NULL
);
COMMENT ON TABLE road_ends IS 'Lookup table of the two ends of the narrow road where cameras and traffic lights are placed.';

-- Table 2. Cameras ----------------------------------------------------
CREATE TABLE cameras (
    camera_id      INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_name    TEXT          NOT NULL,
    road_end_id    INTEGER       NOT NULL
                   REFERENCES road_ends (road_end_id),
    source         TEXT          NOT NULL,              -- USB index or stream URL
    status         camera_status NOT NULL,
    last_frame_at  TIMESTAMP                            -- for fault check
);
COMMENT ON TABLE cameras IS 'All registered cameras in the system network (dedicated table for future scalability).';

-- Table 4. VehicleClasses ---------------------------------------------
CREATE TABLE vehicle_classes (
    class_id     INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    class_code   TEXT    NOT NULL UNIQUE,               -- 2W, 4W or EMG
    class_label  TEXT    NOT NULL,
    priority     INTEGER NOT NULL                       -- higher number wins right-of-way
);
COMMENT ON TABLE vehicle_classes IS 'Vehicle classes the system detects and their right-of-way priority.';

-- Table 5. DetectionEvents --------------------------------------------
CREATE TABLE detection_events (
    event_id     INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id    INTEGER       NOT NULL REFERENCES cameras (camera_id),
    class_id     INTEGER       NOT NULL REFERENCES vehicle_classes (class_id),
    track_id     INTEGER,                               -- tracker ID, counts a vehicle once
    confidence   NUMERIC(4,3)  NOT NULL
                 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    detected_at  TIMESTAMP     NOT NULL,
    flagged_by   UUID          REFERENCES users (user_id),  -- NULL if not flagged
    flagged_at   TIMESTAMP,                                 -- NULL if not flagged
    flag_notes   TEXT                                       -- NULL if not flagged
);
COMMENT ON TABLE detection_events IS 'Every vehicle classified by YOLO inside a detection zone.';

-- Table 6. PairingRules -----------------------------------------------
CREATE TABLE pairing_rules (
    rule_id     INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    class_a_id  INTEGER   NOT NULL REFERENCES vehicle_classes (class_id),
    class_b_id  INTEGER   NOT NULL REFERENCES vehicle_classes (class_id),
    allowed     BOOLEAN   NOT NULL,                     -- TRUE = may pass together
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by  UUID      REFERENCES users (user_id),
    updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_pairing_rules_pair UNIQUE (class_a_id, class_b_id)
);
COMMENT ON TABLE pairing_rules IS 'Which vehicle classes may pass the narrow road together.';

-- Table 7. SignalLogs -------------------------------------------------
CREATE TABLE signal_logs (
    log_id        INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    signal_a      TEXT        NOT NULL CHECK (signal_a IN ('RED', 'GREEN')),
    signal_b      TEXT        NOT NULL CHECK (signal_b IN ('RED', 'GREEN')),
    mode          signal_mode NOT NULL,
    reason        TEXT,
    queue_a       INTEGER     NOT NULL CHECK (queue_a >= 0),
    queue_b       INTEGER     NOT NULL CHECK (queue_b >= 0),
    started_at    TIMESTAMP   NOT NULL,
    ended_at      TIMESTAMP,                            -- NULL while phase is active
    triggered_by  UUID        REFERENCES users (user_id),  -- set only if manual
    CONSTRAINT ck_signal_logs_times CHECK (ended_at IS NULL OR ended_at >= started_at)
);
COMMENT ON TABLE signal_logs IS 'Each signal phase shown on the traffic light, with its mode and queue counts.';

-- Table 10. AlertLogs (created before EmergencyEvents, which references it)
CREATE TABLE alert_logs (
    alert_id    INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    alert_type  TEXT         NOT NULL,                  -- EMERGENCY, FAULT, or custom type
    recipients  TEXT         NOT NULL,                  -- chat IDs of barangay / TMO
    message     TEXT         NOT NULL,
    status      alert_status NOT NULL,
    sent_at     TIMESTAMP    NOT NULL
);
COMMENT ON TABLE alert_logs IS 'Alerts the system sent to the barangay or TMO.';

-- Table 8. EmergencyEvents --------------------------------------------
CREATE TABLE emergency_events (
    emergency_id  INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id      INTEGER NOT NULL REFERENCES detection_events (event_id),
    alert_id      INTEGER REFERENCES alert_logs (alert_id),
    signal_id     INTEGER NOT NULL REFERENCES signal_logs (log_id),
    hold_seconds  INTEGER CHECK (hold_seconds IS NULL OR hold_seconds >= 0)
);
COMMENT ON TABLE emergency_events IS 'Emergency vehicles that triggered priority mode and how long the opposing end was held red.';

-- Table 9. OverrideLogs -----------------------------------------------
CREATE TABLE override_logs (
    override_id    INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id        UUID            NOT NULL REFERENCES users (user_id),
    action         override_action NOT NULL,
    signal_id      INTEGER         NOT NULL REFERENCES signal_logs (log_id),
    auto_reverted  BOOLEAN         NOT NULL DEFAULT FALSE   -- TRUE if returned to AUTO after timeout
);
COMMENT ON TABLE override_logs IS 'Manual overrides made by users from the Live control page.';

-- Table 11. SystemSettings --------------------------------------------
CREATE TABLE system_settings (
    setting_key    TEXT               PRIMARY KEY,
    setting_value  TEXT               NOT NULL,
    value_type     setting_value_type NOT NULL,
    updated_by     UUID               REFERENCES users (user_id),
    updated_at     TIMESTAMP          NOT NULL DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE system_settings IS 'Key-value store of signal timing and other adjustable system settings.';

-- Table 12. AuditLogs -------------------------------------------------
CREATE TABLE audit_logs (
    audit_id    INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id     UUID         REFERENCES users (user_id),    -- NULL for failed logins
    action      audit_action NOT NULL,
    details     TEXT,
    ip_address  TEXT,
    logged_at   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE audit_logs IS 'Records who did what in the web system.';

-- Table 13. ReportExports ---------------------------------------------
CREATE TABLE report_exports (
    export_id    INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id      UUID      NOT NULL REFERENCES users (user_id),
    report_type  TEXT      NOT NULL,   -- TRAFFIC_SUMMARY, EMERGENCY, MISCLASSIFIED, ...
    date_from    DATE      NOT NULL,
    date_to      DATE      NOT NULL,
    file_format  TEXT      NOT NULL CHECK (file_format IN ('PDF', 'CSV')),
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_report_exports_dates CHECK (date_to >= date_from)
);
COMMENT ON TABLE report_exports IS 'Reports exported from the Reports module.';

-- =====================================================================
-- 3. INDEXES (foreign keys and common dashboard / report filters)
-- =====================================================================
CREATE INDEX idx_cameras_road_end        ON cameras (road_end_id);
CREATE INDEX idx_detection_camera        ON detection_events (camera_id);
CREATE INDEX idx_detection_class         ON detection_events (class_id);
CREATE INDEX idx_detection_detected_at   ON detection_events (detected_at);
CREATE INDEX idx_detection_flagged_by    ON detection_events (flagged_by);
CREATE INDEX idx_pairing_class_b         ON pairing_rules (class_b_id);
CREATE INDEX idx_pairing_updated_by      ON pairing_rules (updated_by);
CREATE INDEX idx_signal_started_at       ON signal_logs (started_at);
CREATE INDEX idx_signal_triggered_by     ON signal_logs (triggered_by);
CREATE INDEX idx_emergency_event         ON emergency_events (event_id);
CREATE INDEX idx_emergency_alert         ON emergency_events (alert_id);
CREATE INDEX idx_emergency_signal        ON emergency_events (signal_id);
CREATE INDEX idx_override_user           ON override_logs (user_id);
CREATE INDEX idx_override_signal         ON override_logs (signal_id);
CREATE INDEX idx_alert_sent_at           ON alert_logs (sent_at);
CREATE INDEX idx_settings_updated_by     ON system_settings (updated_by);
CREATE INDEX idx_audit_user              ON audit_logs (user_id);
CREATE INDEX idx_audit_logged_at         ON audit_logs (logged_at);
CREATE INDEX idx_exports_user            ON report_exports (user_id);

-- =====================================================================
-- 4. STARTER DATA (lookup tables only; safe to delete this section)
-- =====================================================================
INSERT INTO road_ends (label) VALUES
    ('North End'),
    ('South End');

INSERT INTO vehicle_classes (class_code, class_label, priority) VALUES
    ('2W',  'Two-wheeler (motorcycle, tricycle)', 1),
    ('4W',  'Four-wheeler (car, van)',            2),
    ('EMG', 'Emergency vehicle',                  3);

INSERT INTO cameras (camera_name, road_end_id, source, status) VALUES
    ('Camera A - North', 1, '0', 'OFFLINE'),
    ('Camera B - South', 2, '1', 'OFFLINE');

-- Pairing rules: only 2W + 2W may pass together
INSERT INTO pairing_rules (class_a_id, class_b_id, allowed)
SELECT a.class_id, b.class_id, (a.class_code = '2W' AND b.class_code = '2W')
FROM (VALUES ('2W','2W'), ('2W','4W'), ('4W','4W')) AS p(ca, cb)
JOIN vehicle_classes a ON a.class_code = p.ca
JOIN vehicle_classes b ON b.class_code = p.cb;

INSERT INTO system_settings (setting_key, setting_value, value_type) VALUES
    ('red_clearance_s',      '5',    'INT'),
    ('min_green_s',          '10',   'INT'),
    ('max_green_s',          '30',   'INT'),
    ('confidence_threshold', '0.90', 'FLOAT'),
    ('emergency_hold_s',     '30',   'INT'),
    ('manual_timeout_s',     '120',  'INT');

COMMIT;

-- Quick check after running:
-- SELECT table_name FROM information_schema.tables
-- WHERE table_schema = 'public' ORDER BY table_name;