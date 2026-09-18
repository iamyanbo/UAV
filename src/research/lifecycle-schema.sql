-- Version 14: one durable lifecycle for discovery, validation and observation.
CREATE TABLE IF NOT EXISTS investigations (
  investigation_id TEXT PRIMARY KEY, direction_id TEXT NOT NULL REFERENCES directions(direction_id),
  run_id TEXT REFERENCES runs(run_id), revises_id TEXT REFERENCES investigations(investigation_id),
  body_md TEXT NOT NULL, body_hash TEXT NOT NULL, created_at TEXT NOT NULL,
  UNIQUE(direction_id,body_hash)
);
CREATE TABLE IF NOT EXISTS investigation_plans (
  investigation_id TEXT PRIMARY KEY REFERENCES investigations(investigation_id),
  direction_id TEXT NOT NULL REFERENCES directions(direction_id),
  state TEXT NOT NULL CHECK(state IN ('active','waiting','closed','dispatched')),
  review_after TEXT, body_md TEXT NOT NULL, body_hash TEXT NOT NULL,
  task_id TEXT REFERENCES tasks(task_id), updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_frames (
  investigation_id TEXT PRIMARY KEY REFERENCES investigations(investigation_id),
  direction_id TEXT NOT NULL REFERENCES directions(direction_id),
  lane TEXT NOT NULL CHECK(lane IN ('discovery','validation','observation')),
  priority INTEGER NOT NULL CHECK(priority BETWEEN 1 AND 5),
  independent INTEGER NOT NULL CHECK(independent IN (0,1)),
  metadata_json TEXT NOT NULL, body_md TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_coverage (
  direction_id TEXT NOT NULL REFERENCES directions(direction_id), topic TEXT NOT NULL,
  investigation_id TEXT NOT NULL REFERENCES investigations(investigation_id),
  PRIMARY KEY(direction_id,topic)
);
CREATE TABLE IF NOT EXISTS research_forecasts (
  forecast_id TEXT PRIMARY KEY, direction_id TEXT NOT NULL REFERENCES directions(direction_id),
  investigation_id TEXT NOT NULL REFERENCES investigations(investigation_id),
  probability REAL NOT NULL CHECK(probability BETWEEN 0 AND 1),
  baseline_probability REAL NOT NULL CHECK(baseline_probability BETWEEN 0 AND 1),
  resolve_after TEXT NOT NULL, target TEXT NOT NULL, resolution_rule TEXT NOT NULL,
  body_md TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(direction_id,body_md)
);
CREATE TABLE IF NOT EXISTS forecast_resolutions (
  forecast_id TEXT PRIMARY KEY REFERENCES research_forecasts(forecast_id),
  outcome INTEGER NOT NULL CHECK(outcome IN (0,1)), source_id TEXT NOT NULL REFERENCES sources(source_id),
  observed_at TEXT NOT NULL, body_md TEXT NOT NULL, recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS adaptation_policies (
  policy_id TEXT PRIMARY KEY, direction_id TEXT NOT NULL REFERENCES directions(direction_id),
  checkpoint_id TEXT NOT NULL REFERENCES program_checkpoints(checkpoint_id),
  revision TEXT NOT NULL, policy_json TEXT NOT NULL, body_md TEXT NOT NULL, created_at TEXT NOT NULL,
  UNIQUE(direction_id,checkpoint_id)
);
CREATE TABLE IF NOT EXISTS adaptation_reviews (
  policy_id TEXT NOT NULL REFERENCES adaptation_policies(policy_id),
  through_session TEXT NOT NULL, status TEXT NOT NULL, evidence_json TEXT NOT NULL,
  investigation_id TEXT REFERENCES investigations(investigation_id), created_at TEXT NOT NULL,
  PRIMARY KEY(policy_id,through_session)
);
CREATE INDEX IF NOT EXISTS research_frames_direction ON research_frames(direction_id,lane,priority);
CREATE INDEX IF NOT EXISTS research_forecasts_due ON research_forecasts(direction_id,resolve_after);
CREATE TRIGGER IF NOT EXISTS immutable_forecast_update BEFORE UPDATE ON research_forecasts BEGIN
  SELECT RAISE(ABORT,'forecasts are append-only; register a new forecast'); END;
CREATE TRIGGER IF NOT EXISTS immutable_forecast_delete BEFORE DELETE ON research_forecasts BEGIN
  SELECT RAISE(ABORT,'forecasts are append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_resolution_update BEFORE UPDATE ON forecast_resolutions BEGIN
  SELECT RAISE(ABORT,'forecast resolutions are append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_resolution_delete BEFORE DELETE ON forecast_resolutions BEGIN
  SELECT RAISE(ABORT,'forecast resolutions are append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_adaptation_policy_update BEFORE UPDATE ON adaptation_policies BEGIN
  SELECT RAISE(ABORT,'adaptation policies are frozen per checkpoint'); END;
CREATE TRIGGER IF NOT EXISTS immutable_adaptation_policy_delete BEFORE DELETE ON adaptation_policies BEGIN
  SELECT RAISE(ABORT,'adaptation policies are append-only'); END;
