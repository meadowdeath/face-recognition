"""Synthetic metadata catalog whose JSONL files must never be decoded."""

from dataclasses import asdict
import json
from pathlib import Path

from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.v3_cv_protocol import V3ExperimentSpecification


def v3_catalog_fixture(directory: Path, known_count=10):
    spec = V3ExperimentSpecification(tuple(f"known_{i:02d}" for i in range(known_count)),
             tuple(f"novel_{i:02d}" for i in range(8)),
             ("capture_a", "capture_b", "capture_c", "capture_d"), ("visit_x", "visit_y"), "synthetic_v3")
    processed = directory / "processed"
    root = processed / spec.representation
    people = {person: spec.known_sessions for person in spec.known_ids}
    people.update({person: spec.unknown_sessions for person in spec.unknown_ids})
    for person, sessions in reversed(tuple(people.items())):
        for session in reversed(sessions):
            folder = root / person / session
            folder.mkdir(parents=True)
            for condition in reversed(CONDITIONS):
                # Deliberately invalid JSON: a catalog operation must not parse it.
                (folder / (condition + ".jsonl")).write_bytes(b"SYNTHETIC_PAYLOAD_DO_NOT_READ\n")
    session_count = sum(map(len, people.values()))
    files = session_count * len(CONDITIONS)
    metadata = {"schema_version": 1, "representation": spec.representation, "completion_state": "completed",
                "participants": len(people), "sessions": session_count, "conditions": files,
                "condition_files": files, "total_samples_processed": files * 3, "rejected_samples": 0, "error_count": 0}
    (root / "processing_manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    specification = directory / "experiment.json"
    specification.write_text(json.dumps(asdict(spec)), encoding="utf-8")
    return processed, specification, spec
