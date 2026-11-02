"""Synthetic geometry and paired populations only; no external biometric data."""

from dataclasses import replace
from hashlib import sha256

import numpy as np

from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.processed_sample import POSE_REPRESENTATION
from tests.v3_novelty_fakes import novelty_data


def face_geometry():
    rng = np.random.default_rng(732)
    points = rng.normal(size=(478, 3)) * (.15, .2, .08) + (.5, .4, .01)
    points[[33, 133]] = ((.35, .35, .01), (.39, .35, .01))
    points[[362, 263]] = ((.61, .35, .01), (.65, .35, .01))
    points[[10, 109, 338]] = ((.5, .15, .02), (.43, .16, .02), (.57, .16, .02))
    points[[152, 148, 377]] = ((.5, .72, .03), (.45, .70, .02), (.55, .70, .02))
    return points


def encoded_face(points, width=640, height=360):
    return FaceLandmarks(tuple(Landmark(float(x), float(y * width / height), float(z)) for x, y, z in points))


def rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x": return np.array(((1, 0, 0), (0, c, -s), (0, s, c)))
    if axis == "y": return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))
    if axis == "z": return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    raise ValueError


def paired_pose_data(folder):
    baseline, processed, specification = novelty_data(folder)
    values = baseline.X.copy()
    values[:, 100:] *= .9  # A candidate population with exactly paired source metadata.
    candidate = replace(baseline, X=values, representation=POSE_REPRESENTATION,
                        payload_sha256=sha256(values.tobytes()).hexdigest(), source_stamps=())
    catalog = replace(baseline.protocol.catalog, representation=POSE_REPRESENTATION,
        processing_manifest_sha256="b" * 64,
        condition_files=tuple(replace(r, source_relative_path=r.source_relative_path.replace("all_xyz_v1/", POSE_REPRESENTATION + "/", 1))
                              for r in baseline.protocol.catalog.condition_files))
    return baseline, candidate, catalog, processed, specification
