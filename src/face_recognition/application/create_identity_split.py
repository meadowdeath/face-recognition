"""Pure logical identity/session partitioning. Never fit or select anything here."""

from face_recognition.domain.interfaces.identity_split import FeatureMetadataSource
from face_recognition.domain.models.identity_split import (
    IdentityPartition, IdentitySplitManifest, IdentitySplitSpec, PartitionCounts, SessionPartition, SplitSampleReference,
)
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION


class IdentitySplitError(ValueError):
    """Configuration/integrity diagnostics without biometric values."""


class IdentitySplitPlanner:
    def plan(self, source: FeatureMetadataSource, specification: IdentitySplitSpec) -> IdentitySplitManifest:
        if source.representation != BASELINE_REPRESENTATION:
            raise IdentitySplitError("Identity splitting requires all_xyz_v1 metadata")
        records = tuple(source.iter_metadata())  # Metadata only, never vectors.
        if not records or len(records) != source.expected_sample_count:
            raise IdentitySplitError("Empty or incomplete source sample metadata")
        records = tuple(sorted(records, key=lambda record: (*record.provenance.condition_key, record.provenance.sample_index)))
        identities = {record.provenance.person_id for record in records}
        known, unknown = set(specification.known_ids), set(specification.unknown_ids)
        missing, unassigned = (known | unknown) - identities, identities - (known | unknown)
        if missing:
            raise IdentitySplitError("Supplied identities not found: " + ", ".join(sorted(missing)))
        if unassigned:
            raise IdentitySplitError("Dataset identities without explicit roles: " + ", ".join(sorted(unassigned)))
        if len({record.feature_count for record in records}) != 1:
            raise IdentitySplitError("Source contains inconsistent feature dimensions")
        available = {identity: {record.provenance.session_id for record in records
                                if record.provenance.person_id == identity} for identity in identities}
        requested = set(specification.train_sessions) | set(specification.known_test_sessions)
        for identity in sorted(known):
            missing_sessions, extra_sessions = requested - available[identity], available[identity] - requested
            if missing_sessions:
                raise IdentitySplitError(f"Missing configured sessions for {identity}: " + ", ".join(sorted(missing_sessions)))
            if extra_sessions:
                raise IdentitySplitError(f"Unassigned known sessions for {identity}: " + ", ".join(sorted(extra_sessions)))

        assignments = []
        lookup = {}
        for identity in sorted(available):
            for session in sorted(available[identity]):
                if identity in unknown:
                    partition = IdentityPartition.UNKNOWN_POOL
                elif session in specification.train_sessions:
                    partition = IdentityPartition.KNOWN_TRAIN_POOL
                else:
                    partition = IdentityPartition.KNOWN_TEST
                lookup[identity, session] = partition
                assignments.append(SessionPartition(identity, session, partition))

        samples = []
        seen = set()
        next_index = {}
        for record in records:
            provenance = record.provenance
            condition_key = (provenance.person_id, provenance.session_id, provenance.condition)
            key = (*condition_key, provenance.sample_index)
            if key in seen or provenance.sample_index != next_index.get(condition_key, 0):
                raise IdentitySplitError("Duplicate or non-contiguous source sample keys")
            seen.add(key)
            next_index[condition_key] = provenance.sample_index + 1
            samples.append(SplitSampleReference(
                provenance.person_id, provenance.session_id, provenance.condition, provenance.sample_index,
                lookup[provenance.person_id, provenance.session_id],
                BASELINE_REPRESENTATION + "/" + provenance.source_relative_path,
            ))
        counts = []
        for partition in IdentityPartition:
            selected = [sample for sample in samples if sample.partition == partition]
            counts.append(PartitionCounts(partition, len({sample.person_id for sample in selected}),
                                          len({(sample.person_id, sample.session_id) for sample in selected}),
                                          len({sample.source_relative_path for sample in selected}), len(selected)))
        if sum(count.samples for count in counts) != len(records):
            raise IdentitySplitError("Source samples are not accounted for exactly once")
        return IdentitySplitManifest(specification, len(known), len(unknown), tuple(assignments), tuple(counts),
                                     len(samples), tuple(samples))
