from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from face_recognition.infrastructure.persistence.artifact_integrity import artifact_fingerprints, file_sha256


class ArtifactIntegrityTests(unittest.TestCase):
    def test_exact_byte_digest_deterministic_order_and_no_mutation(self):
        with TemporaryDirectory(prefix="synthetic_integrity_") as folder:
            root = Path(folder)
            (root / "b.bin").write_bytes(b"synthetic artifact")
            (root / "a.json").write_bytes(b"{}\n")
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            result = artifact_fingerprints(root, ("b.bin", "a.json"))
            self.assertEqual(result, tuple((name, sha256(before[name]).hexdigest()) for name in sorted(before)))
            self.assertEqual(file_sha256(root / "b.bin"), sha256(b"synthetic artifact").hexdigest())
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})

    def test_changed_and_missing_artifacts(self):
        with TemporaryDirectory(prefix="synthetic_integrity_") as folder:
            root = Path(folder)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"first")
            before = artifact_fingerprints(root, (artifact.name,))
            artifact.write_bytes(b"changed")
            self.assertNotEqual(before, artifact_fingerprints(root, (artifact.name,)))
            with self.assertRaises(ValueError):
                artifact_fingerprints(root, ("missing.bin",))

    def test_traversal_absolute_duplicate_and_empty_names_rejected(self):
        with TemporaryDirectory(prefix="synthetic_integrity_") as folder:
            root = Path(folder)
            (root / "a.bin").write_bytes(b"synthetic")
            for names in ((), ("a.bin", "a.bin"), ("..",), ("../a.bin",), ("sub/a.bin",),
                          ("sub\\a.bin",), (str(root / "a.bin"),), ("C:a.bin",), ("",)):
                with self.subTest(names=names), self.assertRaises(ValueError):
                    artifact_fingerprints(root, names)

