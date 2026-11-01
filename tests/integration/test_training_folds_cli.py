from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from face_recognition.presentation.cli.inspect_training_folds import main
from tests.training_fold_fakes import training_fixture


class TrainingFoldsCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="synthetic_fold_cli_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.processed, self.root, _, self.split = training_fixture(self.directory)
        self.args = ["--processed-root", str(self.processed), "--split-manifest", str(self.split)]

    def invoke(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_diagnostics_print_only_aggregate_counts(self):
        status, output, error = self.invoke(self.args)
        self.assertEqual(status, 0)
        self.assertEqual(error, "")
        self.assertEqual(output, "Fold A:\n  Fit identities: 2\n  Fit samples: 8\n  Validation identities: 2\n  Validation samples: 8\n"
                                 "Fold B:\n  Fit identities: 2\n  Fit samples: 8\n  Validation identities: 2\n  Validation samples: 8\n")
        self.assertNotIn("known_alpha", output)
        self.assertNotIn(str(self.directory), output)

    def test_invalid_split_returns_one(self):
        self.split.write_text("invalid", encoding="utf-8")
        status, output, error = self.invoke(self.args)
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("Invalid or unreadable identity split", error)

    def test_invalid_features_fail_without_printing_values(self):
        (self.root / "known_alpha" / "visit_a" / "center.jsonl").write_text('{"features": [987654.321', encoding="utf-8")
        status, output, error = self.invoke(self.args)
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("Invalid or missing development sample", error)
        self.assertNotIn("987654", error)

    def test_required_arguments_use_argparse_exit_code_two(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main([])
        self.assertEqual(caught.exception.code, 2)
