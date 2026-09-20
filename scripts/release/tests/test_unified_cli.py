import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from gel_release import assets
from gel_release.models import CandidateRecord

REPO_ROOT = Path(__file__).resolve().parents[3]


class ModelBoundaryTests(unittest.TestCase):
    def test_candidate_rejects_extra_fields(self):
        with self.assertRaises(ValidationError):
            CandidateRecord.model_validate(
                {
                    "schema_version": 2,
                    "line": "release/v1.x",
                    "pr_number": 1,
                    "phase": None,
                    "version": "1.2.3",
                    "tag": "v1.2.3",
                    "draft_release_id": 1,
                    "source_sha": "a" * 40,
                    "build_sha": "a" * 40,
                    "source_snapshot": "d" * 64,
                    "base_sha": "b" * 40,
                    "build_date": "2026-09-12T00:00:00+00:00",
                    "workflow_runs": [],
                    "attestation": {
                        "predicate_type": "https://slsa.dev/provenance/v1",
                        "subject_count": 1,
                    },
                    "assets": [],
                    "unexpected": True,
                }
            )

    def test_candidate_rejects_tag_version_disagreement(self):
        with self.assertRaisesRegex(ValidationError, "tag must be v1.2.3"):
            CandidateRecord.model_validate(
                {
                    "schema_version": 2,
                    "line": "release/v1.x",
                    "pr_number": 1,
                    "phase": None,
                    "version": "1.2.3",
                    "tag": "v1.2.4",
                    "draft_release_id": 1,
                    "source_sha": "a" * 40,
                    "build_sha": "a" * 40,
                    "source_snapshot": "d" * 64,
                    "base_sha": "b" * 40,
                    "build_date": "2026-09-12T00:00:00+00:00",
                    "workflow_runs": [],
                    "attestation": {
                        "predicate_type": "https://slsa.dev/provenance/v1",
                        "subject_count": 1,
                    },
                    "assets": [],
                }
            )


class CliBoundaryTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "gel_release.cli", *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )

    def test_build_matrix_comes_from_canonical_targets(self):
        completed = self._run("matrix", "build")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout)["include"],
            [
                {
                    "target": target.triple,
                    "runner": target.runner,
                    "linux_packages": target.deb_arch is not None,
                }
                for target in assets.TARGETS
            ],
        )

    def test_release_channel_follows_version(self):
        for version, channel in (
            ("7.11.0", "stable"),
            ("7.11.0-rc.1", "testing"),
        ):
            with self.subTest(version=version):
                completed = self._run("channel", "--version", version)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout.strip(), channel)

    def test_release_channel_rejects_unsupported_version(self):
        for version in ("7.11.0-preview.1", "7.11.0-dev.4121"):
            with self.subTest(version=version):
                completed = self._run("channel", "--version", version)
                self.assertEqual(completed.returncode, 2)
                self.assertIn("unsupported release version", completed.stderr)

    def test_smoke_matrix_excludes_distribution_only_target(self):
        completed = self._run("matrix", "smoke")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        triples = {entry["target"] for entry in json.loads(completed.stdout)["include"]}
        self.assertEqual(triples, {target.triple for target in assets.REGISTRY_TARGETS})
        self.assertNotIn("x86_64-apple-darwin", triples)

    def test_candidate_verify_rejects_malformed_json_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "record.json"
            record.write_text("{}")
            completed = self._run(
                "candidate",
                "verify",
                "--version",
                "1.2.3",
                "--dist-dir",
                tmp,
                "--record",
                str(record),
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("validation", completed.stderr.lower())

    def test_candidate_write_rejects_malformed_asset_ids_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = root / "dist"
            dist.mkdir()
            asset_ids = root / "asset-ids.json"
            output = root / "record.json"
            for malformed in (["name"], None):
                with self.subTest(malformed=malformed):
                    asset_ids.write_text(json.dumps(malformed))
                    completed = self._run(
                        "candidate",
                        "write",
                        "--line",
                        "release/v1.x",
                        "--pr-number",
                        "1",
                        "--version",
                        "1.2.3",
                        "--draft-release-id",
                        "1",
                        "--source-sha",
                        "a" * 40,
                        "--build-sha",
                        "a" * 40,
                        "--source-snapshot",
                        "d" * 64,
                        "--base-sha",
                        "b" * 40,
                        "--build-date",
                        "2026-09-12T00:00:00+00:00",
                        "--run-id",
                        "1",
                        "--run-attempt",
                        "1",
                        "--dist-dir",
                        str(dist),
                        "--asset-ids",
                        str(asset_ids),
                        "--out",
                        str(output),
                    )
                    self.assertEqual(completed.returncode, 2)
                    self.assertIn("validation error", completed.stderr.lower())
                    self.assertNotIn("traceback", completed.stderr.lower())

    def test_candidate_mismatch_is_reported_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "record.json"
            record.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "line": "release/v1.x",
                        "pr_number": 1,
                        "phase": None,
                        "version": "1.2.3",
                        "tag": "v1.2.3",
                        "draft_release_id": 1,
                        "source_sha": "a" * 40,
                        "build_sha": "a" * 40,
                        "source_snapshot": "d" * 64,
                        "base_sha": "b" * 40,
                        "build_date": "2026-09-12T00:00:00+00:00",
                        "workflow_runs": [],
                        "attestation": {
                            "predicate_type": "https://slsa.dev/provenance/v1",
                            "subject_count": 1,
                        },
                        "assets": [
                            {
                                "id": 1,
                                "name": "asset",
                                "size": 1,
                                "sha256": "0" * 64,
                                "blake2b512": "0" * 128,
                            }
                        ],
                    }
                )
            )
            completed = self._run(
                "candidate",
                "verify",
                "--version",
                "1.2.4",
                "--dist-dir",
                tmp,
                "--record",
                str(record),
            )
        self.assertEqual(completed.returncode, 2)
        self.assertNotIn("Traceback", completed.stderr)

    def _identity(self) -> dict[str, object]:
        return {
            "line": "release/v7.x",
            "pr_number": 101,
            "base_sha": "b" * 40,
            "source_sha": "a" * 40,
            "build_sha": "a" * 40,
            "source_snapshot": "d" * 64,
            "phase": None,
            "version": "7.1.0",
            "channel": "stable",
        }

    def _select_draft(self, tmp: Path, releases: object, identity: dict[str, object]):
        releases_json = tmp / "releases.json"
        identity_json = tmp / "identity.json"
        releases_json.write_text(json.dumps(releases))
        identity_json.write_text(json.dumps(identity))
        return self._run(
            "select-draft",
            "--releases-json",
            str(releases_json),
            "--identity-json",
            str(identity_json),
        )

    def test_select_draft_is_the_workflow_boundary_for_draft_reuse(self):
        identity = self._identity()
        draft = {
            "id": 42,
            "tag_name": "v7.1.0",
            "name": "v7.1.0",
            "draft": True,
            "prerelease": False,
            "body": json.dumps({"candidate_identity": identity}),
        }
        stale = {
            **draft,
            "id": 43,
            "body": json.dumps(
                {"candidate_identity": {**identity, "source_sha": "c" * 40, "build_sha": "c" * 40}}
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reuse = self._select_draft(root, [draft], identity)
            self.assertEqual(reuse.returncode, 0, reuse.stderr)
            self.assertEqual(
                json.loads(reuse.stdout),
                {"id": 42, "stale_build_sha": "", "stale_base_sha": ""},
            )

            replace = self._select_draft(root, [stale], identity)
            self.assertEqual(replace.returncode, 0, replace.stderr)
            self.assertEqual(
                json.loads(replace.stdout),
                {"id": 43, "stale_build_sha": "c" * 40, "stale_base_sha": ""},
            )

            none = self._select_draft(root, [], identity)
            self.assertEqual(none.returncode, 0, none.stderr)
            self.assertEqual(none.stdout, "")

    def test_select_draft_fails_closed_on_a_published_release(self):
        identity = self._identity()
        published = {
            "id": 42,
            "tag_name": "v7.1.0",
            "name": "v7.1.0",
            "draft": False,
            "prerelease": False,
            "body": json.dumps({"candidate_identity": identity}),
        }
        with tempfile.TemporaryDirectory() as tmp:
            completed = self._select_draft(Path(tmp), [published], identity)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("published", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_release_and_tag_inputs_require_the_shape_the_workflows_emit(self):
        # The workflows build these with `jq add` and `jq -Rsc split`, so a
        # bare JSON list is the only accepted shape.
        identity = self._identity()
        with tempfile.TemporaryDirectory() as tmp:
            completed = self._select_draft(Path(tmp), {"releases": []}, identity)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("JSON list of release objects", completed.stderr)

    def test_internal_modules_do_not_expose_standalone_clis(self):
        for name in (
            "candidate",
            "linux_packages",
            "package_target",
            "registry_manifest",
            "source_equivalence",
            "verify_draft",
        ):
            module = importlib.import_module(f"gel_release.{name}")
            self.assertFalse(hasattr(module, "main"), name)


if __name__ == "__main__":
    unittest.main()
