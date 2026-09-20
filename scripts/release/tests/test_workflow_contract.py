"""Contracts for the immutable release candidate workflows.

GitHub evaluates workflow YAML outside the Python test process, so these tests
keep the important graph and permission boundaries reviewable locally. They
deliberately inspect parsed job configuration and the commands at each
boundary rather than trying to execute GitHub Actions on a developer machine.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
README = REPO_ROOT / "README.md"
BRANCH_PROTECTION = REPO_ROOT / ".github" / "branch-protection.md"
SHA = re.compile(r"^[0-9a-f]{40}$")
CREATE_APP_TOKEN = "actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1"
RELEASE_TOKEN = "${{ steps.release-token.outputs.token }}"


def _workflow(name: str) -> dict:
    path = WORKFLOWS / name
    assert path.is_file(), f"missing workflow {path}"
    value = yaml.safe_load(path.read_text())
    assert isinstance(value, dict)
    # PyYAML 1.1 resolves the YAML 1.2 key `on` to boolean True.  Normalize
    # that parser quirk so the contract remains independent of the PyYAML
    # version used by the release tooling.
    if True in value and "on" not in value:
        value["on"] = value.pop(True)
    return value


def _steps(job: dict) -> list[dict]:
    steps = job.get("steps", [])
    assert isinstance(steps, list)
    return [step for step in steps if isinstance(step, dict)]


def _run_text(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in _steps(job))


class ContinuousIntegrationContractTests(unittest.TestCase):
    """The release suite only protects the pipeline when CI actually runs it."""

    def test_ci_runs_the_release_test_suite_and_release_linters(self):
        workflow = _workflow("ci.yml")
        jobs = workflow["jobs"]
        assert "release-tooling" in jobs
        text = _run_text(jobs["release-tooling"])
        assert "uv sync --frozen" in text
        assert "ruff check" in text
        assert "ruff format --check" in text
        assert "pytest scripts/release/tests" in text

    def test_quality_lints_the_whole_workflows_directory(self):
        # The check exists but nothing invokes it is the invisible failure:
        # assert the directory-wide form so re-narrowing to one file fails.
        quality_text = _run_text(_workflow("ci.yml")["jobs"]["quality"])
        for line in quality_text.splitlines():
            if line.startswith("scripts/ci/check-action-pins.sh"):
                self.assertEqual(line, "scripts/ci/check-action-pins.sh")
            if line.startswith('"$(go env GOPATH)/bin/actionlint"'):
                self.assertEqual(line, '"$(go env GOPATH)/bin/actionlint"')

    def test_ci_actions_are_sha_pinned_with_a_version_comment(self):
        text = (WORKFLOWS / "ci.yml").read_text()
        for line in text.splitlines():
            if "uses:" not in line or "./" in line:
                continue
            assert re.search(r"uses:\s+[^@\s]+@[0-9a-f]{40}\s+#\s+.+$", line)


class ReleaseAppAuthenticationContractTests(unittest.TestCase):
    """Every privileged job gets its own least-privilege installation token."""

    expected_permissions = {
        ("release-pr.yml", "prepare"): {
            "permission-actions": "write",
            "permission-contents": "write",
            "permission-metadata": "read",
            "permission-pull-requests": "write",
        },
        ("release-controller.yml", "resolve"): {
            "permission-actions": "write",
            "permission-contents": "write",
            "permission-issues": "read",
            "permission-metadata": "read",
            "permission-pull-requests": "read",
        },
        ("release-candidate.yml", "commit-stable"): {
            "permission-actions": "write",
            "permission-contents": "write",
            "permission-metadata": "read",
            "permission-pull-requests": "read",
        },
        ("release-candidate.yml", "cleanup-temporary-refs"): {
            "permission-contents": "write",
            "permission-metadata": "read",
        },
    }

    def test_privileged_jobs_mint_fresh_minimally_scoped_app_tokens(self):
        for (workflow_name, job_name), expected_permissions in self.expected_permissions.items():
            with self.subTest(workflow=workflow_name, job=job_name):
                job = _workflow(workflow_name)["jobs"][job_name]
                mint_steps = [step for step in _steps(job) if step.get("uses") == CREATE_APP_TOKEN]
                self.assertEqual(len(mint_steps), 1)
                mint = mint_steps[0]
                self.assertEqual(mint.get("id"), "release-token")
                inputs = mint.get("with", {})
                self.assertEqual(inputs.get("app-id"), "${{ vars.GEL_RELEASER_APP_ID }}")
                self.assertEqual(inputs.get("private-key"), "${{ secrets.GEL_RELEASER_KEY }}")
                permissions = {
                    key: value for key, value in inputs.items() if key.startswith("permission-")
                }
                self.assertEqual(permissions, expected_permissions)

    def test_app_token_is_consumed_only_as_checkout_or_gh_authentication(self):
        def token_paths(value: object, path: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
            if value == RELEASE_TOKEN:
                return [path]
            if isinstance(value, dict):
                return [
                    found
                    for key, child in value.items()
                    for found in token_paths(child, (*path, key))
                ]
            if isinstance(value, list):
                return [
                    found
                    for index, child in enumerate(value)
                    for found in token_paths(child, (*path, index))
                ]
            return []

        for workflow_name, job_name in self.expected_permissions:
            with self.subTest(workflow=workflow_name, job=job_name):
                job = _workflow(workflow_name)["jobs"][job_name]
                paths = token_paths(job)
                self.assertTrue(paths)
                for path in paths:
                    self.assertEqual(path[0], "steps")
                    self.assertIn(path[-2:], (("with", "token"), ("env", "GH_TOKEN")))

    def test_personal_access_token_configuration_is_gone(self):
        for name in (
            "release-candidate.yml",
            "release-controller.yml",
            "release-pr.yml",
        ):
            with self.subTest(workflow=name):
                text = (WORKFLOWS / name).read_text()
                self.assertNotIn("RELEASE_BOT", text)
                self.assertNotIn("secrets.RELEASE_BOT_TOKEN", text)


class ControllerTriggerContractTests(unittest.TestCase):
    def test_controller_only_triggers_for_release_line_targets(self):
        # GitHub matches `branches:` against the PR base, so feature PRs
        # targeting master never start the controller at all.
        workflow = _workflow("release-controller.yml")
        assert workflow["on"]["pull_request"]["branches"] == ["release/v*.x"]

    def test_every_step_after_the_classify_decision_is_guarded_by_it(self):
        resolve = _workflow("release-controller.yml")["jobs"]["resolve"]
        steps = _steps(resolve)
        classify_index = next(
            index for index, step in enumerate(steps) if step.get("id") == "classify"
        )
        classify_text = str(steps[classify_index].get("run", ""))
        assert "classify-pr" in classify_text
        for step in steps[classify_index + 1 :]:
            with self.subTest(step=step.get("name")):
                assert "steps.classify.outputs.should_run == 'true'" in str(step.get("if", "")), (
                    "is not guarded by the classify decision"
                )

    def test_repository_dispatch_types_are_disjoint(self):
        # Nothing in the repository sends these events; they are manual retry
        # escapes. Sharing one type would let a single operator retry run a
        # line regeneration and a full candidate staging concurrently.
        pr_types = set(_workflow("release-pr.yml")["on"]["repository_dispatch"]["types"])
        controller_types = set(
            _workflow("release-controller.yml")["on"]["repository_dispatch"]["types"]
        )
        self.assertEqual(pr_types, {"release-line"})
        self.assertEqual(controller_types, {"release-candidate"})


class CandidateInputContractTests(unittest.TestCase):
    def test_archive_layout_checks_do_not_sigpipe_the_listing_commands(self):
        stage = _workflow("release-candidate.yml")["jobs"]["stage"]
        check = next(step for step in _steps(stage) if step.get("name") == "Assert archive layouts")
        run = str(check.get("run", ""))
        self.assertNotIn("| grep -q", run)
        self.assertIn("tar -tzf", run)
        self.assertIn("unzip -Z1", run)
        self.assertEqual(run.count(">/dev/null"), 4)

    def test_candidate_uv_steps_pin_a_supported_python(self):
        workflow = _workflow("release-candidate.yml")
        setup_steps = [
            step
            for job in workflow["jobs"].values()
            for step in _steps(job)
            if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
        ]
        self.assertTrue(setup_steps)
        for step in setup_steps:
            with self.subTest(job_step=step):
                self.assertEqual(step.get("with", {}).get("python-version"), "3.13")

    def test_candidate_only_accepts_a_dispatched_json_identity(self):
        workflow = _workflow("release-candidate.yml")
        triggers = workflow["on"]
        assert "workflow_dispatch" in triggers
        # `workflow_call` would bypass the immutable-ref check, which is gated
        # on `github.event_name == 'workflow_dispatch'`. Nothing calls this
        # workflow; the controller dispatches it with `gh workflow run`.
        assert "workflow_call" not in triggers

        dispatch_inputs = triggers["workflow_dispatch"]["inputs"]
        assert set(dispatch_inputs) == {"identity", "line"}
        assert dispatch_inputs["identity"]["required"] is True

        # The explicit-field assembly branch is gone with its inputs.
        identity_text = _run_text(workflow["jobs"]["identity"])
        assert "INPUT_SOURCE_SNAPSHOT" not in identity_text

    def test_candidate_stages_and_verifies_the_draft(self):
        workflow = _workflow("release-candidate.yml")
        assert "jobs" in workflow
        assert "stage" in workflow["jobs"]
        assert "verify" in workflow["jobs"]
        assert "draft_release_id" in workflow["jobs"]["stage"]["outputs"]
        assert "verified_candidate" in workflow["jobs"]["verify"]["outputs"]

    def test_draft_verification_token_can_read_unpublished_releases(self):
        # GitHub only exposes draft releases to identities with push access.
        # For GITHUB_TOKEN that requires contents: write even though this job
        # performs no mutations; contents: read receives HTTP 403 here.
        verify = _workflow("release-candidate.yml")["jobs"]["verify"]
        assert verify["permissions"] == {
            "contents": "write",
            "attestations": "read",
        }

    def test_controller_dispatch_matches_candidate_identity_input(self):
        controller = (WORKFLOWS / "release-controller.yml").read_text()
        assert "gh workflow run release-candidate.yml" in controller
        assert '-f identity="$identity"' in controller
        assert 'build_sha="$(jq -r \'.build_sha\' "$RUNNER_TEMP/identity.json")"' in controller
        assert 'git push origin "$build_sha:refs/heads/$candidate_ref"' in controller
        assert 'jq -c . "$RUNNER_TEMP/identity.json"' in controller
        assert '--ref "$candidate_ref"' in controller

    def test_dispatch_source_is_bound_to_the_immutable_build_sha(self):
        workflow = _workflow("release-candidate.yml")
        identity_text = _run_text(workflow["jobs"]["identity"])
        assert any(
            step.get("if") == "github.event_name == 'workflow_dispatch'"
            for step in _steps(workflow["jobs"]["identity"])
        )
        assert '"$GITHUB_SHA" != "$BUILD_SHA"' in identity_text
        # The gate names expected and actual instead of failing silently.
        assert "is not the build SHA" in identity_text

    def test_json_identity_normalization_keeps_the_object(self):
        identity_text = _run_text(_workflow("release-candidate.yml")["jobs"]["identity"])
        assert 'if type == "object" then . else error(' in identity_text
        assert ".build_sha = (.build_sha // .source_sha)" in identity_text

    def test_stage_uploads_use_authenticated_cli_and_share_mutation_lock(self):
        workflow = _workflow("release-candidate.yml")
        stage = workflow["jobs"]["stage"]
        assert stage["concurrency"] == {
            "group": "release-mutation",
            "cancel-in-progress": False,
        }
        upload = next(
            step for step in _steps(stage) if step.get("name") == "Upload every distribution asset"
        )
        assert upload["env"]["GH_TOKEN"] == "${{ github.token }}"
        record = next(
            step
            for step in _steps(stage)
            if step.get("name") == "Write the immutable candidate record"
        )
        assert record["env"]["GH_TOKEN"] == "${{ github.token }}"
        assert "select-draft" in _run_text(stage)
        assert "git/ref/tags/$TAG" in _run_text(stage)
        assert "git/refs/tags/$TAG" in _run_text(stage)
        assert "git/tags/$tag_target" in _run_text(stage)

    def test_stage_revalidates_the_draft_against_fresh_state_before_replacing_assets(self):
        # Asset replacement is only safe against the draft that was selected,
        # so the stage job re-fetches that one release and re-runs the same
        # selector before mutating it. Both calls go through the CLI, so no
        # workflow imports a module-private selector.
        workflow = _workflow("release-candidate.yml")
        text = _run_text(workflow["jobs"]["stage"])
        assert text.count("gel-release select-draft") == 2
        assert "releases/$existing" in text
        assert '--releases-json "$RUNNER_TEMP/reusable-release.json"' in text
        assert "the selected draft changed before asset replacement" in text
        assert "stale_base_sha" in text
        assert "compare/$stale_base_sha...$current_base_sha" in text
        assert '[[ "$base_status" == ahead ]]' in text
        assert "_body_identity" not in text
        assert "find_replaceable_draft" not in text
        assert "find_reusable_draft" not in text

    def test_candidate_cleanup_keeps_successful_preview_ref_until_publication(self):
        workflow = _workflow("release-candidate.yml")
        cleanup = workflow["jobs"]["cleanup-temporary-refs"]
        assert "always()" in cleanup["if"]
        assert "needs.stage.result == 'success'" in cleanup["if"]
        assert "needs.verify.result == 'success'" in cleanup["if"]
        text = _run_text(cleanup)
        assert "preview_ref" not in text


class CandidateGraphContractTests(unittest.TestCase):
    def test_target_and_smoke_matrices_are_derived_from_release_cli(self):
        workflow = _workflow("release-candidate.yml")
        plan = workflow["jobs"]["plan"]
        plan_run = _run_text(plan)
        assert "gel-release matrix build" in plan_run
        assert "gel-release matrix smoke" in plan_run

        build_matrix = workflow["jobs"]["build"]["strategy"]["matrix"]
        smoke_matrix = workflow["jobs"]["smoke"]["strategy"]["matrix"]
        assert "fromJSON(needs.plan.outputs.build_matrix)" in str(build_matrix)
        assert "fromJSON(needs.plan.outputs.smoke_matrix)" in str(smoke_matrix)

    def test_every_build_job_checks_out_the_immutable_build_sha(self):
        workflow = _workflow("release-candidate.yml")
        jobs = workflow["jobs"]
        for name in ("plan", "completions", "build"):
            steps = _steps(jobs[name])
            checkouts = [
                step for step in steps if step.get("uses", "").startswith("actions/checkout@")
            ]
            assert checkouts, f"{name} has no checkout"
            assert any("build_sha" in str(step.get("with", {}).get("ref")) for step in checkouts)

    def test_install_matrix_consumes_candidate_artifacts_and_gates_staging(self):
        workflow = _workflow("release-candidate.yml")
        install = workflow["jobs"]["install"]
        assert install["uses"] == "./.github/workflows/release-install-e2e.yml"
        assert "build_sha" in install["with"]
        assert "version" in install["with"]
        assert "install" in workflow["jobs"]["stage"]["needs"]

        install_workflow = _workflow("release-install-e2e.yml")
        assert "workflow_call" in install_workflow["on"]
        install_text = "\n".join(
            _run_text(job)
            for name, job in install_workflow["jobs"].items()
            if isinstance(job, dict)
        )
        assert "actions/download-artifact@" in "\n".join(
            str(step.get("uses", ""))
            for job in install_workflow["jobs"].values()
            if isinstance(job, dict)
            for step in _steps(job)
        )
        assert "run-install-scenario.sh" in install_text
        for scenario in (
            "e2e_direct",
            "e2e_apt",
            "e2e_dnf",
            "e2e_pacman",
            "e2e_homebrew",
            "e2e_nix",
            "e2e_scoop",
            "e2e_winget",
        ):
            assert scenario in install_text

    def test_stage_uploads_exact_inventory_attests_and_reads_api_back(self):
        workflow = _workflow("release-candidate.yml")
        stage = workflow["jobs"]["stage"]
        stage_text = _run_text(stage)
        assert "assemble-stage" in stage_text
        assert "expected_assets" in stage_text
        assert "gh release upload" in stage_text
        assert "attest-build-provenance@" in "\n".join(
            str(step.get("uses", "")) for step in _steps(stage)
        )
        assert "verify-draft" in _run_text(workflow["jobs"]["verify"])
        verify_text = _run_text(workflow["jobs"]["verify"])
        assert "--download-dir readback" in verify_text
        assert "releases/" in verify_text

    def test_stable_record_is_committed_to_pr_and_preview_is_release_asset(self):
        workflow = _workflow("release-candidate.yml")
        jobs = workflow["jobs"]
        stable = _run_text(jobs["commit-stable"])
        assert "packaging/release-candidate.json" in stable
        assert "pulls/$PR_NUMBER" in stable
        assert "git push" in stable
        assert "assert_live_identity" in stable

        preview = _run_text(jobs["stage"])
        assert "gel-candidate.json" in preview
        assert "phase" in preview
        assert "gh release upload" in preview

    def test_published_release_workflow_has_no_build_package_or_upload_steps(self):
        path = WORKFLOWS / "release-publish.yml"
        if not path.exists():
            return
        text = path.read_text()
        for forbidden in ("cargo build", "cargo deb", "generate-rpm", "gh release upload"):
            assert forbidden not in text


class WorkflowSafetyContractTests(unittest.TestCase):
    def test_new_workflows_have_read_defaults_and_full_action_pins(self):
        for name in ("release-candidate.yml", "release-install-e2e.yml"):
            text = (WORKFLOWS / name).read_text()
            head = text.split("jobs:", 1)[0]
            assert "permissions:\n  contents: read" in head
            for line in text.splitlines():
                if "uses:" not in line or "./" in line:
                    continue
                assert re.search(r"uses:\s+[^@\s]+@[0-9a-f]{40}\s+#\s+.+$", line)

    def test_candidate_minimizes_elevated_permissions_to_jobs_that_need_them(self):
        workflow = _workflow("release-candidate.yml")
        assert workflow["permissions"] == {"contents": "read"}
        assert workflow["jobs"]["stage"]["permissions"] == {
            "contents": "write",
            "id-token": "write",
            "attestations": "write",
        }
        assert workflow["jobs"]["verify"]["permissions"] == {
            "contents": "write",
            "attestations": "read",
        }

    def test_release_workflow_guards_parameterize_the_operating_repository(self):
        # The production fallback keeps the guard exact by default; the
        # repository variable lets a rehearsal scratch repo run the same
        # workflows against itself.
        for name in (
            "release-candidate-check.yml",
            "release-pr.yml",
            "release-controller.yml",
            "release-publish.yml",
        ):
            text = (WORKFLOWS / name).read_text()
            with self.subTest(workflow=name):
                assert "github.repository == (vars.RELEASE_REPOSITORY || 'gelstable/gel-cli')" in (
                    text
                )
                assert "github.repository == 'gelstable/gel-cli'" not in text

    def test_publication_is_serialized_across_lines_and_uses_read_defaults(self):
        workflow = _workflow("release-publish.yml")
        assert workflow["on"]["push"]["branches"] == ["release/v*.x"]
        assert workflow["on"]["workflow_run"] == {
            "workflows": ["Release candidate"],
            "types": ["completed"],
        }
        assert workflow["permissions"] == {"contents": "read"}
        assert workflow["concurrency"] == {
            "group": "release-publish",
            "cancel-in-progress": False,
        }
        assert workflow["jobs"]["publish"]["concurrency"] == {
            "group": "release-mutation",
            "cancel-in-progress": False,
        }

    def test_publication_rechecks_and_only_patches_existing_releases(self):
        workflow = _workflow("release-publish.yml")
        text = _run_text(workflow["jobs"]["publish"])
        assert "publish_preview" in text
        assert "publish_stable" in text
        assert "should_make_latest" in text
        assert "make_latest" in text
        assert "verify-draft" in text
        assert "source-equivalence" in text
        preview_steps = [
            step
            for step in _steps(workflow["jobs"]["publish"])
            if step.get("name") == "Publish an authorized preview"
        ]
        assert preview_steps
        assert "github.event_name == 'workflow_run'" in str(preview_steps[0].get("if"))
        for forbidden in (
            "cargo build",
            "cargo deb",
            "generate-rpm",
            "gh release upload",
            "gh api -X DELETE",
        ):
            assert forbidden not in text

    def test_preview_publication_refreshes_live_state_and_cleans_refs(self):
        workflow = _workflow("release-publish.yml")
        preview = _run_text(workflow["jobs"]["publish"])
        assert "fetch_live_preview_pr" in preview
        assert "release-candidate.json" in preview
        assert "gel-candidate.json" in preview
        cleanup = workflow["jobs"]["cleanup-temporary-refs"]
        assert "always()" in cleanup["if"]
        assert "needs.publish.result == 'success'" in cleanup["if"]
        assert "workflow_run.head_branch" in str(cleanup)
        assert "preview_ref" in _run_text(cleanup)


class StableMergeWorkflowContractTests(unittest.TestCase):
    def test_stable_gate_runs_for_all_release_pr_state_changes(self):
        workflow = _workflow("release-candidate-check.yml")
        triggers = workflow["on"]
        assert triggers["pull_request"]["types"] == [
            "opened",
            "synchronize",
            "reopened",
            "labeled",
            "unlabeled",
        ]
        assert triggers["pull_request"]["branches"] == ["release/v*.x"]
        assert "candidate" in workflow["jobs"]
        assert workflow["jobs"]["candidate"]["name"] == "stable merge gate"

    def test_stable_gate_uses_only_draft_visibility_permissions_and_pinned_actions(self):
        workflow = _workflow("release-candidate-check.yml")
        assert workflow["permissions"] == {"contents": "read"}
        assert workflow["jobs"]["candidate"]["permissions"] == {
            # GitHub hides drafts from GITHUB_TOKEN unless it has push access.
            # The gate performs no mutation despite this visibility grant.
            "contents": "write",
            "pull-requests": "read",
            "attestations": "read",
        }
        text = (WORKFLOWS / "release-candidate-check.yml").read_text()
        for line in text.splitlines():
            if "uses:" not in line or "./" in line:
                continue
            assert re.search(r"uses:\s+[^@\s]+@[0-9a-f]{40}\s+#\s+.+$", line)

    def test_stable_gate_binds_the_live_head_snapshot(self):
        # The gate computes the live head's meaningful source snapshot (pure
        # git, no toolchain) and threads it into check_stable_merge, so the
        # head GitHub reports right now must match the tested snapshot. No
        # caller-supplied version field is trusted: the record-only successor
        # proof and the Cargo version checks already bind the version exactly.
        text = _run_text(_workflow("release-candidate-check.yml")["jobs"]["candidate"])
        assert "gel-release snapshot --rev" in text
        assert "live_snapshot=" in text

    def test_ordinary_backport_has_safe_path_and_generated_pr_keeps_full_gate(self):
        workflow = _workflow("release-candidate-check.yml")
        text = _run_text(workflow["jobs"]["candidate"])
        assert "release-head" in text
        assert "packaging/release-candidate.json" in text
        # Preview records are release assets; the repo path never exists.
        assert "packaging/gel-candidate.json" not in text
        assert "generated" in text
        assert "candidate record" in text

    def test_stable_gate_is_dispatched_on_the_pr_head_branch(self):
        """A dispatched check run attaches to the head SHA of its ref.

        Branch protection evaluates the required context on the generated PR's
        head commit, so the gate has to be dispatched on the PR head branch.
        Dispatching on a fixed branch (`master`, `$CONTROLLER_REF`, ...) posts
        the result on that branch and can never satisfy the required check.
        """
        check = _workflow("release-candidate-check.yml")
        triggers = check["on"]
        assert "workflow_dispatch" in triggers
        assert "pr_number" in triggers["workflow_dispatch"]["inputs"]

        candidate_workflow = _workflow("release-candidate.yml")
        assert candidate_workflow["jobs"]["commit-stable"]["permissions"]["actions"] == "write"

        sources = {
            "release-candidate.yml": _run_text(candidate_workflow["jobs"]["commit-stable"]),
            "release-controller.yml": (WORKFLOWS / "release-controller.yml").read_text(),
        }
        dispatches = 0
        for name, text in sources.items():
            for match in re.finditer(
                r"gh workflow run release-candidate-check\.yml"
                r"(?P<args>(?:[^\n]*\\\n)+[^\n]*)",
                text,
            ):
                dispatches += 1
                args = match.group("args")
                with self.subTest(workflow=name):
                    ref = re.search(r'--ref\s+"(?P<ref>[^"]+)"', args)
                    assert ref, f"{name} dispatches the gate without an explicit --ref"
                    # The head branch of the live PR, never a fixed branch.
                    self.assertEqual(ref.group("ref"), "$head_ref")
                    assert "-f pr_number=" in args
                    assert "head.ref" in text

        assert dispatches == 2, f"expected both dispatch sites, found {dispatches}"

        # The dispatched run must refuse to report on a commit it did not
        # verify, so dispatching on a moving branch cannot produce a passing
        # result for the wrong head.
        gate = _run_text(check["jobs"]["candidate"])
        assert '"$GITHUB_SHA" != "$HEAD_SHA"' in gate
        pin = next(
            step
            for step in _steps(check["jobs"]["candidate"])
            if "GITHUB_SHA" in str(step.get("run", ""))
        )
        assert pin.get("if") == "github.event_name == 'workflow_dispatch'"
        assert "head_sha" in str(check["jobs"]["candidate"]["steps"])

    def test_event_firing_token_has_no_default_token_fallback(self):
        """`GITHUB_TOKEN` pushes fire no events, so the gate would never run."""
        for name in ("release-candidate.yml", "release-controller.yml"):
            text = (WORKFLOWS / name).read_text()
            with self.subTest(workflow=name):
                assert "GITHUB_TOKEN" in text
                assert "workflow events" in text
                assert "|| github.token" not in text

    def test_controller_and_candidate_report_selection_to_operators(self):
        controller = _run_text(_workflow("release-controller.yml")["jobs"]["resolve"])
        assert "GITHUB_STEP_SUMMARY" in controller
        for field in ("- line: ", "- version: ", "- source SHA: ", "- phase: "):
            assert field in controller
        assert "release rejected: " in controller

        candidate = "\n".join(
            _run_text(job)
            for job in _workflow("release-candidate.yml")["jobs"].values()
            if isinstance(job, dict)
        )
        assert "GITHUB_STEP_SUMMARY" in candidate
        assert "- draft release id: " in candidate
        assert "candidate rejected: " in candidate

        publish = _run_text(_workflow("release-publish.yml")["jobs"]["publish"])
        assert "publication rejected: " in publish
        assert "GITHUB_STEP_SUMMARY" in publish


class ReleaseMigrationDocumentationContractTests(unittest.TestCase):
    """The migration story is the branch's handoff document; keep its anchors."""

    def test_readme_and_protection_rules_cover_the_release_migration(self):
        readme = README.read_text()
        protection = BRANCH_PROTECTION.read_text()
        anchors = {
            readme: (
                "release/vN.x",
                "master",
                "Cargo.toml",
                "Cargo.lock",
                ".changeset/",
                "cherry-pick",
                "stable merge gate",
                "Release publish",
                "separate snapshot pull request",
                "https://packages.geldata.com",
                "[registry]",
            ),
            protection: (
                "release/v*.x",
                "stable merge gate",
                "required status check",
                "Do not allow bypassing",
                "ordinary backport",
                "trust boundary",
            ),
        }
        for text, required in anchors.items():
            for anchor in required:
                with self.subTest(file="readme" if text is readme else "protection", anchor=anchor):
                    self.assertIn(anchor, text)

    def test_readme_documents_the_existing_major_line_procedure(self):
        text = README.read_text()
        # The v7 migration must not tell readers to re-tag an old version.
        self.assertIn("already has published releases", text)
        self.assertIn("7.10.2", text)
        self.assertIn("7.11.0", text)
        self.assertIn("Never", text)

    def test_branch_protection_documents_merge_method_support(self):
        text = BRANCH_PROTECTION.read_text()
        self.assertIn("merge methods", text)
        self.assertIn("up to date", text)


if __name__ == "__main__":
    unittest.main()
