"""GitHub release-line identity and preview commit mechanics.

The controller deliberately keeps GitHub API concerns at the edge.  It passes
the freshly fetched pull request and the published release inventory into
``resolve_candidate``; that function returns one immutable candidate identity
or ``None`` when the current phase and source snapshot have already been
published.  Preview builds are derived from the original pull-request head by
Git plumbing, leaving the prepared stable branch unchanged.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import assets, candidate, preview, release_state, source_equivalence, verify_draft
from .models import CandidateRecord

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SNAPSHOT = re.compile(r"^[0-9a-f]{64}$")
_PREVIEW_VERSION = re.compile(
    r"^(?P<base>(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))"
    r"-(?P<phase>alpha|beta|rc)\.(?P<number>[1-9][0-9]*)$"
)
_STABLE_VERSION = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_PHASE_LABELS = {"alpha": "prerelease:alpha", "beta": "prerelease:beta", "rc": "prerelease:rc"}
_AUTHORIZED_PERMISSIONS = frozenset({"write", "maintain", "admin"})
_IDENTITY_COMMENT_PREFIX = "<!-- gel-candidate-identity: "
_IDENTITY_COMMENT = re.compile(
    r"^<!-- gel-candidate-identity: (?P<payload>\{.*\}) -->$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    """Immutable input accepted by the candidate staging workflow."""

    line: str
    pr_number: int
    base_sha: str
    source_sha: str
    source_snapshot: str
    phase: str | None
    version: str
    channel: str
    build_sha: str | None = None

    def __post_init__(self) -> None:
        major = release_state.parse_line(self.line)
        if (
            isinstance(self.pr_number, bool)
            or not isinstance(self.pr_number, int)
            or self.pr_number < 1
        ):
            raise ValueError(f"invalid release PR number {self.pr_number!r}")
        for name, value in (
            ("base", self.base_sha),
            ("source", self.source_sha),
        ):
            if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
                raise ValueError(f"invalid {name} SHA {value!r}")
        if self.build_sha is None:
            object.__setattr__(self, "build_sha", self.source_sha)
        elif _GIT_SHA.fullmatch(self.build_sha) is None:
            raise ValueError(f"invalid build SHA {self.build_sha!r}")
        if (
            not isinstance(self.source_snapshot, str)
            or _SNAPSHOT.fullmatch(self.source_snapshot) is None
        ):
            raise ValueError(f"invalid source snapshot {self.source_snapshot!r}")
        if self.phase is None:
            preview.stable_version(self.version, major)
            if self.channel != "stable":
                raise ValueError("stable candidates must use the stable channel")
            if self.build_sha != self.source_sha:
                raise ValueError("stable candidate build SHA must equal source SHA")
        else:
            if self.phase not in _PHASE_LABELS:
                raise ValueError(f"unsupported candidate phase {self.phase!r}")
            match = _PREVIEW_VERSION.fullmatch(self.version)
            if match is None:
                raise ValueError(f"unsupported preview version {self.version!r}")
            if int(match.group("base").split(".", 1)[0]) != major:
                raise ValueError(
                    f"line {self.line} major {major} does not match version {self.version}"
                )
            if match.group("phase") != self.phase:
                raise ValueError("candidate phase must match the version")
            if self.channel != "testing":
                raise ValueError("preview candidates must use the testing channel")

    @property
    def tag(self) -> str:
        """Return the Git tag name for this candidate."""

        return f"v{self.version}"

    def as_dict(self) -> dict[str, object]:
        """Serialize only the fields needed to reproduce the candidate."""

        return {
            "line": self.line,
            "pr_number": self.pr_number,
            "base_sha": self.base_sha,
            "source_sha": self.source_sha,
            "source_snapshot": self.source_snapshot,
            "phase": self.phase,
            "version": self.version,
            "channel": self.channel,
            "build_sha": self.build_sha,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CandidateIdentity:
        """Validate an identity read from workflow JSON."""

        if not isinstance(value, Mapping):
            raise ValueError("candidate identity must be a JSON object")
        required = (
            "line",
            "pr_number",
            "base_sha",
            "source_sha",
            "source_snapshot",
            "phase",
            "version",
            "channel",
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"candidate identity is missing fields: {', '.join(missing)}")
        return cls(
            line=value["line"],  # type: ignore[arg-type]
            pr_number=value["pr_number"],  # type: ignore[arg-type]
            base_sha=value["base_sha"],  # type: ignore[arg-type]
            source_sha=value["source_sha"],  # type: ignore[arg-type]
            source_snapshot=value["source_snapshot"],  # type: ignore[arg-type]
            phase=value["phase"],  # type: ignore[arg-type]
            version=value["version"],  # type: ignore[arg-type]
            channel=value["channel"],  # type: ignore[arg-type]
            build_sha=value.get("build_sha"),  # type: ignore[arg-type]
        )


def _coerce_identity(identity: CandidateIdentity | Mapping[str, object]) -> CandidateIdentity:
    if isinstance(identity, CandidateIdentity):
        return identity
    if isinstance(identity, Mapping):
        return CandidateIdentity.from_dict(identity)
    raise ValueError("candidate identity must be a CandidateIdentity or JSON object")


def assert_live_identity(
    identity: CandidateIdentity | Mapping[str, object], live_pr: Mapping[str, object]
) -> None:
    """Require a live pull request to still describe an immutable candidate.

    A staging run may take long enough for the PR head, base, labels, or
    prepared metadata to move.  Stable record publication calls this check
    immediately before pushing the record to the generated PR branch.
    """

    expected = _coerce_identity(identity)
    if not isinstance(live_pr, Mapping):
        raise ValueError("live release PR must be a JSON object")
    base = live_pr.get("base")
    if not isinstance(base, Mapping):
        raise ValueError("live release PR has no base identity")
    base_repo = base.get("repo")
    repository = base_repo.get("full_name") if isinstance(base_repo, Mapping) else None
    if not isinstance(repository, str) or not repository:
        raise ValueError("live release PR has no base repository")
    live = release_state.validate_pr(dict(live_pr), repository)

    # A preview identity carries the selected prerelease version while the
    # generated PR keeps its planned plain stable version.  Compare the
    # prepared PR metadata to that plain base; stable candidates compare the
    # version unchanged.
    prepared_version = expected.version
    if expected.phase is not None:
        prepared_version = expected.version.split("-", 1)[0]
    checks: tuple[tuple[str, object, object], ...] = (
        ("PR number", live.number, expected.pr_number),
        ("release line", live.base_ref, expected.line),
        ("base SHA", live.base_sha, expected.base_sha),
        ("source SHA", live.head_sha, expected.source_sha),
        ("prepared version", _prepared_version(live_pr), prepared_version),
        ("source snapshot", _source_snapshot(live_pr), expected.source_snapshot),
        ("phase", release_state.phase_from_labels(_labels(live_pr)), expected.phase),
    )
    live_build_sha = live_pr.get("build_sha")
    if live_build_sha is not None:
        checks += (("build SHA", live_build_sha, expected.build_sha),)
    live_channel = live_pr.get("channel")
    if live_channel is not None:
        checks += (("channel", live_channel, expected.channel),)
    for name, actual, wanted in checks:
        if actual != wanted:
            raise ValueError(f"live PR {name} {actual!r} does not match candidate {wanted!r}")
    expected_channel = "testing" if expected.phase is not None else "stable"
    if expected.channel != expected_channel:
        raise ValueError(
            f"candidate channel {expected.channel!r} does not match its phase {expected.phase!r}"
        )


def check_stable_merge(
    record: CandidateRecord,
    live_pr: dict,
    merge_sha: str,
    repo: Path,
    live_snapshot: str | None = None,
) -> None:
    """Validate every identity boundary required before a stable merge.

    The candidate record was produced from a reviewed draft on the generated
    PR head. This check re-reads the live PR, the prospective merge tree, both
    Cargo version files, and the draft release before allowing the PR to merge.
    Generated distribution metadata is ignored only by ``source_equivalence``'s
    explicit allowlist.

    ``live_snapshot`` is the meaningful source snapshot of the live PR head,
    computed by the caller with git against the same repository. The gate
    workflow always supplies it, binding the head GitHub reports right now to
    the tested snapshot. The prepared version needs no equivalent check here:
    the record-only successor proof above already pins the live head to the
    exact tested source and record bytes, and the Cargo checks below pin both
    revisions to the record's plain stable version.
    """

    validated = candidate.validate_record(record)
    if validated.phase is not None:
        raise ValueError(
            f"stable merge candidate must have no active phase; record has {validated.phase!r}"
        )
    if not isinstance(repo, Path):
        raise ValueError(f"candidate source repository must be a Path, got {repo!r}")
    if not isinstance(merge_sha, str) or _GIT_SHA.fullmatch(merge_sha) is None:
        raise ValueError(f"prospective merge revision has an invalid SHA {merge_sha!r}")

    live = release_state.validate_pr(live_pr, assets.REPOSITORY)
    active_phase = release_state.phase_from_labels(_labels(live_pr))
    if active_phase is not None:
        raise ValueError(f"stable merge candidate cannot have active phase label {active_phase!r}")

    checks: tuple[tuple[str, object, object], ...] = (
        ("PR number", live.number, validated.pr_number),
        ("release line", live.base_ref, validated.line),
        ("base SHA", live.base_sha, validated.base_sha),
    )
    for name, actual, expected in checks:
        if actual != expected:
            raise ValueError(f"live PR {name} {actual!r} does not match candidate {expected!r}")

    try:
        expected_record = candidate.dump(validated)
        if live.head_sha == validated.source_sha:
            # A retry can find the record already committed by an earlier
            # staging attempt. It is still required to contain the exact
            # record bytes before this equality is accepted.
            source_equivalence.assert_record_present(
                validated.source_sha,
                expected_record,
                repo,
            )
        else:
            # The staging workflow commits the record after recording the
            # tested source SHA. Require that one record-only successor so a
            # newer source or an unrelated generated commit cannot pass.
            source_equivalence.assert_record_successor(
                validated.source_sha,
                live.head_sha,
                expected_record,
                repo,
            )
    except source_equivalence.SourceDrift as error:
        raise ValueError(f"live PR source SHA check failed: {error}") from error

    if live_snapshot is not None and live_snapshot != validated.source_snapshot:
        raise ValueError(
            f"live PR source snapshot {live_snapshot!r} does not match candidate "
            f"{validated.source_snapshot!r}"
        )

    try:
        source_equivalence.assert_snapshot(validated.source_snapshot, validated.source_sha, repo)
        source_equivalence.assert_merge_equivalent(validated.source_sha, merge_sha, repo)
    except source_equivalence.SourceDrift as error:
        raise ValueError(f"stable merge source check failed: {error}") from error

    # The source-equivalence check includes Cargo files, but checking both
    # revisions explicitly also proves that the files resolve to the plain
    # stable version expected by the candidate record.
    verify_draft.check_cargo_version(validated.source_sha, validated.version, repo)
    verify_draft.check_cargo_version(merge_sha, validated.version, repo)

    expected_identity = {
        "line": validated.line,
        "pr_number": validated.pr_number,
        "base_sha": validated.base_sha,
        "source_sha": validated.source_sha,
        "build_sha": validated.build_sha,
        "source_snapshot": validated.source_snapshot,
        "phase": None,
        "version": validated.version,
        "channel": "stable",
    }
    with tempfile.TemporaryDirectory(prefix="stable-merge-gate-") as directory:
        try:
            verify_draft.verify(
                validated,
                Path(directory),
                repo=assets.REPOSITORY,
                verify_attestations_flag=True,
                expected_identity=expected_identity,
            )
        except verify_draft.DraftVerificationError as error:
            raise ValueError(f"stable candidate draft verification failed: {error}") from error


def _body_identity(release: Mapping[str, object]) -> Mapping[str, object]:
    """Read the hidden or legacy candidate identity from a release body.

    New drafts show their changelog and carry the machine identity in an HTML
    comment. Published candidates created before that format remain valid for
    idempotent retries through the legacy whole-body JSON fallback.
    """

    tag = release.get("tag_name")
    body = release.get("body")
    if not isinstance(body, str) or not body.strip():
        raise ValueError(f"release {tag!r} has no candidate identity body")
    comments = list(_IDENTITY_COMMENT.finditer(body))
    if len(comments) > 1:
        raise ValueError(f"release {tag!r} body has more than one candidate identity comment")
    payload = comments[0].group("payload") if comments else body
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"release {tag!r} body is not candidate identity JSON: {error}") from error
    if not isinstance(parsed, Mapping):
        raise ValueError(f"release {tag!r} body is not a JSON object")
    identity = parsed.get("candidate_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"release {tag!r} body has no 'candidate_identity' object")
    return identity


@dataclass(frozen=True, slots=True)
class DraftSelection:
    """The single draft release a staging run may write candidate assets to.

    ``reusable`` marks the exact retry draft, whose identity already equals
    the selected identity.  Otherwise the draft is a same-line, same-PR
    predecessor that a source refresh supersedes, and ``stale_build_sha``
    names the build commit its assets and tag were staged from.
    ``stale_base_sha`` lets the workflow prove that a changed release-line
    base moved forward before it replaces that predecessor.
    """

    release: Mapping[str, object]
    reusable: bool
    stale_build_sha: str | None
    stale_base_sha: str | None

    def as_dict(self) -> dict[str, object]:
        """Serialize the fields the staging workflow consumes."""

        return {
            "id": self.release.get("id"),
            "stale_build_sha": "" if self.reusable else (self.stale_build_sha or ""),
            "stale_base_sha": "" if self.reusable else (self.stale_base_sha or ""),
        }


def select_draft(
    releases: list[Mapping[str, object]],
    identity: CandidateIdentity | Mapping[str, object],
) -> DraftSelection | None:
    """Select the one unpublished draft this candidate may write assets to.

    Releases for other candidate tags are unrelated, even on the same release
    line.  A release carrying this exact candidate tag is either the exact
    retry draft or, under the line mutation lock, a same-line and same-PR
    draft whose source was refreshed and may be superseded.  Published
    releases, mismatched lines or PRs, and malformed records fail closed. A
    changed base is returned to the workflow, which must prove that the old
    base is an ancestor of the current release-line base before replacement.
    """

    expected = _coerce_identity(identity)
    if not isinstance(releases, list):
        raise ValueError("GitHub releases must be a list")
    expected_tag = expected.tag
    expected_prerelease = expected.phase is not None
    selected: DraftSelection | None = None
    for index, release in enumerate(releases):
        if not isinstance(release, Mapping):
            raise ValueError(f"release entry {index} is not an object")
        if release.get("tag_name") != expected_tag:
            continue
        if release.get("name") != expected_tag:
            raise ValueError(f"release for candidate tag {expected_tag} has a tag/name mismatch")
        if release.get("draft") is not True:
            raise ValueError(f"release {expected_tag} is published; refusing to replace it")
        try:
            actual = CandidateIdentity.from_dict(_body_identity(release))
        except (TypeError, ValueError) as error:
            raise ValueError(f"draft {expected_tag} has an invalid candidate identity") from error
        reusable = actual.as_dict() == expected.as_dict()
        if not reusable:
            if actual.line != expected.line or actual.pr_number != expected.pr_number:
                raise ValueError(
                    f"draft {expected_tag} candidate identity does not match authorized line/PR"
                )
        if release.get("prerelease") is not expected_prerelease:
            raise ValueError(f"draft {expected_tag} prerelease flag does not match candidate phase")
        if selected is not None:
            raise ValueError(f"more than one draft exists for {expected_tag}")
        selected = DraftSelection(
            release=release,
            reusable=reusable,
            stale_build_sha=None if reusable else actual.build_sha,
            stale_base_sha=(
                None if reusable or actual.base_sha == expected.base_sha else actual.base_sha
            ),
        )
    return selected


def candidate_identity_body(identity: CandidateIdentity | Mapping[str, object]) -> str:
    """Serialize the legacy whole-body identity format."""

    return json.dumps({"candidate_identity": _coerce_identity(identity).as_dict()}, sort_keys=True)


def _candidate_changelog_section(identity: CandidateIdentity, changelog: str) -> str:
    """Select the release's version section from a Knope changelog."""

    match = _PREVIEW_VERSION.fullmatch(identity.version)
    version = match.group("base") if match is not None else identity.version
    heading = re.compile(rf"^##[ \t]+{re.escape(version)}(?:[ \t].*)?$")
    section_heading = re.compile(r"^##[ \t]+")
    lines = changelog.splitlines()
    starts = [index for index, line in enumerate(lines) if heading.fullmatch(line)]
    if len(starts) != 1:
        raise ValueError(
            f"changelog must contain exactly one section for {version}, found {len(starts)}"
        )
    start = starts[0]
    end = next(
        (index for index in range(start + 1, len(lines)) if section_heading.match(lines[index])),
        len(lines),
    )
    section = "\n".join(lines[start:end]).strip()
    if "\n" not in section or not section.split("\n", 1)[1].strip():
        raise ValueError(f"changelog section for {version} has no release notes")
    if _IDENTITY_COMMENT_PREFIX in section:
        raise ValueError(f"changelog section for {version} contains a reserved identity marker")
    return section


def candidate_release_body(
    identity: CandidateIdentity | Mapping[str, object], changelog: str
) -> str:
    """Render visible release notes plus a hidden retry identity."""

    validated = _coerce_identity(identity)
    notes = _candidate_changelog_section(validated, changelog)
    payload = candidate_identity_body(validated)
    return f"{notes}\n\n{_IDENTITY_COMMENT_PREFIX}{payload} -->\n"


def _labels(pr: Mapping[str, object]) -> list[str]:
    raw = pr.get("labels", [])
    if not isinstance(raw, list):
        raise ValueError("live PR labels must be a list")
    labels: list[str] = []
    for entry in raw:
        if isinstance(entry, str):
            labels.append(entry)
        elif isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
            labels.append(entry["name"])
        else:
            raise ValueError("live PR labels must contain names")
    return labels


def _prepared_version(pr: Mapping[str, object]) -> str:
    """Read the prepared stable version attached to a live PR payload.

    The controller, candidate, and publish workflows each enrich the GitHub
    pull request JSON with a top-level ``prepared_version`` computed from
    ``cargo metadata``.  No caller produces any other spelling or nesting.
    """

    value = pr.get("prepared_version")
    if not isinstance(value, str) or not value:
        raise ValueError(
            "live PR has no top-level string 'prepared_version'; enrich the PR JSON "
            "with the prepared Cargo version before using it as a candidate source"
        )
    return value


def _source_snapshot(pr: Mapping[str, object]) -> str:
    """Read the meaningful source snapshot attached to a live PR payload."""

    value = pr.get("source_snapshot")
    if not isinstance(value, str) or _SNAPSHOT.fullmatch(value) is None:
        raise ValueError(
            "live PR has no top-level 'source_snapshot' holding a 64 hex-character "
            "meaningful-tree digest"
        )
    return value


def _published_candidate_record(release: Mapping[str, object]) -> Mapping[str, object] | None:
    """Return the candidate record the controller attached to a release.

    ``release-controller.yml`` downloads each published release's
    ``gel-candidate.json`` asset and sets it on that release as ``.candidate``.
    A published release without the asset predates this pipeline and therefore
    carries no record at all.
    """

    record = release.get("candidate")
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise ValueError(
            f"release {release.get('tag_name')!r} candidate record is not a JSON object"
        )
    return record


def published_snapshots(releases: list[dict]) -> set[tuple[str, str]]:
    """Return phase/source pairs from already published preview releases.

    The design guarantees that a given phase and source snapshot publishes at
    most once.  A published prerelease whose attached candidate record cannot
    be read leaves that guarantee unprovable, so it fails the run instead of
    being skipped: skipping it would let the same rebuild publish again under
    the next suffix.  A published prerelease with no attached record at all
    predates this pipeline (it has no ``gel-candidate.json`` asset, so the
    controller attached nothing) and reserves no snapshot.
    """

    if not isinstance(releases, list):
        raise ValueError("published releases must be a list")
    result: set[tuple[str, str]] = set()
    for index, release in enumerate(releases):
        if not isinstance(release, Mapping):
            raise ValueError(f"release entry {index} is not an object")
        if release.get("draft") is True or release.get("prerelease") is not True:
            continue
        tag = release.get("tag_name")
        record = _published_candidate_record(release)
        if record is None:
            continue
        phase = record.get("phase")
        if not isinstance(phase, str) or phase not in _PHASE_LABELS:
            raise ValueError(
                f"published prerelease {tag!r} candidate record has phase {phase!r}; "
                f"expected one of {sorted(_PHASE_LABELS)}"
            )
        snapshot = record.get("source_snapshot")
        if not isinstance(snapshot, str) or _SNAPSHOT.fullmatch(snapshot) is None:
            raise ValueError(
                f"published prerelease {tag!r} candidate record has 'source_snapshot' "
                f"{snapshot!r}; expected a 64 hex-character meaningful-tree digest"
            )
        result.add((phase, snapshot))
    return result


def _published_tag_inventory(tags: list[str], releases: list[dict]) -> list[str]:
    """Ignore Git tags that are still attached only to draft releases."""

    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError("published tags must be a list of strings")
    if not isinstance(releases, list):
        raise ValueError("published releases must be a list")
    draft_tags = {
        release.get("tag_name")
        for release in releases
        if isinstance(release, Mapping)
        and release.get("draft") is True
        and isinstance(release.get("tag_name"), str)
    }
    published = {
        release["tag_name"]
        for release in releases
        if isinstance(release, Mapping)
        and release.get("draft") is not True
        and isinstance(release.get("tag_name"), str)
    }
    return [tag for tag in tags if tag not in draft_tags or tag in published]


def _stable_record_staged(
    identity: CandidateIdentity,
    live_pr: Mapping[str, object],
    checkout: Path,
) -> bool:
    """Whether the live PR head already carries this exact stable record.

    The controller recomputes the prepared version and the meaningful source
    snapshot from the current PR head. Staging commits the record as a
    record-only successor and the record path is excluded from the snapshot,
    so a staged head yields the same identity as the tested source. A matching
    record successor at the head is therefore the staging fixed point: the
    candidate exists, the stable merge gate owns the rest, and dispatching
    another build would never terminate.
    """

    head = live_pr.get("head")
    head_sha = head.get("sha") if isinstance(head, Mapping) else None
    if not isinstance(head_sha, str) or _GIT_SHA.fullmatch(head_sha) is None:
        return False
    try:
        record_bytes = _git_blob(checkout, head_sha, str(candidate.CANDIDATE_PATH))
        record = candidate.validate_record(candidate.load_bytes(record_bytes))
    except (OSError, ValueError):
        return False
    if record.phase is not None or record.build_sha != record.source_sha:
        return False
    try:
        # The staged head is a record-only successor of the record's own
        # tested source, never of the head-derived identity source.
        source_equivalence.assert_record_successor(
            record.source_sha, head_sha, record_bytes, checkout
        )
    except (OSError, ValueError):
        return False
    return (
        record.line == identity.line
        and record.pr_number == identity.pr_number
        and record.base_sha == identity.base_sha
        and record.source_snapshot == identity.source_snapshot
        and record.version == identity.version
    )


def resolve_candidate(
    pr: release_state.ReleasePr,
    live_pr: dict,
    tags: list[str],
    releases: list[dict],
    checkout: Path = Path("."),
) -> CandidateIdentity | None:
    """Resolve the current live PR into one immutable candidate identity.

    ``live_pr`` is validated again so stale event payload labels cannot affect
    the result.  The original ``pr`` supplies the trusted PR number and line;
    a moved release-line base is rejected because a prepared candidate tied to
    the old base cannot be reused.  A newer head is allowed and becomes the
    candidate's source SHA.

    A stable candidate whose exact record is already committed to the PR head
    returns ``None``: that head is the staging fixed point and re-staging it
    would loop. ``checkout`` is the working copy used to inspect that head.
    """

    if not isinstance(pr, release_state.ReleasePr):
        raise ValueError("release PR identity must be a ReleasePr")
    live = release_state.validate_pr(live_pr, pr.repository)
    if live.number != pr.number:
        raise ValueError(f"live PR #{live.number} does not match requested PR #{pr.number}")
    for field in ("base_ref", "head_ref", "repository", "major"):
        if getattr(live, field) != getattr(pr, field):
            raise ValueError(
                f"live PR {field} {getattr(live, field)!r} does not match requested identity"
            )
    if live.base_sha != pr.base_sha:
        raise ValueError(
            f"release line base moved from {pr.base_sha} to {live.base_sha}; refresh the PR"
        )
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError("published tags must be a list of strings")

    version = _prepared_version(live_pr)
    snapshot = _source_snapshot(live_pr)
    phase = release_state.phase_from_labels(_labels(live_pr))
    stable = preview.stable_version(version, live.major)
    published = published_snapshots(releases)
    if phase is None:
        identity = CandidateIdentity(
            line=live.base_ref,
            pr_number=live.number,
            base_sha=live.base_sha,
            source_sha=live.head_sha,
            source_snapshot=snapshot,
            phase=None,
            version=stable,
            channel="stable",
            build_sha=live.head_sha,
        )
        if _stable_record_staged(identity, live_pr, Path(checkout)):
            return None
        # Fail before any build, staging, gate, or merge when the prepared
        # version already belongs to a published release. Unpublished tags
        # stay permitted so a retry after a failed publication can re-stage.
        published_tags = {
            release.get("tag_name")
            for release in releases
            if isinstance(release, Mapping) and release.get("draft") is not True
        }
        if identity.tag in published_tags:
            raise ValueError(
                f"prepared version {identity.version} is already published as "
                f"{identity.tag}; a release line must prepare an untagged version"
            )
        return identity

    selected = preview.next_preview_version(
        stable,
        phase,
        _published_tag_inventory(tags, releases),
        published,
        snapshot,
    )
    if selected is None:
        return None
    return CandidateIdentity(
        line=live.base_ref,
        pr_number=live.number,
        base_sha=live.base_sha,
        source_sha=live.head_sha,
        source_snapshot=snapshot,
        phase=phase,
        version=selected,
        channel="testing",
        build_sha=live.head_sha,
    )


def _timeline_sort(timeline: list[dict]) -> list[dict]:
    if not timeline:
        return []
    parsed: list[tuple[datetime, int, dict]] = []
    for index, entry in enumerate(timeline):
        created = entry.get("created_at")
        if not isinstance(created, str):
            return timeline
        try:
            parsed.append((datetime.fromisoformat(created.replace("Z", "+00:00")), index, entry))
        except ValueError:
            return timeline
    parsed.sort(key=lambda item: (item[0], item[1]))
    return [entry for _, _, entry in parsed]


def _timeline_label(entry: Mapping[str, object]) -> str | None:
    label = entry.get("label")
    if isinstance(label, Mapping):
        label = label.get("name")
    return label if isinstance(label, str) else None


def _timeline_actor(entry: Mapping[str, object]) -> tuple[str | None, str | None]:
    actor = entry.get("actor", entry.get("user"))
    if not isinstance(actor, Mapping):
        return None, None
    login = actor.get("login")
    actor_type = actor.get("type")
    return (
        login if isinstance(login, str) else None,
        actor_type if isinstance(actor_type, str) else None,
    )


def _permission(value: object) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("permission")
    return value.lower() if isinstance(value, str) else None


def timeline_actors(timeline: list[dict]) -> list[str]:
    """Return the human actors in a PR timeline, excluding bots.

    Bot transitions never carry phase authority, so their permission lookups
    are skipped entirely.
    """

    if not isinstance(timeline, list):
        raise ValueError("PR timeline must be a list")
    actors: set[str] = set()
    for entry in timeline:
        if not isinstance(entry, Mapping):
            raise ValueError("PR timeline entries must be objects")
        login, actor_type = _timeline_actor(entry)
        if not login or actor_type and actor_type.lower() == "bot" or login.endswith("[bot]"):
            continue
        actors.add(login)
    return sorted(actors)


def phase_authorized(timeline: list[dict], phase: str, permissions: Mapping[str, object]) -> bool:
    """Check the latest active phase label was added by an authorized user.

    The event actor that triggered a controller run is intentionally absent
    from this API.  Authorization comes only from the most recent label
    transition in the PR timeline and a fresh repository permission lookup.
    """

    if phase not in _PHASE_LABELS:
        raise ValueError(f"unsupported preview phase {phase!r}")
    if not isinstance(timeline, list):
        raise ValueError("PR timeline must be a list")
    if not isinstance(permissions, Mapping):
        raise ValueError("phase permissions must be an object")
    label_name = _PHASE_LABELS[phase]
    latest_actor: tuple[str | None, str | None] | None = None
    for entry in _timeline_sort(timeline):
        if not isinstance(entry, Mapping):
            raise ValueError("PR timeline entries must be objects")
        if _timeline_label(entry) != label_name:
            continue
        event = entry.get("event", entry.get("type"))
        if event == "unlabeled":
            latest_actor = None
        elif event == "labeled":
            latest_actor = _timeline_actor(entry)
    if latest_actor is None:
        return False
    login, actor_type = latest_actor
    if not login or actor_type and actor_type.lower() == "bot" or login.endswith("[bot]"):
        return False
    return _permission(permissions.get(login)) in _AUTHORIZED_PERMISSIONS


def _gh_json(*args: str) -> object:
    """Run ``gh`` and decode its JSON output.

    Publication is deliberately kept at the GitHub API edge.  Keeping this
    small wrapper in the module also gives the publication tests a single
    mutation boundary to replace, while the workflow uses the exact same
    code path against GitHub.
    """

    try:
        completed = subprocess.run(
            ["gh", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
        )
        raise ValueError(f"gh {' '.join(args)} failed: {detail}") from error
    output = completed.stdout.strip()
    if not output:
        return None
    decoder = json.JSONDecoder()
    values: list[object] = []
    position = 0
    try:
        while position < len(output):
            while position < len(output) and output[position].isspace():
                position += 1
            if position >= len(output):
                break
            value, position = decoder.raw_decode(output, position)
            values.append(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"gh {' '.join(args)} returned invalid JSON: {error}") from error
    return values[0] if len(values) == 1 else values


def _gh_mutate(
    method: str,
    path: str,
    fields: Mapping[str, object] | None = None,
) -> object:
    """Apply one explicit GitHub API mutation.

    The publication functions call this only for creating an immutable tag or
    changing an existing draft's publication state.  In particular, there is
    no upload, delete, replacement, or force-update operation here.
    """

    if not isinstance(method, str) or not method:
        raise ValueError(f"invalid GitHub API method {method!r}")
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"invalid GitHub API path {path!r}")
    argv = ["api", "-X", method, path]
    for name, value in (fields or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"invalid GitHub API field name {name!r}")
        if isinstance(value, bool):
            argv.extend(["-F", f"{name}={'true' if value else 'false'}"])
        elif isinstance(value, int):
            argv.extend(["-F", f"{name}={value}"])
        elif isinstance(value, str):
            argv.extend(["-f", f"{name}={value}"])
        else:
            raise ValueError(f"GitHub API field {name!r} has unsupported value {value!r}")
    return _gh_json(*argv)


def _record_identity(
    record: CandidateRecord | Mapping[str, object],
) -> tuple[CandidateRecord, dict[str, object]]:
    """Return a validated record and its wire-compatible candidate identity."""

    try:
        validated = candidate.validate_record(record)
    except (TypeError, ValueError) as error:
        raise ValueError(f"candidate record is invalid: {error}") from error
    identity = {
        "line": validated.line,
        "pr_number": validated.pr_number,
        "base_sha": validated.base_sha,
        "source_sha": validated.source_sha,
        "build_sha": validated.build_sha,
        "source_snapshot": validated.source_snapshot,
        "phase": validated.phase,
        "version": validated.version,
        "channel": "testing" if validated.phase is not None else "stable",
    }
    return validated, identity


def _assert_record_identity(
    record: CandidateRecord | Mapping[str, object],
    expected: CandidateIdentity,
) -> CandidateRecord:
    validated, actual = _record_identity(record)
    if actual != expected.as_dict():
        differences = [
            f"{name}={actual[name]!r} (expected {wanted!r})"
            for name, wanted in expected.as_dict().items()
            if actual.get(name) != wanted
        ]
        raise ValueError(
            "candidate record identity does not match selected identity: " + ", ".join(differences)
        )
    return validated


def _assert_release_shape(
    release: Mapping[str, object],
    expected: CandidateIdentity,
    record: CandidateRecord,
    *,
    allow_published: bool,
) -> bool:
    """Validate release identity, state, and any API-listed asset metadata."""

    if not isinstance(release, Mapping):
        raise ValueError("GitHub release is not an object")
    release_id = release.get("id")
    if isinstance(release_id, bool) or not isinstance(release_id, int):
        raise ValueError("GitHub release has no valid release id")
    if release_id != record.draft_release_id:
        raise ValueError(
            f"release id {release_id} does not match candidate draft {record.draft_release_id}"
        )
    if release.get("tag_name") != expected.tag or release.get("name") != expected.tag:
        raise ValueError(f"release tag/name does not match candidate {expected.tag}")
    draft = release.get("draft")
    if not isinstance(draft, bool):
        raise ValueError(f"release {expected.tag} has no valid draft state")
    if not draft and not allow_published:
        raise ValueError(f"release {expected.tag} is already published")
    expected_prerelease = expected.phase is not None
    if release.get("prerelease") is not expected_prerelease:
        raise ValueError(
            f"release {expected.tag} prerelease state {release.get('prerelease')!r} "
            f"does not match candidate phase {expected.phase!r}"
        )
    try:
        release_identity = CandidateIdentity.from_dict(_body_identity(release))
    except (TypeError, ValueError) as error:
        raise ValueError(f"release {expected.tag} has an invalid candidate identity") from error
    if release_identity.as_dict() != expected.as_dict():
        raise ValueError(
            f"release {expected.tag} candidate identity does not match selected identity"
        )

    listed = release.get("assets")
    if listed is not None:
        if not isinstance(listed, list):
            raise ValueError(f"release {expected.tag} assets are not a list")
        normalized: list[dict] = []
        seen_names: set[str] = set()
        seen_ids: set[int] = set()
        for index, item in enumerate(listed):
            if not isinstance(item, Mapping):
                raise ValueError(f"release {expected.tag} asset {index} is not an object")
            name = item.get("name")
            asset_id = item.get("id")
            size = item.get("size")
            if not isinstance(name, str) or not name:
                raise ValueError(f"release {expected.tag} asset {index} has an invalid name")
            if isinstance(asset_id, bool) or not isinstance(asset_id, int) or asset_id <= 0:
                raise ValueError(f"release {expected.tag} asset {name} has an invalid id")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError(f"release {expected.tag} asset {name} has an invalid size")
            if name in seen_names or asset_id in seen_ids:
                raise ValueError(f"release {expected.tag} has duplicate asset metadata")
            seen_names.add(name)
            seen_ids.add(asset_id)
            normalized.append({"id": asset_id, "name": name, "size": size})
        try:
            verify_draft.check_inventory(normalized, expected.version, phase=expected.phase)
        except verify_draft.DraftVerificationError as error:
            raise ValueError(f"release {expected.tag} inventory mismatch: {error}") from error
        by_name = {item["name"]: item for item in normalized}
        for entry in record.assets:
            remote = by_name.get(entry.name)
            if remote is None:
                raise ValueError(f"release {expected.tag} is missing recorded asset {entry.name}")
            if remote["id"] != entry.id or remote["size"] != entry.size:
                raise ValueError(f"release {expected.tag} asset {entry.name} metadata changed")
    return draft


def _verify_draft_before_publication(
    record: CandidateRecord,
    expected: CandidateIdentity,
    *,
    expected_tag_target: str | None = None,
) -> None:
    """Read every draft asset through the authenticated API before publishing."""

    with tempfile.TemporaryDirectory(prefix="release-publication-verify-") as directory:
        try:
            verify_draft.verify(
                record,
                Path(directory),
                assets.REPOSITORY,
                True,
                expected_identity=expected.as_dict(),
                expected_tag_target=expected_tag_target,
            )
        except verify_draft.DraftVerificationError as error:
            raise ValueError(f"draft verification failed: {error}") from error


def _verify_published_asset_bytes(
    record: CandidateRecord,
    release: Mapping[str, object],
) -> None:
    """Verify bytes available from an already published release retry."""

    # A GitHub release payload lists asset ids but never carries bytes, so
    # every recorded asset is downloaded through the authenticated API before
    # an already published retry is treated as successful.  A payload that
    # lists no assets carries nothing to read back here; ``_assert_release_shape``
    # has already bound the release id, tag, and candidate identity, and this
    # path makes no mutation.
    listed = release.get("assets")
    if not isinstance(listed, list) or not listed:
        return
    by_name: dict[str, Mapping[str, object]] = {
        item["name"]: item
        for item in listed
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    names = [entry.name for entry in record.assets]
    if record.phase is not None:
        names.append(candidate.PREVIEW_RECORD_NAME)
    with tempfile.TemporaryDirectory(prefix="release-published-verify-") as directory:
        for name in names:
            item = by_name.get(name)
            asset_id = item.get("id") if item is not None else None
            if isinstance(asset_id, bool) or not isinstance(asset_id, int) or asset_id <= 0:
                raise ValueError(f"published release is missing asset id for {name}")
            path = Path(directory) / name.replace("/", "_")
            try:
                verify_draft.download_asset(asset_id, path, assets.REPOSITORY)
            except (OSError, subprocess.CalledProcessError) as error:
                raise ValueError(f"could not download published asset {name}: {error}") from error
            actual = path.read_bytes()
            if name == candidate.PREVIEW_RECORD_NAME:
                try:
                    candidate.verify_record_bytes(record, actual)
                except candidate.CandidateMismatch as error:
                    raise ValueError(f"published candidate bytes changed: {error}") from error
                continue
            expected = next(entry for entry in record.assets if entry.name == name)
            if len(actual) != expected.size:
                raise ValueError(f"published asset {name} size changed")
            if hashlib.sha256(actual).hexdigest() != expected.sha256:
                raise ValueError(f"published asset {name} SHA-256 changed")
            if hashlib.blake2b(actual, digest_size=64).hexdigest() != expected.blake2b512:
                raise ValueError(f"published asset {name} BLAKE2b changed")


def _tag_target(tag: str) -> str | None:
    """Resolve the actual GitHub ref target, including annotated tags."""

    try:
        return verify_draft.resolve_tag_commit(tag, assets.REPOSITORY)
    except verify_draft.DraftVerificationError as error:
        raise ValueError(f"could not verify tag {tag}: {error}") from error


def _ensure_tag(tag: str, target: str) -> None:
    existing = _tag_target(tag)
    if existing is not None:
        if existing.lower() != target.lower():
            raise ValueError(f"tag {tag} points at {existing}, expected immutable target {target}")
        return
    _gh_mutate(
        "POST",
        f"/repos/{assets.REPOSITORY}/git/refs",
        {"ref": f"refs/tags/{tag}", "sha": target},
    )


def _publish_release(
    record: CandidateRecord,
    *,
    prerelease: bool,
    make_latest: bool,
) -> None:
    # ``make_latest`` is a string enum ("true"|"false"|"legacy") in the GitHub
    # release API, not a JSON boolean; ``draft`` and ``prerelease`` are the
    # genuine booleans. ``_gh_mutate`` encodes str values literally.
    _gh_mutate(
        "PATCH",
        f"/repos/{assets.REPOSITORY}/releases/{record.draft_release_id}",
        {
            "tag_name": record.tag,
            "draft": False,
            "prerelease": prerelease,
            "make_latest": "true" if make_latest else "false",
        },
    )


def _preview_authorized(live_pr: Mapping[str, object], phase: str) -> None:
    """Require fresh phase authorization computed from live timeline state.

    The timeline and permission lookup must come from the caller's fresh
    repository read; there is no caller-supplied approval override.
    """

    timeline = live_pr.get("timeline")
    permissions = live_pr.get("permissions")
    if not isinstance(timeline, list) or not isinstance(permissions, Mapping):
        raise ValueError(f"preview phase {phase} has no fresh authorization proof")
    if not phase_authorized(timeline, phase, permissions):
        raise ValueError(f"preview phase {phase} is not authorized by a maintainer")


def _flatten_api_pages(value: object) -> list[object]:
    if not isinstance(value, list):
        return [value]
    entries: list[object] = []
    for page in value:
        entries.extend(page if isinstance(page, list) else [page])
    return entries


def _cargo_version_at_revision(revision: str, repo: Path = Path(".")) -> str:
    try:
        document = tomllib.loads(_run_git(repo, "show", f"{revision}:Cargo.toml"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"could not read Cargo.toml at {revision}: {error}") from error
    package = document.get("package")
    if not isinstance(package, Mapping) or package.get("name") != "gel-cli":
        raise ValueError(f"Cargo.toml at {revision} has no gel-cli package")
    version = package.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"Cargo.toml at {revision} has no gel-cli version")
    return version


def fetch_live_preview_pr(
    identity: CandidateIdentity,
    _initial: Mapping[str, object],
) -> Mapping[str, object]:
    """Fetch and enrich the PR state used immediately before preview mutation."""

    raw = _gh_json(
        "api",
        f"/repos/{assets.REPOSITORY}/pulls/{identity.pr_number}",
    )
    if not isinstance(raw, Mapping):
        raise ValueError("fresh preview PR response is not an object")
    fresh = dict(raw)
    head = fresh.get("head")
    if not isinstance(head, Mapping) or not isinstance(head.get("sha"), str):
        raise ValueError("fresh preview PR has no head SHA")
    source_sha = head["sha"]
    if _GIT_SHA.fullmatch(source_sha) is None:
        raise ValueError(f"fresh preview PR has invalid head SHA {source_sha!r}")
    fresh["prepared_version"] = _cargo_version_at_revision(source_sha)
    try:
        fresh["source_snapshot"] = source_equivalence.meaningful_tree(source_sha, Path("."))
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise ValueError(f"could not compute fresh preview source snapshot: {error}") from error

    timeline_payload = _gh_json(
        "api",
        "--paginate",
        "--slurp",
        f"/repos/{assets.REPOSITORY}/issues/{identity.pr_number}/timeline",
    )
    timeline = [
        entry for entry in _flatten_api_pages(timeline_payload) if isinstance(entry, Mapping)
    ]
    permissions: dict[str, str] = {}
    for login in timeline_actors(timeline):
        permission = _gh_json(
            "api",
            f"/repos/{assets.REPOSITORY}/collaborators/{login}/permission",
        )
        if isinstance(permission, Mapping) and isinstance(permission.get("permission"), str):
            permissions[login] = permission["permission"]
    fresh["timeline"] = timeline
    fresh["permissions"] = permissions
    return fresh


def publish_preview(
    identity: CandidateIdentity | Mapping[str, object],
    record: CandidateRecord | Mapping[str, object],
    live_pr: Mapping[str, object],
    release: Mapping[str, object],
    *,
    refresh_live_pr: Callable[[CandidateIdentity, Mapping[str, object]], Mapping[str, object]]
    | None = None,
) -> None:
    """Publish an already verified preview draft after fresh identity checks.

    The build workflow owns all distribution bytes.  This function only
    checks those bytes and the live PR/release state, creates the immutable
    version tag at ``build_sha`` when needed, and flips the existing draft to
    a prerelease.  A matching published release is an idempotent success.
    """

    expected = _coerce_identity(identity)
    if expected.phase is None:
        raise ValueError("preview publication requires an active phase")
    if expected.build_sha == expected.source_sha:
        raise ValueError("preview publication requires a derived build SHA")
    validated = _assert_record_identity(record, expected)
    if not isinstance(live_pr, Mapping):
        raise ValueError("live release PR is not an object")
    if not isinstance(release, Mapping):
        raise ValueError("GitHub release is not an object")
    try:
        assert_live_identity(expected, dict(live_pr))
    except ValueError as error:
        raise ValueError(f"preview live identity rejected: {error}") from error
    _preview_authorized(live_pr, expected.phase)
    draft = _assert_release_shape(release, expected, validated, allow_published=True)

    target = _tag_target(expected.tag)
    if target is not None and target.lower() != expected.build_sha.lower():
        raise ValueError(
            f"preview tag {expected.tag} points at {target}, expected build SHA "
            f"{expected.build_sha}"
        )
    if not draft:
        _verify_published_asset_bytes(validated, release)
        print(
            f"release publication line={expected.line} pr={expected.pr_number} "
            f"phase={expected.phase} "
            f"source_sha={expected.source_sha} version={expected.version} "
            f"draft_release_id={validated.draft_release_id} already published"
        )
        return

    # Draft API readback is the final byte/draft gate before either mutation.
    _verify_draft_before_publication(validated, expected)
    if refresh_live_pr is not None:
        try:
            live_pr = refresh_live_pr(expected, live_pr)
            assert_live_identity(expected, dict(live_pr))
            _preview_authorized(live_pr, expected.phase)
        except ValueError as error:
            raise ValueError(f"preview live state changed before mutation: {error}") from error
    _ensure_tag(expected.tag, expected.build_sha)
    _publish_release(validated, prerelease=True, make_latest=False)
    print(
        f"release publication line={expected.line} pr={expected.pr_number} phase={expected.phase} "
        f"source_sha={expected.source_sha} version={expected.version} "
        f"draft_release_id={validated.draft_release_id} published"
    )


def _merged_pr_from_release(
    record: CandidateRecord,
    release: Mapping[str, object],
) -> Mapping[str, object] | None:
    for key in ("merged_pr", "pull_request", "merge_pr", "pr"):
        if key not in release:
            continue
        value = release[key]
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError(f"release {record.tag} has an invalid merged PR payload")
        return value
    try:
        payload = _gh_json(
            "api",
            f"/repos/{assets.REPOSITORY}/pulls/{record.pr_number}",
        )
    except ValueError as error:
        raise ValueError(f"could not fetch merged PR #{record.pr_number}: {error}") from error
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError(f"merged PR #{record.pr_number} response is not an object")
    return payload


def _matching_merge_pr(
    record: CandidateRecord,
    line_push_sha: str,
    merged_pr: Mapping[str, object] | None,
) -> bool:
    """Check whether this line push is the candidate PR merge.

    A normal backport push returns ``False`` and is a successful no-op.  Once
    the PR number and merge commit identify this candidate, malformed branch,
    repository, or head identity is a hard rejection.
    """

    if merged_pr is None:
        return False
    number = merged_pr.get("number")
    merge_commit = merged_pr.get("merge_commit_sha", merged_pr.get("merge_sha"))
    if isinstance(number, bool) or number != record.pr_number or merge_commit != line_push_sha:
        return False
    base = merged_pr.get("base")
    head = merged_pr.get("head")
    if not isinstance(base, Mapping) or not isinstance(head, Mapping):
        raise ValueError(f"merged PR #{record.pr_number} has incomplete branch identity")
    if base.get("ref") != record.line:
        raise ValueError(f"merged PR #{record.pr_number} base does not match {record.line}")
    expected_head = release_state.expected_head(release_state.parse_line(record.line))
    if head.get("ref") != expected_head:
        raise ValueError(f"merged PR #{record.pr_number} head does not match {expected_head}")
    for side, value in (("base", base), ("head", head)):
        repo = value.get("repo")
        full_name = repo.get("full_name") if isinstance(repo, Mapping) else None
        if full_name != assets.REPOSITORY:
            raise ValueError(f"merged PR #{record.pr_number} has a forked {side} repository")
    head_sha = head.get("sha")
    if not isinstance(head_sha, str) or _GIT_SHA.fullmatch(head_sha) is None:
        raise ValueError(f"merged PR #{record.pr_number} has an invalid head SHA")
    if head_sha != record.source_sha:
        # The stable stage commits packaging/release-candidate.json as one
        # record-only successor of the tested source.  GitHub reports that
        # successor as the merged PR head, while the record intentionally
        # retains the original source SHA.  Verify that exact successor when
        # the local checkout contains it.
        try:
            source_equivalence.assert_record_successor(
                record.source_sha,
                head_sha,
                candidate.dump(record),
                Path("."),
            )
        except source_equivalence.SourceDrift as error:
            raise ValueError(
                f"merged PR #{record.pr_number} head {head_sha} does not match candidate "
                f"source {record.source_sha}: {error}"
            ) from error
    merged = merged_pr.get("merged")
    if merged is False or merged_pr.get("state") not in (None, "closed", "merged"):
        return False
    if merged is not True and not merged_pr.get("merge_commit_sha"):
        return False
    return True


def _assert_line_push_base(record: CandidateRecord, line_push_sha: str) -> None:
    """Bind the candidate base to the release-line push's first-parent chain.

    Branch protection may allow merge commits, squash merges, and rebase
    merges. A merge commit and a squash both reach the recorded base as the
    immediate first parent, while a rebase merge reaches it through the
    rewritten preparation commit. The generated pull request carries at most
    those two commits above the base, so a longer chain means the line moved
    after the candidate was prepared.

    A ``steps == 2`` chain cannot by itself tell a rebase merge apart from a
    merge commit whose recorded base is one release behind the line.  That
    stale-base case is caught one layer down: ``_assert_merge_source``
    compares the full prospective merge tree against the tested source, so a
    push that carries any other commit's content fails there.
    """

    try:
        chain = _run_git(Path("."), "rev-list", "--first-parent", line_push_sha).split()
    except ValueError as error:
        raise ValueError(
            f"line push {line_push_sha} has no inspectable history: {error}"
        ) from error
    try:
        steps = chain.index(record.base_sha)
    except ValueError:
        raise ValueError(
            f"line push {line_push_sha} does not sit on the recorded base {record.base_sha}"
        ) from None
    if steps == 0:
        raise ValueError(f"line push {line_push_sha} is the recorded base, not a candidate merge")
    if steps > 2:
        raise ValueError(
            f"line push {line_push_sha} is {steps} commits above the recorded base "
            f"{record.base_sha}; refresh the release pull request"
        )


def _assert_record_introduced(record: CandidateRecord, line_push_sha: str) -> None:
    """Require the exact record to be newly introduced by the line push."""

    expected_record = candidate.dump(record)
    try:
        source_equivalence.assert_record_present(line_push_sha, expected_record, Path("."))
        parents = _run_git(Path("."), "rev-list", "--parents", "-n", "1", line_push_sha).split()
        if len(parents) < 2:
            raise ValueError(f"line push {line_push_sha} has no parent")
        changed = _run_git(
            Path("."),
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            line_push_sha,
            parents[1],
        ).splitlines()
    except (ValueError, source_equivalence.SourceDrift) as error:
        raise ValueError(
            f"candidate record was not newly introduced by push {line_push_sha}: {error}"
        ) from error
    if source_equivalence.RECORD_PATH not in changed:
        raise ValueError(
            f"candidate record was not introduced by push {line_push_sha}; changed={changed}"
        )


def _assert_merge_source(record: CandidateRecord, line_push_sha: str) -> None:
    try:
        source_equivalence.assert_merge_equivalent(record.source_sha, line_push_sha, Path("."))
    except source_equivalence.SourceDrift as error:
        raise ValueError(f"line push {line_push_sha} changed source bytes: {error}") from error


def _published_stable_versions(release: Mapping[str, object]) -> list[str]:
    for key in ("published_stable_versions", "stable_versions"):
        if key in release:
            value = release[key]
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(
                    f"release {release.get('tag_name')} stable version inventory is invalid"
                )
            return list(value)
    for key in ("all_releases", "releases"):
        value = release.get(key)
        if value is not None:
            return _stable_versions_from_payload(value)
    try:
        value = _gh_json("api", "--paginate", f"/repos/{assets.REPOSITORY}/releases")
    except ValueError as error:
        raise ValueError(
            f"could not list published releases for latest selection: {error}"
        ) from error
    return _stable_versions_from_payload(value)


def _stable_versions_from_payload(value: object) -> list[str]:
    if isinstance(value, Mapping):
        value = value.get("releases", value.get("items"))
    if not isinstance(value, list):
        raise ValueError("published release inventory must be a list")
    entries: list[object] = []
    for page in value:
        if isinstance(page, list):
            entries.extend(page)
        else:
            entries.append(page)
    versions: list[str] = []
    for item in entries:
        if not isinstance(item, Mapping):
            continue
        if item.get("draft") is True or item.get("prerelease") is True:
            continue
        tag = item.get("tag_name", item.get("version"))
        if isinstance(tag, str):
            versions.append(tag[1:] if tag.startswith("v") else tag)
    return versions


def _semver_tuple(value: str) -> tuple[int, int, int] | None:
    normalized = value[1:] if isinstance(value, str) and value.startswith("v") else value
    match = _STABLE_VERSION.fullmatch(normalized) if isinstance(normalized, str) else None
    if match is None:
        return None
    major, minor, patch = (int(part) for part in normalized.split("."))
    return major, minor, patch


def should_make_latest(version: str, published_stable_versions: list[str]) -> bool:
    """Return whether ``version`` is the numeric maximum stable SemVer."""

    selected = _semver_tuple(version)
    if selected is None:
        raise ValueError(f"latest selection requires a plain stable SemVer, got {version!r}")
    if not isinstance(published_stable_versions, list):
        raise ValueError("published stable versions must be a list")
    maximum = selected
    for value in published_stable_versions:
        if not isinstance(value, str):
            raise ValueError("published stable versions must contain strings")
        parsed = _semver_tuple(value)
        if parsed is not None and parsed > maximum:
            maximum = parsed
    return selected == maximum


def publish_stable(
    record: CandidateRecord | Mapping[str, object],
    line_push_sha: str,
    release: Mapping[str, object],
) -> None:
    """Publish the reviewed stable draft for the merge that introduced it."""

    validated, identity = _record_identity(record)
    if validated.phase is not None:
        raise ValueError(f"stable publication cannot use preview phase {validated.phase!r}")
    expected = CandidateIdentity.from_dict(identity)
    if not isinstance(line_push_sha, str) or _GIT_SHA.fullmatch(line_push_sha) is None:
        raise ValueError(f"invalid release-line push SHA {line_push_sha!r}")
    if not isinstance(release, Mapping):
        raise ValueError("GitHub release is not an object")

    merged_pr = _merged_pr_from_release(validated, release)
    if not _matching_merge_pr(validated, line_push_sha, merged_pr):
        print(
            f"release publication line={validated.line} pr={validated.pr_number} phase=stable "
            f"source_sha={validated.source_sha} version={validated.version} "
            f"draft_release_id={validated.draft_release_id} "
            "rejection=no merge-matching candidate record"
        )
        return
    _assert_line_push_base(validated, line_push_sha)
    _assert_record_introduced(validated, line_push_sha)
    _assert_merge_source(validated, line_push_sha)
    draft = _assert_release_shape(release, expected, validated, allow_published=True)
    target = _tag_target(validated.tag)
    if target is not None and target.lower() != line_push_sha.lower():
        raise ValueError(
            f"stable tag {validated.tag} points at {target}, expected merge {line_push_sha}"
        )

    if not draft:
        _verify_published_asset_bytes(validated, release)
        print(
            f"release publication line={validated.line} pr={validated.pr_number} phase=stable "
            f"source_sha={validated.source_sha} version={validated.version} "
            f"draft_release_id={validated.draft_release_id} already published"
        )
        return

    _verify_draft_before_publication(
        validated,
        expected,
        expected_tag_target=line_push_sha,
    )
    _ensure_tag(validated.tag, line_push_sha)
    make_latest = should_make_latest(validated.version, _published_stable_versions(release))
    _publish_release(validated, prerelease=False, make_latest=make_latest)
    print(
        f"release publication line={validated.line} pr={validated.pr_number} phase=stable "
        f"source_sha={validated.source_sha} version={validated.version} "
        f"draft_release_id={validated.draft_release_id} published make_latest={make_latest}"
    )


def _run_git(
    repo: Path, *args: str, env: Mapping[str, str] | None = None, input: bytes | None = None
) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            input=input,
            env=dict(env) if env is not None else None,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr.decode(errors="replace").strip()
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        raise ValueError(f"git {' '.join(args)} failed in {repo}: {detail}") from error
    return completed.stdout.decode().strip()


def _replace_package_version(cargo: str, version: str) -> str:
    try:
        document = tomllib.loads(cargo)
    except Exception as error:  # pragma: no cover - error text is asserted by caller
        raise ValueError(f"Cargo.toml is invalid: {error}") from error
    package = document.get("package")
    if not isinstance(package, Mapping):
        raise ValueError("Cargo.toml has no package table")
    package_name = package.get("name")
    if package_name != "gel-cli":
        raise ValueError(f"Cargo.toml package is {package_name!r}, expected 'gel-cli'")
    start = re.search(r"(?m)^\[package\]\s*$", cargo)
    if start is None:
        raise ValueError("Cargo.toml has no [package] section")
    next_section = re.search(r"(?m)^\[(?!\[)[^\n]+\]\s*$", cargo[start.end() :])
    end = start.end() + next_section.start() if next_section else len(cargo)
    section = cargo[start.end() : end]
    version_match = re.search(r'(?m)^(version\s*=\s*)"[^"]+"(\s*)$', section)
    if version_match is None:
        # A workspace version is still a Cargo version field and is common in
        # small fixture workspaces.  Update it in the workspace package table.
        workspace = re.search(r"(?m)^\[workspace\.package\]\s*$", cargo)
        if workspace is None:
            raise ValueError("Cargo.toml package has no concrete version field")
        workspace_next = re.search(r"(?m)^\[(?!\[)[^\n]+\]\s*$", cargo[workspace.end() :])
        workspace_end = workspace.end() + workspace_next.start() if workspace_next else len(cargo)
        workspace_section = cargo[workspace.end() : workspace_end]
        version_match = re.search(r'(?m)^(version\s*=\s*)"[^"]+"(\s*)$', workspace_section)
        if version_match is None:
            raise ValueError("Cargo.toml workspace package has no concrete version field")
        replacement = (
            workspace_section[: version_match.start()]
            + f'{version_match.group(1)}"{version}"{version_match.group(2)}'
            + workspace_section[version_match.end() :]
        )
        return cargo[: workspace.end()] + replacement + cargo[workspace_end:]
    replacement = (
        section[: version_match.start()]
        + f'{version_match.group(1)}"{version}"{version_match.group(2)}'
        + section[version_match.end() :]
    )
    return cargo[: start.end()] + replacement + cargo[end:]


def _replace_lock_version(lock: str, version: str) -> str:
    try:
        document = tomllib.loads(lock)
    except Exception as error:  # pragma: no cover - error text is asserted by caller
        raise ValueError(f"Cargo.lock is invalid: {error}") from error
    packages = document.get("package")
    matches = [
        package
        for package in packages or []
        if isinstance(package, Mapping) and package.get("name") == "gel-cli"
    ]
    if len(matches) != 1:
        raise ValueError(f"Cargo.lock has {len(matches)} gel-cli package entries")
    blocks = list(re.finditer(r"(?ms)^\[\[package\]\]\s*\n.*?(?=^\[\[package\]\]|\Z)", lock))
    for block in blocks:
        text = block.group(0)
        if re.search(r'(?m)^name\s*=\s*"gel-cli"\s*$', text):
            updated, count = re.subn(
                r'(?m)^(version\s*=\s*)"[^"]+"(\s*)$',
                rf'\1"{version}"\2',
                text,
                count=1,
            )
            if count != 1:
                raise ValueError("Cargo.lock gel-cli entry has no version field")
            return lock[: block.start()] + updated + lock[block.end() :]
    raise ValueError("Cargo.lock gel-cli entry could not be located")


def _git_blob(repo: Path, source_sha: str, path: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "show", f"{source_sha}:{path}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr.decode(errors="replace").strip()
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        raise ValueError(f"could not read {path} from {source_sha}: {detail}") from error
    return completed.stdout


def _metadata_version(worktree: Path, expected: str) -> str:
    try:
        completed = subprocess.run(
            [
                "cargo",
                "metadata",
                "--locked",
                "--no-deps",
                "--format-version",
                "1",
                "--manifest-path",
                str(worktree / "Cargo.toml"),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=worktree,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
        )
        raise ValueError(f"cargo metadata failed for derived preview tree: {detail}") from error
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"cargo metadata returned invalid JSON: {error}") from error
    packages = payload.get("packages")
    if not isinstance(packages, list):
        raise ValueError("cargo metadata returned no package list")
    versions = {
        package.get("version")
        for package in packages
        if isinstance(package, Mapping) and package.get("name") == "gel-cli"
    }
    if versions != {expected}:
        raise ValueError(f"cargo metadata resolved gel-cli versions {versions!r}")
    return next(iter(versions))


def derive_preview_commit(source_sha: str, version: str, repo: Path = Path(".")) -> str:
    """Create a deterministic preview commit parented to ``source_sha``.

    The current checkout and index are never changed.  The returned commit is
    reproducible for the same source and version, which lets retries reuse the
    exact temporary ref and build SHA.
    """

    if _GIT_SHA.fullmatch(source_sha) is None:
        raise ValueError(f"invalid source SHA {source_sha!r}")
    preview_match = _PREVIEW_VERSION.fullmatch(version)
    if preview_match is None:
        raise ValueError(f"unsupported preview version {version!r}")
    repo = Path(repo)
    if not repo.is_dir():
        raise ValueError(f"repository does not exist: {repo}")
    _run_git(repo, "rev-parse", "--verify", f"{source_sha}^{{commit}}")
    cargo = _git_blob(repo, source_sha, "Cargo.toml").decode()
    lock = _git_blob(repo, source_sha, "Cargo.lock").decode()
    updated_cargo = _replace_package_version(cargo, version)
    updated_lock = _replace_lock_version(lock, version)

    with tempfile.TemporaryDirectory(prefix="gel-preview-index-") as temp:
        index = Path(temp) / "index"
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(index)
        _run_git(repo, "read-tree", source_sha, env=env)
        for path, contents in (
            ("Cargo.toml", updated_cargo.encode()),
            ("Cargo.lock", updated_lock.encode()),
        ):
            blob = (
                subprocess.run(
                    ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
                    input=contents,
                    check=True,
                    capture_output=True,
                )
                .stdout.decode()
                .strip()
            )
            _run_git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
        tree = _run_git(repo, "write-tree", env=env)

        # Archive this synthetic tree into a temporary directory so Cargo sees
        # exactly the tree that will be committed, including the selected lock
        # version.  ``--locked`` proves Cargo.lock is consistent with Cargo's
        # resolver under the repository's rust-toolchain file.
        with tempfile.TemporaryDirectory(prefix="gel-preview-tree-") as checkout:
            archive = subprocess.run(
                ["git", "-C", str(repo), "archive", tree],
                check=True,
                capture_output=True,
            ).stdout
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
                try:
                    tar.extractall(checkout, filter="data")
                except TypeError:  # Python 3.11 has no extraction filter argument.
                    tar.extractall(checkout)
            if _metadata_version(Path(checkout), version) != version:
                raise ValueError(f"cargo metadata did not resolve selected version {version}")

        changed = _run_git(
            repo,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            source_sha,
            tree,
        ).splitlines()
        if set(changed) != {"Cargo.toml", "Cargo.lock"}:
            raise ValueError(
                "derived preview tree changed paths outside Cargo.toml and Cargo.lock: "
                + ", ".join(changed)
            )

        commit_metadata = _run_git(
            repo,
            "show",
            "-s",
            "--format=%an%n%ae%n%aI%n%cn%n%ce%n%cI",
            source_sha,
        ).splitlines()
        if len(commit_metadata) != 6:
            raise ValueError("source commit has incomplete author metadata")
        commit_env = os.environ.copy()
        (
            commit_env["GIT_AUTHOR_NAME"],
            commit_env["GIT_AUTHOR_EMAIL"],
            commit_env["GIT_AUTHOR_DATE"],
            commit_env["GIT_COMMITTER_NAME"],
            commit_env["GIT_COMMITTER_EMAIL"],
            commit_env["GIT_COMMITTER_DATE"],
        ) = commit_metadata
        return _run_git(
            repo,
            "commit-tree",
            tree,
            "-p",
            source_sha,
            "-m",
            f"chore: derive preview {version}",
            env=commit_env,
        )
