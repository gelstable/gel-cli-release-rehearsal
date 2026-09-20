## 7.11.0 (2026-09-20)

### Features

- add command to hash password (#1295)
- port native release mechanics
- validate release line pull request identity
- select preview versions and source snapshots
- extend candidate records for release lines
- prepare release pull requests per line
- resolve release candidates and preview commits
- Exercise the complete v7 release rehearsal before production cutover.

### Fixes

- check if default `dbschema` dir exist before performing any migrations related stuff (#1296)
- Use actual server version for futures.esdl check (#1298)
- change 'branch' to 'target_branch' (#1300)
- Set branch when updating credentials (#1306)
- clear all proxies in local environment (#1316)
- use readonly lock for `gel generate` (#1745)
- flush registry downloads before returning (#10)
- use ref for registry index references (#11)
- defer cli upgrade to the owning package manager (#12)
- bind draft verification to release identity
- resolve draft tags before source checks
- distinguish missing tags from lookup errors
- require unambiguous HTTP 404 for missing tags
- reject conflicting release phase labels
- distinguish published preview snapshots
- harden candidate verification boundaries
- close release preparation races
- exercise release PR creation path
- preserve immutable preview refs
- bind release staging to candidate identity
- accept staged release candidate heads
- tighten release publication identity gates
- align release version grammar
- close release-line final review findings
- preserve occupied preview tags and failed release refs
- send make_latest as the documented GitHub string enum
- publish stable candidates merged by any GitHub merge method
- give stable candidate staging a fixed point
- correct the v7 line migration and reject already-published versions early
- accept the published replacements manifest shape in ReleaseManifest
- drop caller-supplied publication verdicts and prove them from fixtures
- validate generated manifests strictly beside the legacy shape
- attach the stable merge gate to the commit branch protection reads
- keep the release controller on generated release PRs only
- split the manual retry dispatches and drop the dead record refusal
- read the operating repository from the runner environment
- make the gate's snapshot check live and delete the dead shape sniffing
- stabilize hosted candidate packaging
- consume archive listings under pipefail
- allow draft candidate verification
