# Ordinary backport rehearsal

This docs-only change exists to verify that an ordinary pull request targeting
`release/v7.x` takes the release gate's safe path and that the release
controller skips it without dispatching a candidate build.

The follow-up commit intentionally exercises the pull request synchronization
event consumed by the release controller.
