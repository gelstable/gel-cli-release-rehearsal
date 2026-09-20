//! The unmanaged ("curl the install script") install.
//!
//! This is the other half of the install-manager contract. The managed
//! scenarios prove that `cli upgrade` refuses to touch a package-manager-owned
//! file; this one proves the refusal is not blanket paranoia — a self-owned
//! install still detects as `direct` and still really replaces itself on disk.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use anyhow::Context;
use serde_json::json;

use crate::scenario::{self, Scenario};

/// Sorts above any real `CARGO_PKG_VERSION`, so `cli upgrade` always sees the
/// fixture package as newer and proceeds without `--force`.
const FIXTURE_VERSION: &str = "999.0.0";

/// Opt-in to replacing a `cli.toml` that already exists outside the scenario
/// directory. CI runners set it; a developer's laptop does not.
const ALLOW_GLOBAL_CONFIG: &str = "GEL_E2E_ALLOW_GLOBAL_CONFIG";

fn fixture_channel(version: &str) -> &'static str {
    if version.contains("-dev.") {
        "nightly"
    } else if version.contains('-') {
        "testing"
    } else {
        "stable"
    }
}

pub fn run() {
    let source = scenario::binary_under_test();
    let scenario = DirectScenario::new().expect("prepare the direct scenario");
    // Declared after `scenario` so it drops first: cleanup runs while the
    // scenario it cleans up is still alive.
    let _guard = scenario::CleanupGuard::new(&scenario);

    let installed = scenario
        .install(&source)
        .unwrap_or_else(|error| panic!("direct install failed: {error:#}"));

    scenario::run(
        scenario
            .command(&installed)
            .arg("--no-cli-update-check")
            .arg("--version"),
    )
    .expect_success("installed `gel --version`");

    let detected = scenario::run(
        scenario
            .command(&installed)
            .arg("--no-cli-update-check")
            .arg("info")
            .arg("--get")
            .arg("install-manager"),
    );
    detected.expect_success("`gel info --get install-manager`");
    assert_eq!(
        detected.stdout_trimmed(),
        scenario.expected_slug(),
        "an unmanaged install must detect as `direct`\n--- stderr ---\n{}",
        detected.stderr,
    );

    // Everything below drives a real self-upgrade, which needs a registry the
    // CLI will read — and that means a `cli.toml`.
    let config_dir = scenario::info_get(&installed, "config-dir", scenario.env());
    if let Some(reason) = scenario.upgrade_half_skip_reason(&config_dir) {
        // The literal `skipping:` is load-bearing, not phrasing:
        // `scripts/ci/run-install-scenario.sh` greps for exactly that string to
        // turn "nothing was tested" into a red job. A partial skip is still a
        // skip — the `direct-*` jobs exist to prove a real self-upgrade — so it
        // has to carry the marker too, with the "half" said in the human-facing
        // part rather than folded into the marker.
        eprintln!("skipping: the self-upgrade half — {reason}");
        return;
    }
    let Some(platform) = scenario::cli_platform() else {
        eprintln!("skipping: the self-upgrade half — no CLI build for this host");
        return;
    };

    scenario
        .write_fixture_registry(&installed, platform, &config_dir)
        .unwrap_or_else(|error| panic!("could not build the fixture registry: {error:#}"));

    // `upgrade()` hard-links (Unix) or renames (Windows) the old file aside
    // before writing the new one. Clear any stale sibling first so the assertion
    // below proves *this* run produced it.
    let backup = installed.with_extension("backup");
    let _ = fs_err::remove_file(&backup);

    // Deliberately no `--no-cli-update-check`: `cli upgrade` is the path under
    // test here. `main.rs` skips the background version check for this
    // subcommand anyway, so no network call happens.
    let upgraded = scenario::run(scenario.command(&installed).arg("cli").arg("upgrade"));
    upgraded.expect_success("`gel cli upgrade`");
    // `msg!` is `eprintln!` (src/print/color.rs), so the success line is on
    // stderr; stdout carries nothing here.
    //
    // The version is part of the assertion on purpose. If the fixture `cli.toml`
    // were ever not picked up, `Config::from_inputs`
    // (src/portable/registry/config.rs) falls back to `DEFAULT_PACKAGE_ROOT` —
    // the real network registry — and a bare "Upgraded to version" would still
    // match whenever a newer real release exists, having exercised none of the
    // fixture's hash, size or media-type handling. Only 999.0.0 can come from
    // the fixture.
    let expected = format!("Upgraded to version {FIXTURE_VERSION}");
    assert!(
        upgraded.stderr.contains(&expected),
        "a direct install must really upgrade itself, from the fixture registry \
         (expected {expected:?})\n--- stdout ---\n{}\n--- stderr ---\n{}",
        upgraded.stdout,
        upgraded.stderr,
    );
    assert!(
        backup.is_file(),
        "`cli upgrade` should leave a backup at {}",
        backup.display(),
    );

    scenario::run(
        scenario
            .command(&installed)
            .arg("--no-cli-update-check")
            .arg("--version"),
    )
    .expect_success("`gel --version` after the self-upgrade");
}

pub struct DirectScenario {
    /// A disposable directory inside the *real* home directory. Everything this
    /// scenario creates lives here and `TempDir::drop` removes it.
    root: tempfile::TempDir,
    /// Custody of the `cli.toml` the fixture registry is declared in, held from
    /// the moment it is written until cleanup puts the old one back. Interior
    /// mutability because `Scenario::cleanup` takes `&self`.
    config_file: Mutex<Option<scenario::ConfigFile>>,
}

impl DirectScenario {
    /// Two production checks box this scenario into one layout, and it is worth
    /// spelling out why the obvious "just point `$HOME` at a tempdir" does not
    /// work:
    ///
    /// * `_get_upgrade_path()` (`src/cli/upgrade.rs`) refuses to upgrade a
    ///   binary that does not live under `home_dir()`, so the installed
    ///   executable has to be somewhere inside the home directory the CLI
    ///   resolves.
    /// * `cli install` (`src/cli/install.rs`) aborts on Unix when `$HOME`
    ///   differs from the euid's passwd home ("you may be using sudo") unless it
    ///   is given `-y`. `cli upgrade` re-executes the downloaded binary as
    ///   `cli install --upgrade ...` **without** `-y`, so any scenario that
    ///   overrides `$HOME` can install fine but can never complete an upgrade.
    ///   Nothing in the environment can suppress that check.
    ///
    /// So `$HOME` is left alone and the scenario root is a tempdir *inside* it —
    /// the strategy Windows needs anyway, because there `dirs` resolves the
    /// data/config directories through the Known Folder API, which ignores
    /// `APPDATA`/`LOCALAPPDATA`. The install lands in the scenario root via the
    /// hidden `--installation-path` flag, so the developer's real `gel` is never
    /// touched.
    fn new() -> anyhow::Result<DirectScenario> {
        let home = dirs::home_dir().context("cannot determine the home directory")?;
        let root = tempfile::Builder::new()
            .prefix(".gel-e2e-direct-")
            .tempdir_in(&home)
            .with_context(|| format!("creating a scenario directory in {}", home.display()))?;
        Ok(DirectScenario {
            root,
            config_file: Mutex::new(None),
        })
    }

    fn root(&self) -> &Path {
        self.root.path()
    }

    fn install_dir(&self) -> PathBuf {
        self.root().join("bin")
    }

    /// Whether writing the fixture `cli.toml` would cost the developer anything.
    ///
    /// Three cases, in order:
    ///
    /// 1. The config dir is inside the scenario root (Linux, via
    ///    `XDG_CONFIG_HOME`) — nothing outside the tempdir is involved.
    /// 2. It is the real config dir but holds no `cli.toml` — we create one and
    ///    delete it again in cleanup, displacing nothing.
    /// 3. It is the real config dir and a `cli.toml` is already there. Now the
    ///    test would have to move the developer's own registry configuration out
    ///    of the way, so it refuses unless explicitly allowed.
    fn upgrade_half_skip_reason(&self, config_dir: &Path) -> Option<String> {
        if config_dir.starts_with(self.root()) {
            return None;
        }
        if !scenario::ConfigFile::is_occupied(config_dir) {
            return None;
        }
        if std::env::var_os(ALLOW_GLOBAL_CONFIG).is_some_and(|value| value == "1") {
            return None;
        }
        Some(format!(
            "{} already holds a {} and this platform cannot redirect the CLI \
             config directory; set {ALLOW_GLOBAL_CONFIG}=1 to let the test move \
             it aside and put it back afterwards",
            config_dir.display(),
            scenario::CONFIG_FILE,
        ))
    }

    /// Build a one-package registry the upgrade can actually resolve.
    ///
    /// `file://` sources are first-class (`download_file` in
    /// `src/portable/registry/download.rs`), so no HTTP server is needed. They
    /// have to be declared through `[registry].sources`, though:
    /// `parse_package_root` (`src/portable/registry/config.rs`) rejects any
    /// scheme that is not http/https, so `GEL_PKG_ROOT` cannot point at a local
    /// fixture.
    fn write_fixture_registry(
        &self,
        installed: &Path,
        platform: &str,
        config_dir: &Path,
    ) -> anyhow::Result<()> {
        let root = self.root();
        let artifacts = root.join("artifacts");
        fs_err::create_dir_all(&artifacts)?;

        // The "new version" is a copy of the binary we just installed: the point
        // is that the upgrade machinery runs end to end, not that the contents
        // differ.
        let artifact = artifacts.join(format!("gel-cli-{FIXTURE_VERSION}"));
        fs_err::copy(installed, &artifact)?;
        let size = fs_err::metadata(&artifact)?.len();
        let blake2b = scenario::blake2b_hex(&artifact);
        let artifact_url = url::Url::from_file_path(&artifact)
            .map_err(|()| anyhow::anyhow!("cannot build a file URL for {}", artifact.display()))?;

        let index = json!({
            "packages": [{
                "basename": "gel-cli",
                "version": FIXTURE_VERSION,
                "slot": "999",
                "tags": {},
                "installrefs": [{
                    "ref": artifact_url.as_str(),
                    // Platform-dependent: `validate_cli`
                    // (src/portable/registry/index.rs) drops installrefs whose
                    // media type does not match the index platform.
                    "type": scenario::cli_media_type(platform),
                    "encoding": "identity",
                    "verification": { "size": size, "blake2b": blake2b },
                }],
            }],
        });
        fs_err::write(root.join("index.json"), serde_json::to_vec(&index)?)?;

        // Match `cli::upgrade::channel_of`: release-candidate harnesses compile
        // this test from a prerelease-derived commit, so hardcoding `stable`
        // makes their direct-upgrade scenario miss the fixture entirely.
        let channel = fixture_channel(env!("CARGO_PKG_VERSION"));
        let registry = json!({
            "schema_version": 1,
            "indexes": [{
                "channel": channel,
                "platform": platform,
                "ref": "index.json",
            }],
        });
        let registry_path = root.join("registry.json");
        fs_err::write(&registry_path, serde_json::to_vec(&registry)?)?;

        self.write_cli_toml(config_dir, &registry_path)
    }

    fn write_cli_toml(&self, config_dir: &Path, registry_path: &Path) -> anyhow::Result<()> {
        // `{:?}` on the path string yields a quoted, escaped TOML basic string,
        // which is what makes Windows backslashes survive.
        let body = format!(
            "[registry]\nsources = [\n  {:?},\n]\n",
            registry_path.display().to_string(),
        );
        // `ConfigFile` moves any existing config to a sibling file and holds the
        // harness-wide config lock until `restore`.
        *self
            .config_file
            .lock()
            .unwrap_or_else(PoisonError::into_inner) =
            Some(scenario::ConfigFile::replace(config_dir, &body)?);
        Ok(())
    }
}

impl Scenario for DirectScenario {
    fn expected_slug(&self) -> &'static str {
        "direct"
    }

    /// `HOME` is conspicuously absent — see `DirectScenario::new`. The XDG
    /// variables move the config, data and cache directories into the scenario
    /// root on Linux, where `dirs` honours them; on macOS and Windows `dirs`
    /// ignores them and the CLI uses the real directories, which is what
    /// `upgrade_half_skip_reason` exists to notice.
    fn env(&self) -> Vec<(&'static str, PathBuf)> {
        let root = self.root();
        vec![
            ("XDG_CONFIG_HOME", root.join("config")),
            ("XDG_DATA_HOME", root.join("data")),
            ("XDG_CACHE_HOME", root.join("cache")),
            ("XDG_BIN_HOME", self.install_dir()),
        ]
    }

    fn install(&self, source: &Path) -> anyhow::Result<PathBuf> {
        let install_dir = self.install_dir();
        let staged = scenario::stage_binary(source, &self.root().join("stage"))?;

        let out = scenario::run(
            self.command(&staged)
                .arg("--no-cli-update-check")
                .arg("cli")
                .arg("install")
                .arg("-y")
                .arg("--no-modify-path")
                // Explicit on every platform: `$HOME` is the developer's real
                // one, so the default install path is their real `gel`.
                .arg("--installation-path")
                .arg(&install_dir),
        );
        anyhow::ensure!(
            out.success,
            "`gel cli install` failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
            out.stdout,
            out.stderr,
        );

        let installed = install_dir.join(scenario::EXE_NAME);
        anyhow::ensure!(
            installed.is_file(),
            "`gel cli install` did not produce {}",
            installed.display(),
        );
        Ok(installed)
    }

    fn cleanup(&self) {
        // `root` — the install tree and the fixture registry — is removed by
        // `TempDir::drop` once this scenario goes out of scope. The only thing
        // that can live outside it is a `cli.toml` in the real config dir.
        //
        // Recovering from a poisoned lock rather than bailing: cleanup is the
        // only thing that puts the developer's config back, so it has to run
        // even when the scenario panicked.
        let taken = self
            .config_file
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(config) = taken
            && let Err(error) = config.restore()
        {
            eprintln!("cleanup: could not restore the CLI config: {error:#}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixture_channel;

    #[test]
    fn fixture_channel_matches_the_binary_upgrade_channel() {
        assert_eq!(fixture_channel("7.11.0"), "stable");
        assert_eq!(fixture_channel("7.11.0-rc.1"), "testing");
        assert_eq!(fixture_channel("7.11.0-dev.42"), "nightly");
    }
}
