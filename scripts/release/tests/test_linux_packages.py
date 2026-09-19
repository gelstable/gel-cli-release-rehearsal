import subprocess
import tempfile
import tomllib
import unittest
import unittest.mock
from pathlib import Path

from gel_release import assets, linux_packages

CARGO = tomllib.loads(Path("Cargo.toml").read_text())
AMD64 = assets.BY_TRIPLE["x86_64-unknown-linux-musl"]
ARM64 = assets.BY_TRIPLE["aarch64-unknown-linux-musl"]


class CargoMetadataTests(unittest.TestCase):
    def test_deb_assets_install_binary_and_completions(self):
        deb = CARGO["package"]["metadata"]["deb"]
        destinations = {row[1] for row in deb["assets"]}
        binary = next(row for row in deb["assets"] if row[1] == "usr/bin/")
        self.assertEqual(binary[0], "target/release/gel")
        self.assertIn("usr/bin/", destinations)
        self.assertIn("usr/share/bash-completion/completions/gel", destinations)
        self.assertIn("usr/share/zsh/site-functions/_gel", destinations)
        self.assertIn("usr/share/fish/vendor_completions.d/gel.fish", destinations)
        self.assertEqual(deb["name"], "gel")

    def test_rpm_assets_install_binary_and_completions(self):
        rpm = CARGO["package"]["metadata"]["generate-rpm"]
        destinations = {row["dest"] for row in rpm["assets"]}
        binary = next(row for row in rpm["assets"] if row["dest"] == "/usr/bin/gel")
        self.assertEqual(binary["source"], "target/RELEASE_TARGET/release/gel")
        self.assertIn("/usr/bin/gel", destinations)
        self.assertIn("/usr/share/bash-completion/completions/gel", destinations)
        self.assertIn("/usr/share/zsh/site-functions/_gel", destinations)
        self.assertIn("/usr/share/fish/vendor_completions.d/gel.fish", destinations)
        self.assertEqual(rpm["release"], "1")

    def test_rpm_and_deb_do_not_depend_on_a_dynamic_libc(self):
        self.assertEqual(CARGO["package"]["metadata"]["deb"]["depends"], "")
        self.assertEqual(CARGO["package"]["metadata"]["generate-rpm"]["auto-req"], "no")


class CommandTests(unittest.TestCase):
    def test_package_manager_versions_sort_prereleases_before_stable(self):
        self.assertEqual(linux_packages.package_manager_version("7.11.0"), "7.11.0")
        self.assertEqual(linux_packages.package_manager_version("7.11.0-rc.1"), "7.11.0~rc.1")

    def test_deb_and_rpm_commands_order_every_supported_prerelease_below_stable(self):
        stable_deb = linux_packages.deb_command(AMD64, "7.1.0", Path("target/completions"))
        stable_rpm = linux_packages.rpm_command(AMD64, "7.1.0")
        stable_deb_version = stable_deb[stable_deb.index("--deb-version") + 1]
        stable_rpm_version = stable_rpm[stable_rpm.index("--set-metadata") + 1]
        self.assertEqual(stable_deb_version, "7.1.0-1")
        self.assertEqual(stable_rpm_version, 'version = "7.1.0"')

        for phase in ("alpha", "beta", "rc"):
            version = f"7.1.0-{phase}.1"
            with self.subTest(version=version):
                deb = linux_packages.deb_command(AMD64, version, Path("target/completions"))
                rpm = linux_packages.rpm_command(AMD64, version)
                deb_version = deb[deb.index("--deb-version") + 1]
                rpm_version = rpm[rpm.index("--set-metadata") + 1]
                self.assertEqual(deb_version, f"7.1.0~{phase}.1-1")
                self.assertEqual(rpm_version, f'version = "7.1.0~{phase}.1"')
                # Debian and RPM both define `~` as sorting before the same
                # version without a prerelease suffix.  Compare the exact
                # package values after removing each tool's fixed revision
                # decoration so this test does not rely on Python's lexical
                # ordering (where `~` sorts after `-`).
                deb_core, _, deb_revision = deb_version.rpartition("-")
                stable_core, _, stable_revision = stable_deb_version.rpartition("-")
                self.assertEqual(deb_revision, stable_revision)
                self.assertEqual(deb_core.split("~", 1)[0], stable_core)
                self.assertTrue(deb_core.partition("~")[1])

                rpm_core = rpm_version.removeprefix('version = "').removesuffix('"')
                stable_rpm_core = stable_rpm_version.removeprefix('version = "').removesuffix('"')
                self.assertEqual(rpm_core.split("~", 1)[0], stable_rpm_core)
                self.assertTrue(rpm_core.partition("~")[1])

    def test_package_manager_version_rejects_unsupported_suffixes(self):
        for version in (
            "7.1.0-dev.1",
            "7.1.0-preview.1",
            "7.1.0-foo.1",
            "7.1.0+build.123",
            "7.1.0-alpha.0",
            "7.1.0-beta.0",
            "7.1.0-rc.0",
        ):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "unsupported release version"):
                    linux_packages.package_manager_version(version)

    def test_prerelease_commands_override_internal_package_versions(self):
        deb = linux_packages.deb_command(AMD64, "7.11.0-rc.1", Path("target/completions"))
        rpm = linux_packages.rpm_command(AMD64, "7.11.0-rc.1")
        self.assertEqual(deb[deb.index("--deb-version") + 1], "7.11.0~rc.1-1")
        self.assertEqual(rpm[rpm.index("--set-metadata") + 1], 'version = "7.11.0~rc.1"')

    def test_deb_command_targets_the_prebuilt_binary(self):
        argv = linux_packages.deb_command(AMD64, "7.11.0", Path("target/completions"))
        self.assertEqual(argv[:2], ["cargo", "deb"])
        self.assertIn("--no-build", argv)
        self.assertIn("--target", argv)
        self.assertIn(AMD64.triple, argv)

    def test_rpm_command_uses_matching_arch(self):
        argv = linux_packages.rpm_command(ARM64, "7.11.0")
        self.assertEqual(argv[:2], ["cargo", "generate-rpm"])
        self.assertIn("--target", argv)
        self.assertIn(ARM64.triple, argv)
        self.assertIn("--arch", argv)
        self.assertIn("aarch64", argv)

    def test_non_linux_targets_are_rejected(self):
        with self.assertRaises(ValueError):
            linux_packages.deb_command(
                assets.BY_TRIPLE["aarch64-apple-darwin"], "7.11.0", Path(".")
            )


class NamingTests(unittest.TestCase):
    def test_canonical_output_names(self):
        self.assertEqual(assets.deb_name("7.11.0", AMD64), "gel_7.11.0_amd64.deb")
        self.assertEqual(assets.deb_name("7.11.0", ARM64), "gel_7.11.0_arm64.deb")
        self.assertEqual(assets.rpm_name("7.11.0", AMD64), "gel-7.11.0-1.x86_64.rpm")
        self.assertEqual(assets.rpm_name("7.11.0", ARM64), "gel-7.11.0-1.aarch64.rpm")


class BuildTests(unittest.TestCase):
    def test_placeholder_missing_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cargo.toml").write_text("[package]\nname = 'gel'\n")
            with self.assertRaises(ValueError) as ctx:
                linux_packages._manifest_with_target(root, AMD64)
            self.assertIn(linux_packages.TARGET_PLACEHOLDER, str(ctx.exception))

    def _build(self, cargo):
        """Run ``build`` with ``cargo`` standing in for ``subprocess.run``.

        Reports what the real ``Cargo.toml`` looked like while cargo ran, the
        backup file the build made, and whatever ``build`` returned or raised.
        """

        seen: dict[str, object] = {}
        original_named_temporary_file = tempfile.NamedTemporaryFile

        def spy_named_temporary_file(*args, **kwargs):
            handle = original_named_temporary_file(*args, **kwargs)
            seen["backup"] = Path(handle.name)
            return handle

        def run(cmd, *args, **kwargs):
            seen["manifest"] = Path("Cargo.toml").read_text()
            return cargo(cmd)

        with (
            unittest.mock.patch(
                "tempfile.NamedTemporaryFile", side_effect=spy_named_temporary_file
            ),
            unittest.mock.patch("subprocess.run", side_effect=run),
            tempfile.TemporaryDirectory() as tmp_out,
        ):
            out_dir = Path(tmp_out)
            seen["out_dir"] = out_dir
            try:
                produced = linux_packages.build(
                    AMD64, "7.11.0", Path("target/completions"), out_dir
                )
            except subprocess.CalledProcessError as error:
                seen["error"] = error
            else:
                seen["produced"] = produced
                seen["existing"] = [path for path in produced if path.is_file()]
        return seen

    def _assert_manifest_is_restored(self, seen, original_manifest):
        # The build must patch the target placeholder in place and put the
        # pristine manifest back, leaving no backup file behind.
        self.assertIn(AMD64.triple, seen["manifest"])
        self.assertNotIn(linux_packages.TARGET_PLACEHOLDER, seen["manifest"])
        self.assertEqual(Path("Cargo.toml").read_text(), original_manifest)
        self.assertFalse(seen["backup"].exists())

    def test_build_replaces_placeholder_and_rolls_back_on_failure(self):
        def failing_cargo(cmd):
            raise subprocess.CalledProcessError(1, cmd)

        original_manifest = Path("Cargo.toml").read_text()
        seen = self._build(failing_cargo)

        self.assertIsInstance(seen["error"], subprocess.CalledProcessError)
        self._assert_manifest_is_restored(seen, original_manifest)

    def test_build_success_restores_manifest_and_collects_packages(self):
        def succeeding_cargo(cmd):
            dist = Path("dist")
            dist.mkdir(parents=True, exist_ok=True)
            (dist / assets.deb_name("7.11.0", AMD64)).write_text("dummy deb")
            (dist / assets.rpm_name("7.11.0", AMD64)).write_text("dummy rpm")

        original_manifest = Path("Cargo.toml").read_text()
        seen = self._build(succeeding_cargo)

        out_dir = seen["out_dir"]
        self.assertEqual(
            seen["produced"],
            [
                out_dir / assets.deb_name("7.11.0", AMD64),
                out_dir / assets.rpm_name("7.11.0", AMD64),
            ],
        )
        self.assertEqual(seen["existing"], seen["produced"])
        self._assert_manifest_is_restored(seen, original_manifest)


if __name__ == "__main__":
    unittest.main()
