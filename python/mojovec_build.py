"""PEP 517 setuptools hook for compiling MojoVec's native Python module."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from setuptools import Extension
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
MOJO_SOURCE = PROJECT_ROOT / "python_native" / "mojovec_python.mojo"

if sys.platform == "darwin":
    # Apple Silicon starts at macOS 11. Do not inherit the runner's newer OS
    # version in either the compiled library or the wheel platform tag.
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.0")


def _is_linux_x86_64() -> bool:
    return sys.platform.startswith("linux") and platform.machine().lower() in {
        "amd64",
        "x86_64",
    }


class MojoBuildExt(build_ext):
    """Compile the Mojo extension only during the native build phase."""

    def get_ext_filename(self, ext_name: str) -> str:
        # Mojo emits a CPython extension with the stable import filename used
        # by the runtime dispatcher rather than setuptools' ABI suffix.
        return os.path.join(*ext_name.split(".")) + ".so"

    def _package_directory(self) -> Path:
        if self.inplace:
            return PYTHON_ROOT / "mojovec"
        return Path(self.build_lib) / "mojovec"

    def _native_outputs(self) -> list[Path]:
        package_directory = self._package_directory()
        if _is_linux_x86_64():
            return [
                package_directory / "_avx2" / "_native.so",
                package_directory / "_avx512" / "_native.so",
            ]
        return [package_directory / "_native.so"]

    def _compile(self, output: Path, target_cpu: str | None = None) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "mojo",
            "build",
            "-I",
            str(PROJECT_ROOT),
            "--emit",
            "shared-lib",
            str(MOJO_SOURCE),
            "-o",
            str(output),
        ]
        if sys.platform == "darwin":
            # Resolve CPython C-API symbols from the importing interpreter.
            # This is the standard extension-module model on macOS and avoids
            # pinning the wheel to one particular libpython installation.
            command.extend(
                ["-Xlinker", "-undefined", "-Xlinker", "dynamic_lookup"]
            )
        if target_cpu is not None:
            command.extend(["--target-cpu", target_cpu])
        self.announce(
            f"building {output} ({target_cpu or 'host target'})", level=2
        )
        subprocess.check_call(command, cwd=PROJECT_ROOT)

    def build_extension(self, ext: Extension) -> None:
        if _is_linux_x86_64():
            # x86-64-v3 is the portable AVX2 baseline. x86-64-v4 adds the
            # complete AVX-512 subset selected by mojovec._dispatch.
            avx2_output, avx512_output = self._native_outputs()
            self._compile(avx2_output, "x86-64-v3")
            self._compile(avx512_output, "x86-64-v4")
            return
        self._compile(self._native_outputs()[0])

    def get_outputs(self) -> list[str]:
        return [str(path) for path in self._native_outputs()]


class MojoBdistWheel(bdist_wheel):
    """Keep Apple Silicon wheels compatible with the macOS 11 baseline."""

    def finalize_options(self) -> None:
        if sys.platform == "darwin" and platform.machine().lower() == "arm64":
            self.plat_name = "macosx_11_0_arm64"
        super().finalize_options()
