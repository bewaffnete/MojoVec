import os
import platform
import subprocess
import sys
from setuptools import setup, Distribution


if sys.platform == "darwin":
    # Apple Silicon starts at macOS 11. Keep the compiled extension compatible
    # with every ARM Mac instead of inheriting the build runner's OS version.
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.0")

def build_native(output: str, target_cpu: str | None = None) -> None:
    command = [
        "mojo",
        "build",
        "-I",
        "..",
        "--emit",
        "shared-lib",
        "mojovec_python.mojo",
        "-o",
        output,
    ]
    if target_cpu is not None:
        command.extend(["--target-cpu", target_cpu])
    print(f"Building {output} ({target_cpu or 'host target'})...")
    subprocess.check_call(command)


is_linux_x86_64 = sys.platform.startswith("linux") and platform.machine().lower() in {
    "amd64",
    "x86_64",
}

if is_linux_x86_64:
    # Do not inherit the GitHub runner's CPU features. x86-64-v3 is the
    # portable AVX2 baseline; x86-64-v4 adds the complete AVX-512 subset.
    build_native("mojovec/_avx2/_native.so", "x86-64-v3")
    build_native("mojovec/_avx512/_native.so", "x86-64-v4")
else:
    build_native("mojovec/_native.so")


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(self):
        return True

native_package_data = {
    "mojovec": [
        "py.typed",
        ".dylibs/*.dylib",
        ".libs/*.so*",
    ],
}
packages = ["mojovec"]
if is_linux_x86_64:
    packages.extend(["mojovec._avx2", "mojovec._avx512"])
    native_package_data["mojovec._avx2"] = ["_native.so"]
    native_package_data["mojovec._avx512"] = ["_native.so"]
else:
    native_package_data["mojovec"].append("_native.so")


setup(
    name="mojovec",
    version="0.7.0",
    description="Python bindings for MojoVec",
    packages=packages,
    package_data=native_package_data,
    distclass=BinaryDistribution,
)
