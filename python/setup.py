import os
import subprocess
import sys
from setuptools import setup, Distribution


if sys.platform == "darwin":
    # Apple Silicon starts at macOS 11. Keep the compiled extension compatible
    # with every ARM Mac instead of inheriting the build runner's OS version.
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.0")

print("Building mojovec_python.mojo...")
subprocess.check_call([
    "mojo",
    "build",
    "-I",
    "..",
    "--emit",
    "shared-lib",
    "mojovec_python.mojo",
    "-o",
    "mojovec/_native.so",
])


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(self):
        return True

setup(
    name="mojovec",
    version="0.6.1",
    description="Python bindings for MojoVec",
    packages=["mojovec"],
    package_data={
        "mojovec": [
            "_native.so",
            "py.typed",
            ".dylibs/*.dylib",
            ".libs/*.so*",
        ]
    },
    distclass=BinaryDistribution,
)
