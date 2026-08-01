import subprocess
from setuptools import setup, Distribution


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
