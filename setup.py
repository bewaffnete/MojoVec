import os
import subprocess
from setuptools import setup, Distribution
from setuptools.command.build_py import build_py

class BuildMojoPy(build_py):
    def run(self):
        # Build the Mojo shared library
        print("Building mojovec_python.mojo...")
        subprocess.check_call(["mojo", "build", "--emit", "shared-lib", "mojovec_python.mojo", "-o", "mojovec.so"])
        super().run()

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(self):
        return True

setup(
    name="mojovec",
    version="0.1.0",
    description="Python bindings for MojoVec",
    packages=[],
    # We include the compiled .so file
    data_files=[(".", ["mojovec.so"])],
    cmdclass={
        'build_py': BuildMojoPy,
    },
    distclass=BinaryDistribution,
)
