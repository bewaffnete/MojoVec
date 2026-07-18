import os
import subprocess
from setuptools import setup, Distribution

print("Building mojovec_python.mojo...")
subprocess.check_call(["mojo", "build", "-I", "..", "--emit", "shared-lib", "mojovec_python.mojo", "-o", "mojovec.so"])

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(self):
        return True

setup(
    name="mojovec",
    version="0.4.1",
    description="Python bindings for MojoVec",
    packages=[],
    data_files=[(".", ["mojovec.so"])],
    distclass=BinaryDistribution,
)
