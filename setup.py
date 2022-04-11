#!/usr/bin/env python3
import sys
import json
import pathlib
import os
from setuptools import find_packages
from skbuild import setup

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

print("system.platform is {}".format(sys.platform))
if (sys.platform == "darwin"):
    from distutils import sysconfig
    vars = sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')

fldr_path = pathlib.Path(__file__).parent.absolute()
with open(os.path.join(fldr_path, 'cmake_config_file.json')) as fp:
    d = json.load(fp)

setup(
    cmake_args=d['cmake_args'],
    cmdclass={'bdist_wheel': None}
    )