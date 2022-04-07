#!/usr/bin/env python3
import sys
import json
import pathlib
import os
from skbuild import setup

print("system.platform is {}".format(sys.platform))
if (sys.platform == "darwin"):
    from distutils import sysconfig
    vars = sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')

fldr_path = pathlib.Path(__file__).parent.absolute()

with open(os.path.join(fldr_path, 'cmake_config_file.json')) as fp:
    d = json.load(fp)

class EmptyListWithLength(list):
    def __len__(self):
        return 1

setup(cmake_args=d['cmake_args'])