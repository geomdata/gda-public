#!/bin/env python3

import sys
import toml
import pathlib
pyproject_path = pathlib.Path("pyproject.toml")


def error(key=None):
    raise ValueError(f"Unknown keyword {key}. Try 'project' or the entries in 'project.optional-dependencies' in pyproject.toml")

keyword = None
if len(sys.argv) > 1:
    keyword = sys.argv[1]

toml_dict = toml.load(pyproject_path.open())

if keyword is None:
    print(" ".join(toml_dict["build-system"]["requires"]))
elif keyword == "project":
    print(" ".join(toml_dict["project"]["dependencies"]))
elif keyword in toml_dict['project']['optional-dependencies'].keys():
    print(" ".join(toml_dict['project']['optional-dependencies'][keyword]))
else:
    error(keyword)
