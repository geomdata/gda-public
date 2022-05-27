#!/bin/env python3
r""" A simple script for helping us pyproject.toml information in CI/CD, tests, etc. """


import sys
import logging
import setuptools
import argparse

parser = argparse.ArgumentParser()                                              
parser.add_argument('-t', '--toml', type=str, default="pyproject.toml",
    help='path to pyproject.toml TOML file with package configuration')

out_group = parser.add_mutually_exclusive_group(required=True)
out_group.add_argument('-d', '--deps', nargs='+',
    help='List the dependencies of the given types. Available types shown by --listdeps')

out_group.add_argument('-l', '--listdeps', action="store_true",
    help='List the options available for --deps')
    
out_group.add_argument('-n', '--name', action="store_true", 
    help='Show the package name')

out_group.add_argument('-v', '--version', action="store_true", 
    help='Show the package version')

out_group.add_argument('-N', '--nameversion', action="store_true", 
    help='Show the package name==version for pip specifications')


args = parser.parse_args()

# This throws annoying warnings. Suppress them
logging.captureWarnings(True)
log = logging.getLogger()
log.setLevel(logging.ERROR)
toml_dict = setuptools.config.pyprojecttoml.read_configuration(args.toml)

list_deps = []
if "build-system" in toml_dict:
    if "requires" in toml_dict["build-system"]:
        list_deps.append("build")
if "project" in toml_dict:
    if "dependencies" in toml_dict["project"]:
        list_deps.append("project")
list_deps.extend(toml_dict['project']['optional-dependencies'].keys())

if args.listdeps:
    print(f" ".join(list_deps))
    sys.exit(0)

if args.deps:
    out_deps = []
    for t in args.deps:
        if t == "build":
            out_deps.extend(toml_dict['build-system']['requires'])
        elif t == "project":
            out_deps.extend(toml_dict['project']['dependencies'])
        else:
            try:
                out_deps.extend(toml_dict['project']['optional-dependencies'][t])
            except KeyError:
                print(f"Invalid option {t}. Use --listdeps to see the valid dependency types.")
                sys.exit(1)

    print(f" ".join(out_deps))
    sys.exit(0)

if args.name:
    print(toml_dict['project']['name'])
    sys.exit(0)

if args.version:
    print(toml_dict['project']['version'])
    sys.exit(0)

if args.nameversion:
    print(toml_dict['project']['name']+'=='+toml_dict['project']['version'])
    sys.exit(0)