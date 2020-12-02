#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import os
import re
import sys
import pathlib
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from distutils.version import LooseVersion
import numpy


class build_ext(_build_ext):
    def run(self):
        root_dir = pathlib.Path().absolute()
        build_dir = pathlib.Path(self.build_temp).absolute() / "cmake_build"
        install_dir = build_dir / "install"

        config = "Debug" if self.debug else "Release"

        build_dir.mkdir(parents=True, exist_ok=True)
        self.spawn(
            [
                "cmake",
                "-S",
                str(root_dir),
                "-B",
                str(build_dir),
                "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
                "-DCMAKE_BUILD_TYPE=" + config,
            ]
        )
        if not self.dry_run:
            self.spawn(["cmake", "--build", str(build_dir), "--target", "install"])

        with open(install_dir / "lib" / "pkgconfig" / "gdstk.pc", "r") as pkg:
            for line in pkg:
                if line.startswith("Cflags:"):
                    for arg in line.split()[1:]:
                        if arg.startswith("-I"):
                            self.extensions[0].include_dirs.append(arg[2:])
                        else:
                            self.extensions[0].extra_compile_args.append(arg)
                elif line.startswith("Libs:"):
                    for arg in line.split()[1:]:
                        if arg.endswith(".framework"):
                            # MacOS-specific
                            self.extensions[0].extra_link_args.extend(
                                ["-framework", arg[arg.rfind("/") + 1 : -10]]
                            )
                        elif arg.startswith("-L"):
                            self.extensions[0].library_dirs.append(arg[2:])
                        elif arg.startswith("-l"):
                            self.extensions[0].libraries.append(arg[2:])
                        else:
                            self.extensions[0].extra_link_args.append(arg)

        super().run()


with open("README.md") as fin:
    long_description = fin.read()

setup_requires = ["numpy"]
if "build_sphinx" in sys.argv:
    setup_requires.extend(["sphinx", "sphinx_rtd_theme"])

extra_compile_args = []
extra_link_args = []

if platform.system() == "Darwin" and LooseVersion(platform.release()) >= LooseVersion(
    "17.7"
):
    extra_compile_args.extend(["-std=c++11", "-mmacosx-version-min=10.9"])
    extra_link_args.extend(["-stdlib=libc++", "-mmacosx-version-min=10.9"])

version = None
version_re = re.compile('#define GDSTK_VERSION "(.*)"')
with open("src/gdstk.h") as fin:
    for line in fin:
        m = version_re.match(line)
        if m:
            version = m[1]
            break
if version is None:
    raise RuntimeError("Unable to determine version.")

setup(
    name="gdstk",
    version=version,
    author="Lucas H. Gabrielli",
    author_email="heitzmann@gmail.com",
    license="Boost Software License v1.0",
    url="https://github.com/heitzmann/gdstk",
    description="Python module for creation and manipulation of GDSII files.",
    long_description=long_description,
    keywords="GDSII CAD layout",
    provides=["gdstk"],
    ext_modules=[
        Extension(
            "gdstk",
            ["python/gdstk_module.cpp"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={"build_ext": build_ext},
    command_options={
        "build_sphinx": {
            "project": ("setup.py", "gdstk"),
            "version": ("setup.py", version),
            "release": ("setup.py", version),
            "copyright": ("setup.py", "2020, Lucas H. Gabrielli"),
            "source_dir": ("setup.py", "docs"),
            "build_dir": ("setup.py", "docs/_build"),
        }
    },
    install_requires=["numpy"],
    setup_requires=setup_requires,
    platforms="OS Independent",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Manufacturing",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Boost Software License 1.0 (BSL-1.0)",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
    zip_safe=False,
)
