#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020-2020 Lucas Heitzmann Gabrielli.
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
        build_dir = pathlib.Path(self.build_temp)
        build_dir.mkdir(parents=True, exist_ok=True)

        config = "Debug" if self.debug else "Release"
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(build_dir.absolute()),
            "-DCMAKE_BUILD_TYPE=" + config,
        ]
        build_args = ["--config", config]

        os.chdir(build_dir)
        self.spawn(["cmake", str(root_dir)] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", "."] + build_args)
        os.chdir(root_dir)

        if (build_dir / "is_multiconfig").exists():
            # Visual Studio and Xcode
            self.extensions[0].library_dirs.append(str((build_dir / config).absolute()))
        else:
            self.extensions[0].library_dirs.append(str(build_dir.absolute()))

        for arg in (build_dir / "lapack_libs").read_text().split():
            if arg.endswith(".framework"):
                # MacOS-specific
                self.extensions[0].extra_link_args.extend(
                    ["-framework", arg[arg.rfind("/") + 1 : -10]]
                )
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
            include_dirs=["src", numpy.get_include()],
            libraries=["gdstk"],
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
        "Development Status :: 1 - Planning",
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
