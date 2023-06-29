#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import numpy
import pathlib
import platform
import re
import setuptools
from setuptools.command.build_ext import build_ext


class CMakeBuilder(build_ext):
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
            self.spawn(
                [
                    "cmake",
                    "--build",
                    str(build_dir),
                    "--config",
                    config,
                    "--target",
                    "install",
                ]
            )

        pkgconfig = list(install_dir.glob("**/gdstk.pc"))
        if len(pkgconfig) == 0:
            raise RuntimeError(f"File gdstk.pc not found in cmake install tree: {install_dir}")
        with open(pkgconfig[0]) as pkg:
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


def loose_version(version_string):
    """Extracted from distutils.version.LooseVersion"""
    version = []
    for component in re.split(r'(\d+ | [a-z]+ | \.)', version_string, flags=re.VERBOSE):
        if len(component) == 0 or component == ".":
            continue
        try:
            version.append(int(component))
        except ValueError:
            version.append(component)
    return tuple(version)


extra_compile_args = []
extra_link_args = []
if platform.system() == "Darwin" and loose_version(platform.release()) >= (17, 7):
    extra_compile_args.extend(["-std=c++11", "-mmacosx-version-min=10.9"])
    extra_link_args.extend(["-stdlib=libc++", "-mmacosx-version-min=10.9"])

setuptools.setup(
    ext_modules=[
        setuptools.Extension(
            "gdstk.gdstk",
            ["python/gdstk_module.cpp"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    packages=["gdstk"],
    package_data={"gdstk": ["*.pyi", "py.typed"]},
    include_package_data=True,
    cmdclass={"build_ext": CMakeBuilder},
    zip_safe=False,
)
