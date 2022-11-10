#!/usr/bin/env python

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import sys
import resource
import gc
import timeit
import pathlib
import importlib

import numpy
import psutil

import gdspy
import gdstk


def prefix_format(value, stdev=None, unit="", fmtstr=None, base=10):
    if fmtstr is None:
        fmtstr = (
            "{value:.3g} {prefix}{unit}"
            if stdev == None
            else "({value:.3g} ± {stdev:.3g}) {prefix}{unit}"
        )
    if stdev is None:
        stdev = 0
    if base == "bin":
        base = 2 ** (10 / 3)
    if value == 0:
        m = 0
        prefix = ""
    else:
        m = int(numpy.floor(numpy.log10(abs(value)) / numpy.log10(base)))
        m = int(numpy.floor(m / 3.0) * 3)
        prefix = {
            -18: "a",
            -15: "f",
            -12: "p",
            -9: "n",
            -6: "μ",
            -3: "m",
            0: "",
            3: "k",
            6: "M",
            9: "G",
            12: "T",
        }.get(m, f"× {base}^{m} ")
    value *= base**-m
    stdev *= base**-m
    return fmtstr.format(value=value, stdev=stdev, prefix=prefix, unit=unit)


def timing_benchmark():
    namespace = dict(globals())

    def print_row(*vals, hsep=False):
        columns = [16, 16, 16, 8]
        print(
            "|",
            vals[0].ljust(columns[0]),
            "|",
            " | ".join(v.center(c) for v, c in zip(vals[1:], columns[1:])),
            "|",
        )
        if hsep:
            print(
                "|",
                ":" + "-" * (columns[0] - 1),
                "|",
                " | ".join(":" + "-" * (c - 2) + ":" for c in columns[1:]),
                "|",
            )

    path = pathlib.Path(__file__).absolute()
    modules = ("gdspy", "gdstk")
    bench_names = sorted(f.stem for f in path.parent.glob("*.py") if f != path)

    timers = {}
    best_avg_time = {}
    for name in bench_names:
        bench_module = importlib.import_module(name)
        for mod in modules:
            func = f"{mod}_{name.replace('-', '')}"
            namespace[func] = getattr(bench_module, f"bench_{mod}")
            key = (name, mod)
            timers[key] = timeit.Timer(f"{func}()", "gc.enable()", globals=namespace)
            best_avg_time[key] = 1e300

    sets = 16
    repeats = 8
    for s in range(1, 1 + sets):
        print(f"\n{s}/{sets}:", end="")
        for name in numpy.random.permutation(bench_names):
            print(f" {name}", end="", flush=True)
            for mod in numpy.random.permutation(modules):
                key = (name, mod)
                t = timers[key].timeit(repeats) / repeats
                if t < best_avg_time[key]:
                    best_avg_time[key] = t

    print(f"\n\nBest average time out of {sets} sets of {repeats}:\n")
    print_row(
        "Benchmark",
        "Gdspy " + gdspy.__version__,
        "Gdstk " + gdstk.__version__,
        "Gain",
        hsep=True,
    )
    for name in bench_names:
        gdspy_time = best_avg_time[(name, "gdspy")]
        gdstk_time = best_avg_time[(name, "gdstk")]
        print_row(
            name,
            prefix_format(gdspy_time, unit="s"),
            prefix_format(gdstk_time, unit="s"),
            f"{gdspy_time / gdstk_time:.3g}",
        )


def memory_benchmark():
    proc = psutil.Process()
    total = 100000
    print(f"\nMemory usage per object for {total} objects:\n")

    def print_row(*vals, hsep=False):
        columns = [20, 16, 16, 9]
        print(
            "|",
            vals[0].ljust(columns[0]),
            "|",
            " | ".join(v.center(c) for v, c in zip(vals[1:], columns[1:])),
            "|",
        )
        if hsep:
            print(
                "|",
                ":" + "-" * (columns[0] - 1),
                "|",
                " | ".join(":" + "-" * (c - 2) + ":" for c in columns[1:]),
                "|",
            )

    print_row(
        "Object",
        "Gdspy " + gdspy.__version__,
        "Gdstk " + gdstk.__version__,
        "Reduction",
        hsep=True,
    )

    def mem_test(func):
        start_mem = proc.memory_info()
        r = [func(i) for i in range(total)]
        end_mem = proc.memory_info()
        return (end_mem.vms - start_mem.vms) / total, r

    data = []
    gdspy_cell = gdspy.Cell("TEMP", exclude_from_current=True)
    gdstk_cell = gdstk.Cell("TEMP")
    for obj, gdspy_func, gdstk_func in [
        (
            "Rectangle",
            lambda i: gdspy.Rectangle((i, i), (i + 1, i + 1)),
            lambda i: gdstk.rectangle((i, i), (i + 1, i + 1)),
        ),
        (
            "Circle (r = 10)",
            lambda i: gdspy.Round((0, 10 * i), 10),
            lambda i: gdstk.ellipse((0, 10 * i), 10),
        ),
        (
            "FlexPath segment",
            lambda i: gdspy.FlexPath([(i, i + 1), (i + 1, i)], 0.1),
            lambda i: gdstk.FlexPath([(i, i + 1), (i + 1, i)], 0.1),
        ),
        (
            "FlexPath arc",
            lambda i: gdspy.FlexPath([(10 * i, 0)], 0.1).arc(10, 0, numpy.pi),
            lambda i: gdstk.FlexPath([(10 * i, 0)], 0.1).arc(10, 0, numpy.pi),
        ),
        (
            "RobustPath segment",
            lambda i: gdspy.RobustPath((i, i + 1), 0.1).segment((i + 1, i)),
            lambda i: gdstk.RobustPath((i, i + 1), 0.1).segment((i + 1, i)),
        ),
        (
            "RobustPath arc",
            lambda i: gdspy.RobustPath((10 * i, 0), 0.1).arc(10, 0, numpy.pi),
            lambda i: gdstk.RobustPath((10 * i, 0), 0.1).arc(10, 0, numpy.pi),
        ),
        (
            "Label",
            lambda i: gdspy.Label(str(i), (i, i)),
            lambda i: gdstk.Label(str(i), (i, i)),
        ),
        (
            "Reference",
            lambda i: gdspy.CellReference(gdspy_cell, (0, 0)),
            lambda i: gdstk.Reference(gdstk_cell, (0, 0)),
        ),
        (
            "Reference (array)",
            lambda i: gdspy.CellArray(gdspy_cell, (0, 0), 1, 1, (0, 0)),
            lambda i: gdstk.Reference(gdstk_cell, (0, 0), rows=1, columns=1),
        ),
        (
            "Cell",
            lambda i: gdspy.Cell(str(i), exclude_from_current=True),
            lambda i: gdstk.Cell(str(i)),
        ),
    ]:
        gdspy_mem, gdspy_data = mem_test(gdspy_func)
        data.append(gdspy_data)
        gdstk_mem, gdstk_data = mem_test(gdstk_func)
        data.append(gdstk_data)
        print_row(
            obj,
            prefix_format(gdspy_mem, unit="B", base="bin"),
            prefix_format(gdstk_mem, unit="B", base="bin"),
            f"{100 - 100 * gdstk_mem / gdspy_mem:.0f}%",
        )


if __name__ == "__main__":
    print("Make sure the library is built in release mode!")
    timing_benchmark()
    memory_benchmark()
