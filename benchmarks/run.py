#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020-2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import sys
import resource
import gc
import timeit
import numpy
import pathlib
import importlib


def engineering(value, stdev, fmtstr):
    if value == 0:
        m = 0
        prefix = ""
    else:
        m = int(numpy.floor(numpy.log10(abs(value))))
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
        }.get(m, f"× 10^{m} ")
    value *= 10 ** -m
    stdev *= 10 ** -m
    return fmtstr.format(value=value, stdev=stdev, prefix=prefix)


if __name__ == "__main__":
    path = pathlib.Path(__file__).absolute()
    bench_files = sorted([f for f in path.parent.glob("*.py") if f != path])

    def print_row(*vals):
        columns = [16, 12, 12, 8]
        print(
            vals[0].ljust(columns[0]),
            " ".join(v.center(c) for v, c in zip(vals[1:], columns[1:])),
            flush=True,
        )

    repeats = 32
    sets = 32
    print(f"Best average time out of {sets} sets of {repeats} runs.")
    print_row("Benchmark", "Gdspy", "Gdstk", "Gain")
    for fname in bench_files:
        module = importlib.import_module(fname.stem)
        best = {}
        formated = {}
        for mod in ("gdspy", "gdstk"):
            func = getattr(module, f"bench_{mod}")
            timer = timeit.Timer("func()", "gc.enable()", globals=globals())
            elapsed = 1e300
            for _ in range(sets):
                t = timer.timeit(repeats)
                elapsed = min(elapsed, t / repeats)
            best[mod] = elapsed
            formated[mod] = engineering(
                best[mod],
                0,
                "{value:.3g} {prefix}s",
            )
        print_row(
            fname.stem,
            formated["gdspy"],
            formated["gdstk"],
            f"{best['gdspy'] / best['gdstk']:.3g}",
        )
