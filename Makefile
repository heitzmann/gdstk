default: test

clean:
	-rm -rf build dist gdstk.egg-info src/gdstk.egg-info
	-rm -rf docs/_build docs/geometry/* docs/library/*
	-rm -rf *.svg
	-rm -rf *.gds
	-rm -rf *.oas
	-rm -rf *.out
	-rm -rf docs/*.gds
	-rm -rf docs/*.svg
	-rm -rf docs/*/*.svg

all: test docs examples

lib:
	cmake -S . -B build -G Ninja -DCMAKE_INSTALL_PREFIX=build -DCMAKE_BUILD_TYPE=Debug
	cmake --build build --target install

module:
	python -m build -w

docs: module
	sphinx-build docs docs/_build

test: module
	python -m pytest

examples: lib
	cmake --build build --target examples
	cmake --build build --target test

%.out: %.cpp lib
	$(CXX) $(CXXFLAGS) -o $@ $< $(shell pkg-config --with-path=build --cflags gdstk) $(shell pkg-config --with-path=build --libs gdstk)

%.run: %.out
	-./$<

%.grind: %.out
	valgrind --undef-value-errors=no --leak-check=full --error-exitcode=1 --quiet ./$<

.PHONY: default clean all lib module docs test examples
.PRECIOUS: %.out
