LIB_SRC=$(wildcard src/*.cpp) src/clipperlib/clipper.cpp
LIB_HDR=$(wildcard src/*.h) src/clipperlib/clipper.hpp
LIB_BUILD_PREFIX=cmake_build
LIB_INSTALL_PREFIX=$(LIB_BUILD_PREFIX)/install
LIB=$(LIB_INSTALL_PREFIX)/lib/libgdstk.a

PY_SRC=$(wildcard python/*.cpp)

IMG_SRC=$(wildcard docs/*_images.py) docs/pcell.py docs/photonics.py docs/merging.py docs/transforms.py docs/repetitions.py docs/apply_repetition.py docs/fonts.py docs/pos_filtering.py docs/path_markers.py docs/pads.py
DOCS_SRC=$(wildcard docs/*.rst)
DOCS=docs/_build/html/index.html

CPP_EXAMPLES=$(wildcard docs/cpp/*.cpp)

CFLAGS+=-O3
CXXFLAGS+=-O3
CMAKE_BUILD_TYPE=Release
PYTHON_RELEASE=

# CFLAGS+=-ggdb -Og
# CXXFLAGS+=-ggdb -Og
# CMAKE_BUILD_TYPE=Debug
# PYTHON_RELEASE=--debug

default: module

clean:
	-rm -rf build dist gdstk.egg-info
	-rm -rf docs/_build/* docs/geometry/* docs/library/*
	-rm -rf *.svg
	-rm -rf *.gds
	-rm -rf *.oas
	-rm -rf *.out
	-rm -rf docs/*.gds
	-rm -rf docs/*.svg
	-rm -rf docs/*/*.svg
	-rm -rf docs/cpp/*.svg
	-rm -rf docs/cpp/*.gds
	-rm -rf $(LIB_BUILD_PREFIX)
	-rm $(CPP_EXAMPLES:.cpp=.out)

all: module test docs examples

module: $(PY_SRC) $(LIB_SRC) $(LIB_HDR) Makefile
	python setup.py build $(PYTHON_RELEASE)

docs: $(DOCS)

test: module
	pytest

examples: $(LIB) $(CPP_EXAMPLES:.cpp=.run)

valgrind: $(LIB) $(CPP_EXAMPLES:.cpp=.grind)

$(LIB): $(LIB_SRC) $(LIB_HDR) Makefile
	cmake -S . -B $(LIB_BUILD_PREFIX) -DCMAKE_INSTALL_PREFIX=$(LIB_INSTALL_PREFIX) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	cmake --build $(LIB_BUILD_PREFIX) --target install

$(DOCS): module $(DOCS_SRC) $(IMG_SRC) docs/layout.py docs/filtering.py
	python setup.py build_sphinx

docs/filtering.py: module docs/layout.py
	python $@

docs/layout.py: module docs/photonics.py
	python $@

$(IMG_SRC): module
	python $@

%.out: %.cpp $(LIB)
	$(CXX) $(CXXFLAGS) -o $@ $< -I$(LIB_INSTALL_PREFIX)/include -L$(LIB_INSTALL_PREFIX)/lib -llapack -lpthread -lm -ldl -lgdstk /usr/lib/libz.a

%.run: %.out
	-./$<

%.grind: %.out
	valgrind --undef-value-errors=no --leak-check=full --error-exitcode=1 --quiet ./$<

release:
	git push
	git push origin --tags

.PHONY: default clean all docs test examples valgrind
.PRECIOUS: %.out
