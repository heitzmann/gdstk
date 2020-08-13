PY_SRC=$(wildcard python/*.cpp)
LIB_SRC=$(wildcard src/*)
IMG_SRC=$(wildcard docs/*_images.py)
DOCS_SRC=$(wildcard docs/*.rst)
PY_MOD=build/lib.linux-x86_64-3.8/gdstk.cpython-38-x86_64-linux-gnu.so
DOCS=docs/_build/html/index.html

default: all

clean:
	-rm -rf build
	-rm -rf docs/_build/* docs/_static/*/*.svg docs/geometry/* docs/library/*

all: $(PY_MOD) test docs

docs: $(DOCS)

test: $(PY_MOD)
	pytest

$(DOCS): $(PY_MOD) $(DOCS_SRC) $(IMG_SRC)
	python setup.py build_sphinx

$(PY_MOD): $(PY_SRC) $(LIB_SRC)
	-rm -rf build
	python setup.py build

$(IMG_SRC): $(PY_MOD)
	python $@

.PHONY: default clean all docs test
