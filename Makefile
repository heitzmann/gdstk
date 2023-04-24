default: module

help:
	@echo 'make library:		Build C++ library in build/'
	@echo 'make module:			Build python library'
	@echo 'make test:       	Run tests with pytest'

library:
	cmake -S . -B build
	cmake --build build

module:
	python setup.py build

clean:
	-rm -r build

install:
	pip install -r requirements.txt
	pip install -r requirements_dev.txt
	pip install -e .
	#pre-commit install

test:
	flake8 python
	pytest

docs:
	python docs/cell_images.py
	python docs/curve_images.py
	python docs/flexpath_images.py
	python docs/function_images.py
	python docs/label_images.py
	python docs/polygon_images.py
	python docs/reference_images.py
	python docs/robustpath_images.py
	python docs/tutorial_images.py
	python docs/pcell.py
	python docs/photonics.py
	python docs/layout.py
	python docs/merging.py
	python docs/transforms.py
	python docs/repetitions.py
	python docs/apply_repetition.py
	python docs/filtering.py
	python docs/pos_filtering.py
	python docs/path_markers.py
	python docs/pads.py
	python docs/fonts.py
	python setup.py build_sphinx

release:
	git push
	git push origin --tags

.PHONY: docs
