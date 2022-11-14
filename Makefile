help:
	@echo 'make install:          Install package, hook, notebooks and gdslib'
	@echo 'make test:             Run tests with pytest'
	@echo 'make test-force:       Rebuilds regression test'

install:
	pip install -r requirements.txt
	pip install -r requirements_dev.txt
	pip install -e .
	pre-commit install

test:
	flake8 python
	pytest -s

lint:
	flake8 python

pylint:
	pylint

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

release:
	git push
	git push origin --tags

spell:
	codespell -i 3 -w -L TE,TE/TM,te,ba,FPR,fpr_spacing

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

.PHONY: docs
