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
