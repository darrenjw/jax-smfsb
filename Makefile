# Makefile

VERSION=0.0.1

FORCE:
	make install
	make test

build:
	python3 -m build

install:
	make build
	python3 -m pip install ./dist/jax-smfsb-$(VERSION).tar.gz

test:
	pytest tests/

publish:
	make build
	python3 -m twine upload dist/*$(VERSION)*


edit:
	emacs Makefile *.toml *.md src/jax-smfsb/*.py tests/*.py &


# eof
