# Makefile

VERSION=1.1.2

FORCE:
	make install
	make test

build:
	python3 -m build

install:
	make build
	python3 -m pip install ./dist/jsmfsb-$(VERSION).tar.gz

test:
	pytest tests/

publish:
	make build
	python3 -m twine upload dist/*$(VERSION)*

format:
	black .

check:
	ruff check --select N
	ruff check

edit:
	emacs Makefile *.toml *.md src/jsmfsb/*.py demos/*.py tests/*.py &

todo:
	grep TODO: demos/*.py src/jsmfsb/*.py tests/*.py



# eof
