# Makefile

SOURCE=$(wildcard *.py)

TARGETS=$(patsubst %.py,%.txt,$(SOURCE))

FORCE:
	make $(TARGETS)


%.txt: %.py
	python3 $< > $@

edit:
	emacs Makefile *.py &

clean:
	rm -f *.txt *.pdf


# eof

