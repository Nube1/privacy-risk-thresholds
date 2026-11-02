# Everything the book and the paper print is generated from this repository's code.
.PHONY: reproduce reproduce-fast test book numbers
reproduce-fast:
	python3 reproduce.py --fast
reproduce:
	python3 reproduce.py --full
test:
	PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider -o addopts="" tests --ignore=tests/test_realdata.py
numbers:
	PYTHONPATH=src python3 scripts/book_numbers.py --check
book:
	cd book_revised && latexmk -pdf -interaction=nonstopmode main.tex && latexmk -pdf -interaction=nonstopmode main.tex
