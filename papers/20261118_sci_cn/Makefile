LATEX ?= xelatex
BIBTEX ?= bibtex

.PHONY: all pdf
all: pdf

pdf:
	mkdir -p output
	$(LATEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=output main.tex
	$(BIBTEX) output/main
	$(LATEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=output main.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=output main.tex
	cp output/main.pdf output/中文论文.pdf
