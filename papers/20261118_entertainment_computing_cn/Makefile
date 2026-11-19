LATEX ?= /Library/TeX/texbin/xelatex
BIBTEX ?= /Library/TeX/texbin/bibtex
REPO_ROOT ?=
PYTHON ?= $(REPO_ROOT)/.venv/bin/python

.NOTPARALLEL:
.PHONY: all pdf supplement evidence verify render
all: pdf supplement

pdf:
	mkdir -p output
	$(LATEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=output main.tex > output/main-pass1.stdout
	$(BIBTEX) output/main > output/main-bibtex.stdout
	$(LATEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=output main.tex > output/main-pass2.stdout
	$(LATEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=output main.tex > output/main-pass3.stdout
	cp output/main.pdf output/中文论文.pdf

supplement:
	mkdir -p output
	$(LATEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=output supplement.tex > output/supplement-pass1.stdout
	$(BIBTEX) output/supplement > output/supplement-bibtex.stdout
	$(LATEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=output supplement.tex > output/supplement-pass2.stdout
	$(LATEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=output supplement.tex > output/supplement-pass3.stdout
	cp output/supplement.pdf output/补充材料.pdf

evidence:
	@test -n "$(REPO_ROOT)" || (echo 'Set REPO_ROOT to the absolute repository root'; exit 2)
	$(PYTHON) -B scripts/bind_round2_evidence.py --repo-root "$(REPO_ROOT)" --paper-root "$(CURDIR)"

verify:
	@test -n "$(REPO_ROOT)" || (echo 'Set REPO_ROOT to the absolute repository root'; exit 2)
	$(PYTHON) -B scripts/verify_round2.py --paper-root "$(CURDIR)" --repo-root "$(REPO_ROOT)"

render:
	@test -n "$(REPO_ROOT)" || (echo 'Set REPO_ROOT to the absolute repository root'; exit 2)
	$(PYTHON) -B scripts/render_round2.py --paper-root "$(CURDIR)"
