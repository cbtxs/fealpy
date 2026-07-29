TEX := hu_zhang_metaprogramming.tex
PDF := $(TEX:.tex=.pdf)
LATEXMK := latexmk

.PHONY: all pdf clean distclean help

all: pdf

pdf: $(PDF)

$(PDF): $(TEX)
	$(LATEXMK) -g -xelatex -interaction=nonstopmode -halt-on-error $(TEX)
	nohup evince $(PDF) >/dev/null 2>&1 &

clean:
	$(LATEXMK) -c $(TEX)

distclean:
	$(LATEXMK) -C $(TEX)

help:
	@printf '%s\n' 'make / make pdf   Build the PDF with XeLaTeX' 'make clean        Remove temporary build files' 'make distclean    Also remove the generated PDF'
