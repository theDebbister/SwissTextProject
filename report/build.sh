#!/bin/bash
pdflatex acl2021.tex
bibtex acl2021
pdflatex acl2021.tex
rm *.aux *.log *.out
