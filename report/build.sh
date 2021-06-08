#!/bin/bash
bibtex report
pdflatex report.tex

rm *.aux *.log *.out
