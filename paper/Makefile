.PHONY: pdf clean

MAIN = loato_bench

pdf: $(MAIN).pdf

# Try latexmk first, fall back to tectonic
$(MAIN).pdf: $(MAIN).tex references.bib figures/*.pdf
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode $(MAIN).tex; \
	elif command -v tectonic >/dev/null 2>&1; then \
		tectonic $(MAIN).tex; \
	else \
		echo "Error: neither latexmk nor tectonic found. Install one:"; \
		echo "  brew install --cask mactex-no-gui  # or"; \
		echo "  brew install tectonic"; \
		exit 1; \
	fi

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
		$(MAIN).run.xml $(MAIN).synctex.gz $(MAIN).fdb_latexmk $(MAIN).fls \
		$(MAIN).xdv $(MAIN).pdf
