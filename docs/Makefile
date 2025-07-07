# Makefile for TSBee documentation

# You can set these variables from the command line
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = build

# Put it first so that "make" without argument is like "make help"
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets
clean:
	rm -rf $(BUILDDIR)/*

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

livehtml:
	sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html

install:
	uv pip install --group docs

dev-install:
	uv pip install --group docs
	uv pip install sphinx-autobuild

build-package:
	cd .. && bash -c "source .venv/bin/activate && maturin develop --release"

full-build: build-package html

serve: html
	cd $(BUILDDIR)/html && python -m http.server 8000

.PHONY: clean html livehtml install dev-install build-package full-build serve
