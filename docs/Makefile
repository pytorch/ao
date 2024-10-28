# Minimal makefile for Sphinx documentation
#

ifneq ($(EXAMPLES_PATTERN),)
    EXAMPLES_PATTERN_OPTS := -D sphinx_gallery_conf.filename_pattern="$(EXAMPLES_PATTERN)"
endif

# You can set these variables from the command line.

# TODO: Revert this when have docs on pytorch.org/ao
# SPHINXOPTS    = -W -j auto $(EXAMPLES_PATTERN_OPTS)
# SPHINXOPTS    = -WT -j auto --keep-going # enable later when the files are included in the doc build 


SPHINXBUILD   = sphinx-build
SPHINXPROJ    = torchao
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html-noplot:  # Avoids running the gallery examples, which may take time
	$(SPHINXBUILD) -D plot_gallery=0 -b html "${SOURCEDIR}" "$(BUILDDIR)"/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(SOURCEDIR)/generated_examples/  # sphinx-gallery
	rm -rf $(SOURCEDIR)/gen_modules/  # sphinx-gallery
	rm -rf $(SOURCEDIR)/sg_execution_times.rst  # sphinx-gallery
	rm -rf $(SOURCEDIR)/generated/  # autosummary

.PHONY: help Makefile docset

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
