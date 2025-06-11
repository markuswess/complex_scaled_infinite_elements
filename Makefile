DEPENDS= intro.md _config.yml _toc.yml logo.png 
.PHONY: clean

_build: $(DEPENDS)
	jupyter-book build --all .
