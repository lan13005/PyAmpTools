# Variables
BOOK_DIR := .
BUILD_DIR := $(BOOK_DIR)/_build
AUTOSUM_DIR := $(BOOK_DIR)/_autosummary

Q := @

# Default target
all: build

# Build the Jupyter Book
# CCDB uses a different sqlalchemy version which gets appended into PYTHONPATH. We have to remove it to avoid conflicts
#    We run in a subshell to avoid changing the user's PYTHONPATH
build:
	@echo "Building Jupyter Book..."
	@PYTHONPATH=$$(echo $$PYTHONPATH | tr ':' '\n' | grep -v 'ccdb/' | tr '\n' ':' | sed 's/:$$//'); \
	export PYTHONPATH; \
	jupyter-book build $(BOOK_DIR)

push:
	ghp-import -n -p -f $(BUILD_DIR)/html

update: build push

rebuild: clean build push

# Clean the build directory
clean:
	@echo "Cleaning up build directory..."
	@-rm -rf $(BUILD_DIR)
	@-rm -rf $(AUTOSUM_DIR)

# The .PHONY rule keeps make from doing something with a file named clean
.PHONY: all build clean serve
