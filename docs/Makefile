# Variables
BOOK_DIR := .
BUILD_DIR := $(BOOK_DIR)/_build
AUTOSUM_DIR := $(BOOK_DIR)/_autosummary

Q := @

# Default target
all: build

# Build the Jupyter Book
build:
	@echo "Building Jupyter Book..."
	mkdir -p $(AUTOSUM_DIR)
	cp Distribution/module.rst $(AUTOSUM_DIR)
	jupyter-book build $(BOOK_DIR)

push:
	ghp-import -n -p -f $(BUILD_DIR)/html

update: build push

# Clean the build directory
clean:
	@echo "Cleaning up build directory..."
	@-rm -rf $(BUILD_DIR)
	@-rm -rf $(AUTOSUM_DIR)

# The .PHONY rule keeps make from doing something with a file named clean
.PHONY: all build clean serve
