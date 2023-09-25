##################################################
# Distributes modified Makefiles into Amptools to
#   produce a shared library
##################################################

# check for existence of ROOTSYS environment variable
ifndef ROOTSYS
$(error ROOTSYS environment variable is not set)
endif

SRC_DIRS := AmpTools

# # Default target: Distribute Makefiles and run 'make'
default: distribute_makefiles run_make

distribute_makefiles:
	@echo " === Distributing Makefiles... === "
	@for dir in $(shell ls -d ${SRC_DIRS}/*/); do \
		cp $$dir/Makefile ../AmpTools/$$dir; \
		echo "Copied $$dir/Makefile to ../AmpTools/$$dir"; \
	done

# Target to run 'make' in the target directory
run_make:
	@echo " === Running 'make' in the target directory... ==="
	@cd ../AmpTools/AmpTools; make

.PHONY: distribute_makefiles run_make default

clean:
	@echo " === Cleaning up target directory === "
	@cd ../AmpTools; make clean
