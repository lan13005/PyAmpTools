##################################################
# Distributes modified Makefiles into Amptools to
#   produce a shared library
##################################################

# check for existence of ROOTSYS environment variable
ifndef ROOTSYS
$(error ROOTSYS environment variable is not set)
endif

SRC_DIRS := Distribution
SUB_DIRS := IUAmpTools IUAmpToolsMPI MinuitInterface UpRootMinuit

# # Default target: Distribute Makefiles and run 'make'
default: distribute_makefiles
	@echo " === Running 'make' in the target directory... ==="
	@cd ../AmpTools/AmpTools; make

mpi: default
	@echo " === Running 'make mpi' in the target directory... ==="
	@cd ../AmpTools/AmpTools; make MPI=1

gpu:
	@echo " === Running 'make gpu' in the target directory... ==="
	@cd ../AmpTools/AmpTools; make GPU=1

mpigpu: gpu
	@echo " === Running 'make mpigpu' in the target directory... ==="
	@cd ../AmpTools/AmpTools; make GPU=1 MPI=1

gpumpi: mpigpu

distribute_makefiles:
	@echo " === Distributing Makefiles... === "
	@for dir in $(SUB_DIRS); do \
		cp $(SRC_DIRS)/AmpTools_X_Makefile ../AmpTools/AmpTools/$$dir/Makefile; \
		echo "Copied $(SRC_DIRS)/AmpTools_X_Makefile to ../AmpTools/AmpTools/$$dir/Makefile"; \
	done
	cp $(SRC_DIRS)/AmpTools_Makefile ../AmpTools/AmpTools/Makefile

.PHONY: distribute_makefiles default mpi gpu mpigpu gpumpi clean

clean:
	@echo " === Cleaning up target directory === "
	@cd ../AmpTools; make clean
