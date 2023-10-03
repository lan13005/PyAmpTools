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
GPU_DIRS := GPUManager

# # Default target: Distribute Makefiles and run 'make'
default: distribute_makefiles
	@echo " === Running 'make' in the target directory... ==="
	@cd AmpTools/AmpTools; make
	@cd AmpTools/AmpPlotter; make
	@cd FSRoot; make
	@cd FSRroot/Executables; make
	@cd AMPTOOLS_AMPS_DATAIO; make

# mpi and gpu both requiring default might double run things
mpi: default
	@echo " === Running 'make mpi' in the target directory... ==="
	@cd AmpTools/AmpTools; make MPI=1
	@cd AmpTools/AmpPlotter; make
	@cd AMPTOOLS_AMPS_DATAIO; make MPI=1

gpu: default
	@echo " === Running 'make gpu' in the target directory... ==="
	@cd AmpTools/AmpTools; make GPU=1
	@cd AmpTools/AmpPlotter; make
	@cd AMPTOOLS_AMPS_DATAIO; make GPU=1

mpigpu: gpu mpi
	@echo " === Running 'make mpigpu' in the target directory... ==="
	@cd AmpTools/AmpTools; make GPU=1 MPI=1
	@cd AmpTools/AmpPlotter; make
	@cd AMPTOOLS_AMPS_DATAIO; make GPU=1 MPI=1

gpumpi: mpigpu

distribute_makefiles:
	@echo " === Distributing Makefiles... === "
	@for dir in $(SUB_DIRS); do \
		cp $(SRC_DIRS)/AmpTools_X_Makefile AmpTools/AmpTools/$$dir/Makefile; \
		echo "Copied $(SRC_DIRS)/AmpTools_X_Makefile to AmpTools/AmpTools/$$dir/Makefile"; \
	done
	@for dir in $(GPU_DIRS); do \
		cp $(SRC_DIRS)/AmpTools_GPU_Makefile AmpTools/AmpTools/$$dir/Makefile; \
		echo "Copied $(SRC_DIRS)/AmpTools_GPU_Makefile to AmpTools/AmpTools/$$dir/Makefile"; \
	done
	cp $(SRC_DIRS)/AmpTools_Makefile AmpTools/AmpTools/Makefile
	cp $(SRC_DIRS)/ParameterManager.* AmpTools/AmpTools/IUAmpTools
	cp $(SRC_DIRS)/PlotGenerator.* AmpTools/AmpTools/IUAmpTools
	cp $(SRC_DIRS)/DataReaderMPI.h AmpTools/AmpTools/IUAmpToolsMPI
	cp $(SRC_DIRS)/FSROOT_Makefile FSRoot/Makefile

.PHONY: distribute_makefiles default mpi gpu mpigpu gpumpi clean

clean:
	@echo " === Cleaning up target directory === "
	@cd AmpTools; make clean
	@cd AMPTOOLS_AMPS_DATAIO; make clean
	@cd FSRoot; make clean
	@cd FSRroot/Executables; make clean
