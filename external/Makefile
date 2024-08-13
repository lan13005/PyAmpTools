##################################################
# Distributes modified Makefiles into Amptools to
#   produce a shared library
##################################################

# check for existence of ROOTSYS environment variable
ifndef ROOTSYS
$(error ROOTSYS environment variable is not set. If ROOT is already built, please source set_environment.sh)
endif

# SUPPORTED AMPTOOLS / FSROOT VERSIONS
# Features introduced beyond these commits might get overwritten
# by the distribution process and can crash the build
AMPTOOLS_COMMIT := 0753e28 # master as of 07/20/2024
FSROOT_COMMIT := d4d7e73

SRC_DIRS := Distribution
SUB_DIRS := IUAmpTools IUAmpToolsMPI MinuitInterface UpRootMinuit
GPU_DIRS := GPUManager

# run_default acts as a guard for default recipe to prevent double executions, I think
.PHONY: run_default
run_default: default
	@echo " === Default target already executed ==="

# Default target: Distribute Makefiles and run 'make'
.PHONY: default
default: distribute_makefiles
	@echo " === Running 'make' in the target directory... ==="
	@cd AmpTools/AmpTools; make
	@cd AmpTools/AmpPlotter; make
	@cd FSRoot; make
	@cd FSRoot/Executables; make
	@cd AMPTOOLS_AMPS_DATAIO; make
	@cd AMPTOOLS_GENERATORS; make

mpi: | run_default
	@echo " === Running 'make mpi' in the target directory... ==="
	@cd AmpTools/AmpTools; make MPI=1
	@cd AmpTools/AmpPlotter; make
	@cd AMPTOOLS_AMPS_DATAIO; make MPI=1

gpu: | run_default
# gpu: distribute_makefiles
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
	@echo " === Checking out Amptools and FSRoot... === "
	@echo " === Using Amptools commit: $(AMPTOOLS_COMMIT) === "
	@echo " === Using FSRoot commit: $(FSROOT_COMMIT) === "
	@echo " === Checking out Amptools... === "
	@cd AmpTools; git reset --hard; git checkout $(AMPTOOLS_COMMIT)
	@echo " === Checking out FSRoot... === "
	@cd FSRoot; git reset --hard; git checkout $(FSROOT_COMMIT)

	@echo " === Distributing Makefiles and modified source codes... === "
### Distribute AmpTools makefiles
	cp $(SRC_DIRS)/AmpTools_Base_Makefile AmpTools/Makefile
	cp $(SRC_DIRS)/Makefile.settings AmpTools/Makefile.settings
	@for dir in $(SUB_DIRS); do \
		cp $(SRC_DIRS)/AmpTools_X_Makefile AmpTools/AmpTools/$$dir/Makefile; \
		echo "Copied $(SRC_DIRS)/AmpTools_X_Makefile to AmpTools/AmpTools/$$dir/Makefile"; \
	done
	@for dir in $(GPU_DIRS); do \
		cp $(SRC_DIRS)/AmpTools_GPU_Makefile AmpTools/AmpTools/$$dir/Makefile; \
		echo "Copied $(SRC_DIRS)/AmpTools_GPU_Makefile to AmpTools/AmpTools/$$dir/Makefile"; \
	done
	cp $(SRC_DIRS)/AmpTools_Makefile AmpTools/AmpTools/Makefile
### Distribute AmpPlotter makefiles
	cp $(SRC_DIRS)/AmpPlotter_Makefile AmpTools/AmpPlotter/Makefile
	cp $(SRC_DIRS)/AmpPlotter_X_Makefile AmpTools/AmpPlotter/AmpPlotter/Makefile
### Pythonization + ease of modifying parameters + GradientManager
	cp $(SRC_DIRS)/ParameterManager.* AmpTools/AmpTools/IUAmpTools
### Extract amplitude weights
	cp $(SRC_DIRS)/PlotGenerator.* AmpTools/AmpTools/IUAmpTools
### kDRModule cannot be found (template). Replace with char
	cp $(SRC_DIRS)/DataReaderMPI.h AmpTools/AmpTools/IUAmpToolsMPI
### Add getReal and getImag methods
	cp $(SRC_DIRS)/ComplexParameter.* AmpTools/AmpTools/IUAmpTools
### Not freed m_pdDevNICalc and delete m_pcCalcAmp
	cp $(SRC_DIRS)/GPUManager.* AmpTools/AmpTools/GPUManager
### Add more report statements
	cp $(SRC_DIRS)/AmpToolsInterface.* AmpTools/AmpTools/IUAmpTools
	cp $(SRC_DIRS)/MinuitMinimizationManager.* AmpTools/AmpTools/MinuitInterface
	cp $(SRC_DIRS)/MISubject.* AmpTools/AmpTools/MinuitInterface
	cp $(SRC_DIRS)/URMinuit.* AmpTools/AmpTools/UpRootMinuit
	cp $(SRC_DIRS)/AmpVecs.* AmpTools/AmpTools/IUAmpTools
### UserDataReader needs a method to clear m_dataReaderInstances
	cp $(SRC_DIRS)/UserDataReader.h AmpTools/AmpTools/IUAmpTools
	cp $(SRC_DIRS)/DataReader.h AmpTools/AmpTools/IUAmpTools
### Allow report to dump to a log file if AMPTOOLS_LOG_FILE env variable is set. i.e. AMPTOOLS_LOG_FILE="log.txt"
	cp $(SRC_DIRS)/report.cc AmpTools/AmpTools/IUAmpTools
### Introduce GradientCalculator
	cp $(SRC_DIRS)/GradientCalculator.* AmpTools/AmpTools/IUAmpTools
### Allow getting of m_termNames
	cp $(SRC_DIRS)/NormIntInterface.h AmpTools/AmpTools/IUAmpTools
### Allow getting of signal / bkgnd ampvecs
	cp $(SRC_DIRS)/LikelihoodCalculator.h AmpTools/AmpTools/IUAmpTools
### Distribute FSRoot makefiles and src files
	cp $(SRC_DIRS)/FSROOT_Makefile FSRoot/Makefile
	cp $(SRC_DIRS)/initialize_fsroot.* FSRoot

.PHONY: distribute_makefiles default mpi gpu mpigpu gpumpi clean

clean:
	@echo " === Cleaning up target directory === "
	@cd AmpTools; git reset --hard; make clean
	@cd AMPTOOLS_AMPS_DATAIO; make clean
	@cd FSRoot; git reset --hard; make clean
	@cd FSRoot/Executables; make clean
	@cd AMPTOOLS_GENERATORS; make clean
