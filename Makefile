AMPTOOLS_AMPS_DATAIO := $(shell pwd)/src/GlueX
ifndef AMPTOOLS_HOME
  AMPTOOLS_HOME := $(shell pwd)/external/AmpTools
endif
AMPTOOLS := $(AMPTOOLS_HOME)/AmpTools

export

.PHONY: default clean

default:
	@echo "$(ATSUFFIX)"
	@echo "=== Building src directory ==="
	@$(MAKE) -C src/GlueX
	@echo "=== Building script directory ==="
	@$(MAKE) -C scripts

# mpi: default
# 	@echo "=== Building src directory with MPI ==="
# 	@$(MAKE) -C src/GlueX MPI=1
# 	@echo "=== Building script directory with MPI ==="
# 	# @$(MAKE) -C scripts MPI=1

# gpu:
# 	@echo "=== Building src directory with GPU acceleration ==="
# 	@$(MAKE) -C src/GlueX GPU=1
# 	@echo "=== Building script directory with GPU acceleration ==="
# 	# @$(MAKE) -C scripts GPU=1

# mpigpu: gpu
# 	@echo "=== Building src directory with MPI and GPU acceleration ==="
# 	@$(MAKE) -C src/GlueX GPU=1 MPI=1
# 	@echo "=== Building script directory with MPI and GPU acceleration ==="
# 	# @$(MAKE) -C scripts GPU=1 MPI=1
	
# gpumpi: mpigpu

clean:
	@$(MAKE) -C src/GlueX clean
	# @$(MAKE) -C scripts clean
