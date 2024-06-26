SHELL = /bin/bash # Use bash syntax

ifndef AMPTOOLS_HOME
$(error Please set AMPTOOLS_HOME to point to the root of the AmpTools source tree. )
endif

include $(AMPTOOLS_HOME)/Makefile.settings
######################################################################################
Q := @

####### SET ENVIORNMENT VARIALBES ########
PYAMPTOOLS_HOME := $(shell echo $$PYAMPTOOLS_HOME)
ifndef PYAMPTOOLS_HOME
$(error Please set PYAMPTOOLS_HOME by running set_environment.py in the base directory)
endif
ifndef ROOTSYS
$(error ROOTSYS is not set. Please load your root environment first)
endif
AMPTOOLS_DATAIO := $(PYAMPTOOLS_HOME)/external/AMPTOOLS_AMPS_DATAIO
ifndef AMPTOOLS_HOME
  AMPTOOLS_HOME := $($PYAMPTOOLS_HOME)/external/AmpTools
endif
AMPTOOLS := $(AMPTOOLS_HOME)/AmpTools
AMPPLOTTER := $(AMPTOOLS_HOME)/AmpPlotter

# Consider all subdirectories when searching for source files
SRC_DIRS := $(shell find . -type d -not -path "./IN_PROGRESS/*")
SRC_DIRS := $(subst ./,,$(SRC_DIRS))

######## SET COMPILER AND LIBRARY LOCATIONS ########
INC_DIR  := -I$(AMPTOOLS_DATAIO) -I$(shell root-config --incdir) -I$(AMPTOOLS) -I$(AMPPLOTTER)

CXX_FLAGS += -Wall -shared -fPIC -std=c++17
GPU_ACCEL :=
ROOTLIBS  := $(shell root-config --libs)
AMPTOOLS_HEADERS := $(shell find $(AMPTOOLS) -type f -name '*.h' -not -name '*GPU*' -not -name '*MPI*' -not -name '*LinkDef*' -not -name "MainPage.h")

ifdef GPU
AMPTOOLS_HEADERS += $(shell find $(AMPTOOLS) -type f -name '*GPU*.h')
GPU_ACCEL := -DGPU_ACCELERATION
INC_DIR += -I$(CUDA_INSTALL_PATH)/include
CXX_FLAGS += $(GPU_ACCEL) -I$(CUDA_INSTALL_PATH)/include
SUFFIX := _GPU
NVCC_FLAGS += --compiler-options '-fPIC' --shared -std=c++17
CUDA_SOURCES := $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.cu))
CUDA_OBJECTS := $(CUDA_SOURCES:.cu=$(SUFFIX).o)
endif

ifdef MPI
CXX := $(MPICXX)
CXX_FLAGS += -DUSE_MPI
AMPTOOLS_HEADERS += $(shell find $(AMPTOOLS) -type f -name '*MPI*.h')
MPI_SHOWME := $(shell $(CXX) -showme)
MPI_INCLUDE := $(shell echo $(MPI_SHOWME) | cut -d ' ' -f 2)
INC_DIR += $(MPI_INCLUDE)
AMPTOOLS_HEADERS += mpi.h
SUFFIX := $(SUFFIX)_MPI
endif

CXX_FLAGS += $(INC_DIR)
NVCC_FLAGS += $(INC_DIR)

AMPTOOLS_HEADERS := $(patsubst $(AMPTOOLS)/%,%,$(AMPTOOLS_HEADERS))
AMPPLOTTER_HEADERS := $(shell find $(AMPPLOTTER) -type f -name '*.h')
AMPPLOTTER_HEADERS := $(patsubst $(AMPPLOTTER)/%,%,$(AMPPLOTTER_HEADERS))

AMPTOOLS_LIBS := -L$(AMPTOOLS)/lib -lAmpTools$(SUFFIX)
AMPPLOTTER_LIBS := -L$(AMPPLOTTER)/lib -lAmpPlotter

######## SET SOURCES AND HEADER FILES ########
# Source Files, deriving object and header files from them
SOURCES := $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.cc))
SOURCES := $(subst ./,,$(SOURCES))
OBJECTS = $(SOURCES:.cc=$(SUFFIX).o)
HEADERS = $(SOURCES:.cc=.h)

######## SET TARGET LIBRARY NAME AND DICTIONARY SOURCE FILE ########
TARGET_LIBRARY = libAmpsDataIO$(SUFFIX).so # Output library name
TARGET_DICT = AmpsDataIODict$(SUFFIX).cxx  # Dictionary for PyROOT to locate library objects
TARGET_LINKDEF = AmpsDataIODict$(SUFFIX)_Linkdef.h  # This file is need to create the dictionary

$(info SRC_DIRS is $(SRC_DIRS))
$(info SOURCES is $(SOURCES))
$(info OBJECTS is $(OBJECTS))
$(info CUDA_SOURCES is $(CUDA_SOURCES))
$(info CUDA_OBJECTS is $(CUDA_OBJECTS))
$(info CXX_FLAGS is $(CXX_FLAGS))

######## MAKE RULES ########
.PHONY: all clean

# Targets
all: $(TARGET_LIBRARY)

$(TARGET_LIBRARY): $(OBJECTS) $(CUDA_OBJECTS) $(TARGET_DICT)
	@echo " ==== Creating final library $@ ==== "
	$(Q)$(CXX) $(CXX_FLAGS) -o $@ $^ $(ROOTLIBS) $(AMPTOOLS_LIBS) $(AMPPLOTTER_LIBS)

$(TARGET_DICT): $(HEADERS)
	@echo " ==== Creating Dictionary $@ ==== "

	@echo "#ifdef __CLING__" > ${TARGET_LINKDEF}
	@for header in $^; do \
		echo "#pragma link C++ defined_in \"$$header\";" >> ${TARGET_LINKDEF}; \
	done
	@for header in $(AMPTOOLS_HEADERS); do \
		echo "#pragma link C++ defined_in \"$$header\";" >> ${TARGET_LINKDEF}; \
	done
	@for header in $(AMPPLOTTER_HEADERS); do \
		echo "#pragma link C++ defined_in \"$$header\";" >> ${TARGET_LINKDEF}; \
	done
	@echo "#endif" >> ${TARGET_LINKDEF}

	$(Q)rootcling -f $(INC_DIR) $@ $^ $(AMPTOOLS_HEADERS) $(GPU_ACCEL) $(AMPPLOTTER_HEADERS) ${TARGET_LINKDEF}

%$(SUFFIX).o: %.cc
	$(vecho) "-> Compiling $<"
	$(Q)$(CXX) $(CXX_FLAGS) -MMD -MP -c $< -o $@

%$(SUFFIX).o: %.cu
	$(vecho) "-> Compiling cuda $<"
	$(Q)nvcc $(NVCC_FLAGS) -MMD -MP -c $< -o $@

# Include the dependency files
-include $(OBJECTS:.o=.d)

clean:
	@echo " ==== Cleaning all subdirectories ==== "
	$(Q)find . -name "*.o" -type f -delete
	$(Q)find . -name "*.d" -type f -delete
	$(Q)find . -name ".*" -type f -delete
	$(Q)rm -f *cxx* *.so *.pcm *Linkdef.h *dict*.h
