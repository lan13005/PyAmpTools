SHELL = /bin/bash # Use bash syntax

ifndef AMPTOOLS_HOME
$(error Please set AMPTOOLS_HOME to point to the root of the AmpTools source tree. )
endif

include $(AMPTOOLS_HOME)/Makefile.settings
######################################################################################
Q := @

####### SET ENVIORNMENT VARIALBES ########
REPO_HOME := $(shell echo $$REPO_HOME)
ifndef REPO_HOME
$(error Please set REPO_HOME by running set_environment.py in the base directory)
endif
ifndef ROOTSYS
$(error ROOTSYS is not set. Please load your root environment first)
endif
AMPTOOLS_DATAIO := $(REPO_HOME)/external/AMPTOOLS_AMPS_DATAIO
ifndef AMPTOOLS_HOME
  AMPTOOLS_HOME := $($REPO_HOME)/external/AmpTools
endif
AMPTOOLS := $(AMPTOOLS_HOME)/AmpTools
AMPPLOTTER := $(AMPTOOLS_HOME)/AmpPlotter
# FSROOT := $(FSROOT)

######## SET COMPILER AND LIBRARY LOCATIONS ########
INC_DIR  := -I$(AMPTOOLS_DATAIO) -I$(shell root-config --incdir) -I$(AMPTOOLS) -I$(AMPPLOTTER)
# INC_DIR  := -I$(AMPTOOLS_DATAIO) -I$(shell root-config --incdir) -I$(AMPTOOLS) -I$(AMPPLOTTER) -I$(FSROOT)
CXX_FLAGS += -Wall -shared -fPIC -std=c++17
ROOTLIBS  := $(shell root-config --libs)
AMPTOOLS_HEADERS := $(shell find $(AMPTOOLS) -type f -name '*.h' -not -name '*GPU*' -not -name '*MPI*' -not -name '*LinkDef*' -not -name "MainPage.h")
# FSROOT_HEADERS := $(shell find $(FSROOT) -type f -name '*.h')

ifdef GPU
AMPTOOLS_HEADERS += $(shell find $(AMPTOOLS) -type f -name '*GPU*.h')
INC_DIR += -I$(CUDA_INSTALL_PATH)/include
CXX_FLAGS += -DGPU_ACCELERATION -I$(CUDA_INSTALL_PATH)/include
SUFFIX := _GPU
NVCC_FLAGS += --compiler-options '-fPIC' --shared
CUDA_SOURCES := $(wildcard *.cu)
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
# FSROOT_HEADERS := $(patsubst $(FSROOT)/%,%,$(FSROOT_HEADERS))

AMPTOOLS_LIBS := -L$(AMPTOOLS)/lib -lAmpTools$(SUFFIX)
AMPPLOTTER_LIBS := -L$(AMPPLOTTER)/lib -lAmpPlotter
# FSROOT_LIBS := -L$(FSROOT)/lib -lFSRoot

######## SET SOURCES AND HEADER FILES ########
# Source Files, deriving object and header files from them
SOURCES := $(wildcard *.cc)
OBJECTS := $(SOURCES:.cc=$(SUFFIX).o)
HEADERS := $(SOURCES:.cc=.h)

######## SET TARGET LIBRARY NAME AND DICTIONARY SOURCE FILE ########
TARGET_LIBRARY = libAmpsDataIO$(SUFFIX).so # Output library name
TARGET_DICT = AmpsDataIODict$(SUFFIX).cxx  # Dictionary for PyROOT to locate library objects
TARGET_LINKDEF = AmpsDataIODict$(SUFFIX)_Linkdef.h  # This file is need to create the dictionary

######## MAKE RULES ########
.PHONY: all clean

# Targets
all: $(TARGET_LIBRARY)

$(TARGET_LIBRARY): $(OBJECTS) $(CUDA_OBJECTS) $(TARGET_DICT)
	@echo " ==== Creating final library $@ ==== "
	$(Q)$(CXX) $(CXX_FLAGS) -o $@ $^ $(ROOTLIBS) $(AMPTOOLS_LIBS) $(AMPPLOTTER_LIBS)
# $(Q)$(CXX) $(CXX_FLAGS) -o $@ $^ $(ROOTLIBS) $(AMPTOOLS_LIBS) $(AMPPLOTTER_LIBS) $(FSROOT_LIBS)

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
	@for header in $(FSROOT_HEADERS); do \
		echo "#pragma link C++ defined_in \"$$header\";" >> ${TARGET_LINKDEF}; \
	done
	@echo "#endif" >> ${TARGET_LINKDEF}

	$(Q)rootcling -f $(INC_DIR) $@ $^ $(AMPTOOLS_HEADERS) $(AMPPLOTTER_HEADERS) ${TARGET_LINKDEF}
# $(Q)rootcling -f $(INC_DIR) $@ $^ $(AMPTOOLS_HEADERS) $(AMPPLOTTER_HEADERS) $(FSROOT_HEADERS) ${TARGET_LINKDEF}

%$(SUFFIX).o: %.cc .%$(SUFFIX).d
	@echo " ==== Compiling $< ==== "
	$(Q)$(CXX) $(CXX_FLAGS) -M -MP -MT $@ -o .$*$(SUFFIX).d $<
	$(Q)$(CXX) $(CXX_FLAGS) -o $@ $<

%$(SUFFIX).o: %.cu .%$(SUFFIX).d
	$(vecho) "-> Compiling cuda $<"
	$(Q)$(NVCC) $(NVCC_FLAGS) -M -MT $@ -o .$*$(SUFFIX).d $<
	$(Q)$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

DEPFILES := $(SOURCES:%.cc=.%$(SUFFIX).d) $(CUDA_SOURCES:%.cu=.%$(SUFFIX).d)
$(DEPFILES):

clean:
	$(Q)-rm -f *.o *.d *.so *.pcm *.cxx* *dict*.h *_Linkdef.h .nfs*
