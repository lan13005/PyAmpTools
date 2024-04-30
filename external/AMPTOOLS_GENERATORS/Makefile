SHELL = /bin/bash # Use bash syntax

ifndef AMPTOOLS_HOME
$(error Please set AMPTOOLS_HOME to point to the root of the AmpTools source tree. )
endif

include $(AMPTOOLS_HOME)/Makefile.settings
######################################################################################
Q :=

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
AMPGEN := $(PYAMPTOOLS_HOME)/external/AMPTOOLS_GENERATORS
CCDB_HOME := $(PYAMPTOOLS_HOME)/external/AMPTOOLS_GENERATORS/ccdb
CCDB := $(CCDB_HOME)/include
MYSQL := /usr/include/mysql/
PYTHON := $(CONDA_PREFIX)/include/python3.9/

# Consider all subdirectories when searching for source files
SRC_DIRS := $(shell find . -type d -name ccdb -prune -o -print)

######## SET COMPILER AND LIBRARY LOCATIONS ########
INC_DIR  := -I$(AMPTOOLS_DATAIO) -I$(AMPGEN) -I$(AMPTOOLS) -I$(shell root-config --incdir) -I$(CCDB) -I$(MYSQL) -I$(PYTHON)
CXX_FLAGS += -Wall -shared -fPIC -std=c++17
ROOTLIBS  := $(shell root-config --libs)

CXX_FLAGS += $(INC_DIR)
NVCC_FLAGS += $(INC_DIR)

AMPTOOLS_HEADERS := $(shell find $(AMPTOOLS) -type f -name '*.h' -not -name '*GPU*' -not -name '*MPI*' -not -name '*LinkDef*' -not -name "MainPage.h")
AMPTOOLS_HEADERS := $(patsubst $(AMPTOOLS)/%,%,$(AMPTOOLS_HEADERS))
AMPSDATAIO_HEADERS := $(shell find $(AMPTOOLS_DATAIO) -type f -name '*.h' -not -name '*Linkdef*')
AMPSDATAIO_HEADERS := $(patsubst $(AMPTOOLS_DATAIO)/%,%,$(AMPSDATAIO_HEADERS))

AMPTOOLS_LIBS := -L$(AMPTOOLS)/lib -lAmpTools
AMPSDATAIO_LIBS := -L$(AMPTOOLS_DATAIO) -lAmpsDataIO

######## SET SOURCES AND HEADER FILES ########
# Source Files, deriving object and header files from them
SOURCES := $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.cc))
OBJECTS = $(SOURCES:.cc=.o)
HEADERS = $(SOURCES:.cc=.h)
HEADERS := $(filter $(wildcard $(HEADERS)), $(HEADERS))

######## SET TARGET LIBRARY NAME AND DICTIONARY SOURCE FILE ########
TARGET_LIBRARY = libAmpsGen.so # Output library name
TARGET_DICT = AmpsGenDict.cxx  # Dictionary for PyROOT to locate library objects
TARGET_LINKDEF = AmpsGenDict_Linkdef.h  # This file is need to create the dictionary

$(info SRC_DIRS is $(SRC_DIRS))
$(info SOURCES is $(SOURCES))
$(info OBJECTS is $(OBJECTS))
$(info HEADERS is $(HEADERS))

$(info CCDB is $(CCDB))
$(info CCDB_HOME is $(CCDB_HOME))

######## MAKE RULES ########
.PHONY: all clean

# Targets
all: CCDB_LOAD $(TARGET_LIBRARY)

# CCDB scons print statements written for python2. recursively update for python3
CCDB_INSTALL:
	@echo " ==== Installing CCDB ==== "
	@if [ ! -d "$(CCDB_HOME)" ]; then \
		echo "CCDB_HOME directory does not exist, cloning from github"; \
		$(Q)git clone https://github.com/JeffersonLab/ccdb.git --branch v1.06.09; \
		$(Q)cp Distribution/PthreadMutex.h ccdb/include/CCDB; \
		$(Q)find . -type f -name 'SCon*' -exec sed -Ei 's/\bprint\s+"([^"]*)"\s*/print("\1")/g' {} +; \
		$(Q)cd $(CCDB_HOME) && export CCDB_HOME=$(CCDB_HOME) && scons; \
	fi

CCDB_LOAD: CCDB_INSTALL
# Sourcing and evaluation has to be in different rules
#  	otherwise the environment variables are not set
	$(Q)source $(CCDB_HOME)/environment.bash
	$(eval CCDB_HEADERS := $(shell find $(CCDB)/CCDB -type f -name '*.h') )
	$(eval CCDB_HEADERS := $(patsubst $(CCDB)/%,%,$(CCDB_HEADERS)) )
	@echo "CCDB_HEADERS is $(CCDB_HEADERS)"

$(TARGET_LIBRARY): $(CCDB_INSTALL) $(OBJECTS) $(CUDA_OBJECTS) $(TARGET_DICT)
	@echo " ==== Creating final library $@ ==== "
	$(Q)$(CXX) $(CXX_FLAGS) -o $@ $^ $(ROOTLIBS) $(AMPTOOLS_LIBS) $(AMPSDATAIO_LIBS)

$(TARGET_DICT): $(HEADERS)
	@echo " ==== Creating Dictionary $@ ==== "

	@echo "#ifdef __CLING__" > ${TARGET_LINKDEF}
	@for header in $^; do \
		echo "#pragma link C++ defined_in \"$$header\";" >> ${TARGET_LINKDEF}; \
	done
	@for header in $(AMPSDATAIO_HEADERS); do \
		echo "#pragma link C++ defined_in \"$$header\";" >> ${TARGET_LINKDEF}; \
	done
	@for header in $(CCDB_HEADERS); do \
		echo "#pragma link C++ defined_in \"$$header\";" >> ${TARGET_LINKDEF}; \
	done
	@echo "#endif" >> ${TARGET_LINKDEF}

	$(Q)rootcling -f $(INC_DIR) $@ $^ $(AMPSDATAIO_HEADERS) $(CCDB_HEADERS) ${TARGET_LINKDEF}

%.o: %.cc
	echo "-> Compiling $<"
	$(Q)$(CXX) $(CXX_FLAGS) -MMD -MP -c $< -o $@

-include $(OBJECTS:.o=.d)

clean:
	@echo " ==== Cleaning all subdirectories ==== "
	$(Q)find . -name "*.o" -type f -delete
	$(Q)find . -name ".*" -type f -delete
	$(Q)find . -name "*.d" -type f -delete
	$(Q)-rm -f *cxx* *.so *.pcm *Linkdef.h *dict*.h
	$(Q)-rm -rf ccdb
	$(Q)-rm -rf __pycache__
