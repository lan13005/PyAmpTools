SHELL = /bin/bash # Use bash syntax

####### SET ENVIORNMENT VARIALBES ########
REPO_HOME := $(shell echo $$REPO_HOME)
ifndef REPO_HOME
$(error Please set REPO_HOME by running set_environment.py in the base directory)
endif
ifndef ROOTSYS
$(error ROOTSYS is not set. Please load your root environment first)
endif
AMPTOOLS_AMPS_DATAIO := $(REPO_HOME)/src/GlueX
ifndef AMPTOOLS_HOME
  AMPTOOLS_HOME := $($REPO_HOME)/external/AmpTools
endif
AMPTOOLS := $(AMPTOOLS_HOME)/AmpTools
$(info AMPTOOLS_AMPS_DATAIO: $(AMPTOOLS_AMPS_DATAIO))
$(info AMPTOOLS_HOME: $(AMPTOOLS_HOME))
$(info AMPTOOLS: $(AMPTOOLS))

######## SET COMPILER AND LIBRARY LOCATIONS ########
include $(AMPTOOLS_HOME)/Makefile.settings
CXX = gcc
INC_DIR  = -I$(AMPTOOLS_AMPS_DATAIO) -I$(shell root-config --incdir) -I$(AMPTOOLS) -I$(AMPPLOTTER)
CXXFLAGS = -Wall -shared -fPIC $(INC_DIR)
ROOTLIBS   = $(shell root-config --libs)
AMPTOOLS_LIBS := -L$(AMPTOOLS)/lib -lAmpTools
AMPPLOTTER_LIBS := -L$(AMPPLOTTER)/lib -lAmpPlotter
AMPTOOLS_HEADERS := $(shell find $(AMPTOOLS) -type f -name '*.h' -not -name '*GPU*' -not -name '*MPI*' -not -name '*LinkDef*')

ifdef MPI
CXX := $(MPICXX)
CXXFLAGS += -DUSE_MPI
AMPTOOLS_HEADERS := $(shell find $(AMPTOOLS) -type f -name '*.h' -not -name '*GPU*' -not -name '*LinkDef*')
MPI_SHOWME := $(shell $(CXX) -showme)
MPI_INCLUDE := $(shell echo $(MPI_SHOWME) | cut -d ' ' -f 2)
INC_DIR += $(MPI_INCLUDE)
AMPTOOLS_HEADERS += mpi.h
endif

AMPTOOLS_HEADERS := $(patsubst $(AMPTOOLS)/%,%,$(AMPTOOLS_HEADERS))

AMPPLOTTER_HEADERS := $(shell find $(AMPPLOTTER) -type f -name '*.h')
AMPPLOTTER_HEADERS := $(patsubst $(AMPPLOTTER)/%,%,$(AMPPLOTTER_HEADERS))

######## SET SOURCES AND HEADER FILES ########
# Source Files, deriving object and header files from them
SOURCES := $(wildcard *.cc)
OBJECTS := $(SOURCES:.cc=.o)
HEADERS := $(SOURCES:.cc=.h)
$(info SOURCES: $(SOURCES))
$(info OBJECTS: $(OBJECTS))
$(info HEADERS: $(HEADERS))

######## SET TARGET LIBRARY NAME AND DICTIONARY SOURCE FILE ########
TARGET_LIBRARY = libDataIO.so # Output library name
TARGET_DICT = DataIODict.cxx  # Dictionary for PyROOT to locate library objects
TARGET_LINKDEF = Linkdef.h    # This file is need to create the dictionary

######## MAKE RULES ########
.PHONY: all clean

# Targets
all: $(TARGET_LIBRARY)

$(TARGET_LIBRARY): $(OBJECTS) $(TARGET_DICT)
	@echo " === Creating final library $@ === "
	$(CXX) $(CXXFLAGS) -o $@ $^ $(ROOTLIBS) $(AMPTOOLS_LIBS) $(AMPPLOTTER_LIBS)

$(TARGET_DICT): $(HEADERS)
	@echo " === Creating Dictionary $@ === "

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
	@echo "#pragma link C++ defined_in \"mpi.h\";" >> ${TARGET_LINKDEF}
	@echo "#endif" >> ${TARGET_LINKDEF}

	rootcling -f $(INC_DIR) $@ $^ $(AMPTOOLS_HEADERS) $(AMPPLOTTER_HEADERS) ${TARGET_LINKDEF}

%.o: %.cc
	@echo " ==== Compiling $< === "
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -f *.o *.cxx* *.so *.pcm *dict*.h
