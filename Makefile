# Compiler and hypre location
CC        = h5pcc
HYPRE_DIR = /deac/phy/cardenasGrp/cardenas/Codes/hypre/src/hypre
ifndef HYPRE_DIR
$(info HYPRE_DIR not defined, trying the one typed in this file)
endif

#SPACK_CALL_ROOT=/projects/ngc/shared/flecsph/flecsph_flecsi2_centos8_spack_environments/spack/opt/spack/linux-centos8-broadwell/gcc-10.2.0/
#GSL_DIR=$(SPACK_CALL_ROOT)/gsl-2.7-b53w5nysgk5ja7tsraiywnse556kqdvn

# Local directories
INC_DIR = $(CURDIR)/include
SRC_DIR = $(CURDIR)/src
OBJ_DIR = $(CURDIR)/obj

# Compiling and linking options
COPTS     = -g -Wall -O2 -Wformat-overflow
CINCLUDES = -I$(HYPRE_DIR)/include -I$(INC_DIR) -I$(GSL_DIR)/include
CDEFS     = -DHAVE_CONFIG_H -DHYPRE_TIMING
CFLAGS    = $(COPTS) $(CINCLUDES) $(CDEFS)

LINKOPTS = $(COPTS)
LDFLAGS  = -L$(HYPRE_DIR)/lib -L$(GSL_DIR)/lib
LIBS     = -lHYPRE -lgsl -lgslcblas -lm -lstdc++
#LIBS     = -lHYPRE -lm -lgsl -lgslcblas -shlib -lstdc++
LFLAGS   = $(LINKOPTS) $(LIBS)

# List of all programs to be compiled

EXE =  inoisy4d

SRC := $(addprefix $(SRC_DIR)/,main.c hdf5_utils.c model_%.c param_%.c)
OBJ := $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

all: $(EXE)

$(EXE): %: $(OBJ)
	$(CC) $(LDFLAGS) $^ $(LFLAGS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir $@

default: all

# Clean up

clean:
	$(RM) -r $(OBJ_DIR) inoisy4d
distclean: clean
	$(RM) $(EXE)