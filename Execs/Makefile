# Load the configuration file
include ../config.mk
include ./Make.defs

# RMT3D requires PGMG library
# Input PGMG path in Make.defs, otherwise, default is ../../
PGMG_DIR?=../../
# Assemble the subdirectories that contain source files
source_dirs += ../F3D_Source ../Extrapolation ../IO ../Sim ../Utilities

# source files and corresponding objects
src+=$(foreach subdir,$(source_dirs),$(wildcard $(subdir)/*.cc))
objs=$(patsubst %.cc,%.o,$(src))

# header files
headers+=$(foreach subdir,$(source_dirs),$(wildcard $(subdir)/*.hh))

# Add these directory to include path
iflags = $(foreach subdir,$(source_dirs), -I./$(subdir))
# Add a few flags for GNU Scientific Library
iflags+=-I$(PGMG_DIR) `gsl-config --cflags`
# Add these directory to library path
lflags+=-L. -L$(PGMG_DIR) -lpgmg `gsl-config --libs`
# add C++-11 std to clfags

########################################
# Definition of macros in Make.defs
ifeq ($(DEBUG), TRUE)
	cflags += -DDEBUG
	debugsuffix := .debug
else
	debugsuffix :=
endif

# COMPILE A VARIABLE DENSITY VERSION
ifeq ($(VAR_DEN), TRUE)
	cflags += -DVAR_DEN
	vardensuffix := .varden
else
	vardensuffix :=
endif

# COMPILE A VERSION THAT DOES TANGENTIAL STABILITY CALCULATION
ifeq ($(TANG), TRUE)
	cflags += -DTANG
	tangsuffix := .tang
else
	tangsuffix :=
endif


execsuffix :=$(vardensuffix)$(debugsuffix)$(tangsuffix)

ifeq ($(FFTW), TRUE)
	# User-overridable FFTW include/library locations.
	ifneq ($(strip $(FFTW_PREFIX)),)
		iflags += -I$(FFTW_PREFIX)/include
		lflags += -L$(FFTW_PREFIX)/lib
	endif
	ifneq ($(strip $(FFTW_INC)),)
		iflags += $(FFTW_INC)
	endif
	ifneq ($(strip $(FFTW_LIB)),)
		lflags += $(FFTW_LIB)
	endif
	cflags += -DHIT_USE_FFTW
	lflags += -lfftw3
endif

########################################
# required executables
execs=ctr_stats unpack tr_unpack run_sim$(execsuffix)

########################################
# make everything in pgmg and then here
all: check_fftw
	$(MAKE) depend
	$(MAKE) -C $(PGMG_DIR) objects
	$(MAKE) -C ../pov_headers
	$(MAKE) executables

check_fftw:
ifeq ($(FFTW), TRUE)
	@echo "[check_fftw] FFTW=TRUE, checking header and linker..."
	@printf '#include <fftw3.h>\nint main(){fftw_plan p=0; (void)p; return 0;}\n' > /tmp/fftw_check.cpp
	@$(cxx) $(iflags) /tmp/fftw_check.cpp $(lflags) -lfftw3 -o /tmp/fftw_check.bin >/tmp/fftw_check.log 2>&1 || \
		(echo "[check_fftw] ERROR: FFTW check failed."; cat /tmp/fftw_check.log; exit 1)
	@echo "[check_fftw] OK"
endif

executables: $(execs)

-include Makefile.dep

depend: $(src)
	$(cxx) $(cflags) $(iflags) -MM $(src) >Makefile.dep

clean:
	rm -rf $(execs) $(objs) librmt3d.a *.dSYM

realclean:
	rm -rf $(execs) $(objs) librmt3d.a *.dSYM
	$(MAKE) -C $(PGMG_DIR)/ clean


# object rules
%.o: %.cc
	$(mpicxx) $(cflags) $(cpp11) $(iflags) -c $< -o $@

librmt3d.a: $(objs)
	rm -f librmt3d.a
	ar rs librmt3d.a $^

run_sim$(execsuffix): run_sim.cc librmt3d.a $(headers)
	$(mpicxx) $(cflags) $(iflags) -o $@ $^ $(lflags) -lrmt3d $(lp_lflags)

ctr_stats: ctr_stats.cc librmt3d.a
	$(cxx) $(cflags) $(iflags) -o $@ $^ $(lflags) -lrmt3d $(lp_lflags) -g -O0

unpack: unpack.cc librmt3d.a
	$(cxx) $(cflags) $(iflags) -o $@ $^ $(lflags) -lrmt3d $(lp_lflags)

tr_unpack: tr_unpack.cc
	$(cxx) $(cflags) $(iflags) -o $@ $^ $(lflags) $(lp_lflags)

.PHONY: clean all executables depend realclean
