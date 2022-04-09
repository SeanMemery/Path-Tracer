.DEFAULT_GOAL=skepu

# Default backend set for SkePU precompiler.
BACKENDS = -openmp -cuda #-opencl

PRECOMPILER_OPTIONS = --no-preserve-lines 

# Backend-specific flags
BACKEND_FLAGS = #-DSKEPU_ENABLE_EXCEPTIONS -DSKEPU_DEBUG=3 

# Default OpenCL specific flags (matching a CUDA-provided installation)
OPENCL_FLAGS = -I /usr/local/cuda/include -L /usr/local/cuda/lib64/ -lOpenCL

# path to Clang source directory (repository root)
CLANG_INCLUDE = $(shell pwd)/../skepu/llvm/clang/lib/Headers

# path to the SkePU sources
SKEPU_INCLUDE = $(shell pwd)/../skepu/skepu-headers/src

# path to PT include
PT_INCLUDE = $(shell pwd)/deps

# path to PT source
PT_SOURCE = $(shell pwd)
PT_HEADERS = $(shell pwd)/headers

# ---------------------------------------------------------------------------- #
# Compilers.

# Location of SkePU precompiler binary.
SKEPU = $(shell pwd)/../skepu/build/llvm/bin/skepu-tool

# ---------------------------------------------------------------------------- #
# Compiler flags begin here.

# Flags for precompiler.
SKEPU_FLAGS = $(PRECOMPILER_OPTIONS)
SKEPU_FLAGS += -- -std=c++11 -Wno-expansion-to-defined
SKEPU_FLAGS += -I $(CLANG_INCLUDE)
SKEPU_FLAGS += -I $(SKEPU_INCLUDE)
SKEPU_FLAGS += -I $(PT_INCLUDE) 
SKEPU_FLAGS += -I $(PT_SOURCE)
SKEPU_FLAGS += -I $(PT_HEADERS)

# Activate backend flags for CUDA backend
ifneq (,$(findstring cuda, $(BACKENDS)))
BACKEND_FLAGS += -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used" -Xcompiler
FILETYPE = cu
else
BACKEND_FLAGS += -Wno-attributes
FILETYPE = cpp
endif

# Activate backend flags for OpenMP backend
ifneq (,$(findstring openmp, $(BACKENDS)))
BACKEND_FLAGS += -fopenmp
endif

# Activate backend flags for OpenCL backend
ifneq (,$(findstring opencl, $(BACKENDS)))
BACKEND_FLAGS += $(OPENCL_FLAGS)
endif

LIB_DIRS = 
LIB_DIRS += -L$(PT_INCLUDE)/lib/glew/x64
LIB_DIRS += -L$(PT_INCLUDE)/lib/SDL/x64
LIB_DIRS += -L$(PT_INCLUDE)/lib/assimp/x64
LIB_DIRS += -L$(PT_INCLUDE)/lib/cuda/x64

LIBS = 
LIBS += -lSDL2
LIBS += -lSDL2main
LIBS += -lGLEW
LIBS += -lGLU
LIBS += -lGL
LIBS += -lassimp
LIBS += -lgomp
LIBS += -lcudart
LIBS += -lcuda

# Flags for target compiler (preprocessed sources).
TARGET_FLAGS = -w -std=c++11 -I $(SKEPU_INCLUDE) -I $(PT_HEADERS) -I $(PT_INCLUDE) -I $(PT_SOURCE) -I ./ $(LIB_DIRS) $(LIBS) 

BUILD = build
OBJ_DIR = obj
#-----------Cpps-------------
PT = Main
PT += PT

PT += deps/ext/imgui
PT += deps/ext/imgui_demo
PT += deps/ext/imgui_draw
PT += deps/ext/imgui_impl_opengl3
PT += deps/ext/imgui_impl_sdl
PT += deps/ext/imgui_tables
PT += deps/ext/imgui_widgets

PT_CPP = $(addprefix $(PT_SOURCE)/, $(PT))
PT_CPP_FILES = $(addprefix $(PT_SOURCE)/,$(addsuffix .cpp, $(PT)))

FNAMES = -fnames
FNAMES += " __builtin_inff"

CUDA_FILES = SkePURenderers.cu SkePUDenoiser.cu SkePUDenoiserNN.cu CUDARender.cu CUDADenoiser.cu CUDADenoiserNN.cu


#----------------------------	

$(BUILD):
	@mkdir -p 

skepu:
	$(SKEPU) $(FNAMES) $(BACKENDS) -name SkePURenderers $(PT_SOURCE)/Renderers.cpp -dir $(PT_SOURCE) $(SKEPU_FLAGS)
	$(SKEPU) $(FNAMES) $(BACKENDS) -name SkePUDenoiser $(PT_SOURCE)/Denoiser.cpp -dir $(PT_SOURCE) $(SKEPU_FLAGS)
	$(SKEPU) $(FNAMES) $(BACKENDS) -name SkePUDenoiserNN $(PT_SOURCE)/DenoiserNN.cpp -dir $(PT_SOURCE) $(SKEPU_FLAGS)
	nvcc $(CUDA_FILES) $(PT_CPP_FILES) -w $(TARGET_FLAGS) $(BACKEND_FLAGS) -O3 -o $(BUILD)/PT

debug:
	$(SKEPU) $(FNAMES) $(BACKENDS) -name SkePURenderers $(PT_SOURCE)/Renderers.cpp -dir $(PT_SOURCE) $(SKEPU_FLAGS)
	$(SKEPU) $(FNAMES) $(BACKENDS) -name SkePUDenoiser $(PT_SOURCE)/Denoiser.cpp -dir $(PT_SOURCE) $(SKEPU_FLAGS)
	$(SKEPU) $(FNAMES) $(BACKENDS) -name SkePUDenoiserNN $(PT_SOURCE)/DenoiserNN.cpp -dir $(PT_SOURCE) $(SKEPU_FLAGS)
	nvcc $(CUDA_FILES) $(PT_CPP_FILES) -w $(TARGET_FLAGS) $(BACKEND_FLAGS) -g -O0 -o $(BUILD)/PT_debug



