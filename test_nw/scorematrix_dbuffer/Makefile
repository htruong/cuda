include ../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include -I ./

SRC = apitest.cu

EXE = apitest_$(EXTRASUFFIX)

DEPENDS = needle_gpu
DEPENDS2 = needle_cpu

CFLAGS = $(EXTRAFLAGS) -O3 -Xptxas=-v -I$(INCLUDE) -L$(CUDA_LIB_DIR) 
DEBUGFLAGS = -g $(CFLAGS) -DVERBOSE 
# -DDUAL_BUFFERING -DUSE_PINNED_MEM

DEBUG_PREFIX = debug_

%.o: %.cu
	$(CC) $(CFLAGS) -c $<

release: $(SRC) $(DEPENDS).o $(DEPENDS2).o
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) $(DEPENDS).o $(DEPENDS2).o

debug: $(SRC) $(DEPENDS).o $(DEPENDS2).o
	$(CC) $(DEBUGFLAGS) $(SRC) -o $(DEBUG_PREFIX)$(EXE) $(DEPENDS).o $(DEPENDS2).o

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXEDB) $(DEBUG_PREFIX)$(EXE) $(DEBUG_PREFIX)$(EXEDB) *.o result.txt
