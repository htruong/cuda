include ../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include -I ./

SRC = apitest.cu

EXE = apitest
EXEDB = apitest_dualbuffering

DEPENDS = needle_gpu
DEPENDS2 = needle_cpu

CFLAGS = -g -O3 -Xptxas=-v -I$(INCLUDE) -L$(CUDA_LIB_DIR) 
DEBUGFLAGS = $(CFLAGS) -DVERBOSE 
# -DDUAL_BUFFERING

DEBUG_PREFIX = debug_

%.o: %.cu
	$(CC) $(DEBUGFLAGS) -c $<

release: $(SRC) $(DEPENDS).o $(DEPENDS2).o
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) $(DEPENDS).o $(DEPENDS2).o
#	$(CC) $(CFLAGS) $(SRC) -o $(EXEDB) $(DEPENDS).o $(DEPENDS2).o

debug: $(SRC) $(DEPENDS).o $(DEPENDS2).o
	$(CC) $(DEBUGFLAGS) $(SRC) -o $(DEBUG_PREFIX)$(EXE) $(DEPENDS).o $(DEPENDS2).o
#	$(CC) $(DEBUGFLAGS) $(SRC) -o $(DEBUG_PREFIX)$(EXEDB) $(DEPENDS).o $(DEPENDS2).o

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXEDB) $(DEBUG_PREFIX)$(EXE) $(DEBUG_PREFIX)$(EXEDB) *.o result.txt
