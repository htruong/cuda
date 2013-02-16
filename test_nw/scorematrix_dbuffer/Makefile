include ../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include -I ./

SRC = needle.cu

EXE = needle
EXEDB = needle_dualbuffering

DEBUG_PREFIX = debug_

release: $(SRC)
	$(CC) $(SRC) -O3 -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -Xptxas=-v
	$(CC) $(SRC) -O3 -o $(EXEDB) -DDUAL_BUFFERING -I$(INCLUDE) -L$(CUDA_LIB_DIR) -Xptxas=-v

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) -g $(SRC) -o $(DEBUG_PREFIX)$(EXE) -DVERBOSE -I$(INCLUDE) -L$(CUDA_LIB_DIR)
	$(CC) -g $(SRC) -o $(DEBUG_PREFIX)$(EXEDB) -DVERBOSE -DDUAL_BUFFERING -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
