#ifndef __NEEDLE_H__
#define __NEEDLE_H__
//#define TRACE



#ifdef _LP64 // 64 bit detection.
#else
#endif

#include <pthread.h>

//////////////////
#define TRACE_U 1 // 01
#define TRACE_L 2 // 10
#define TRACE_UL 3 // 11
//////////////////

#define BYTETOBINARYPATTERN "%d%d%d%d%d%d%d%d"
#define BYTETOBINARY(byte)  \
  (byte & 0x80 ? 1 : 0), \
  (byte & 0x40 ? 1 : 0), \
  (byte & 0x20 ? 1 : 0), \
  (byte & 0x10 ? 1 : 0), \
  (byte & 0x08 ? 1 : 0), \
  (byte & 0x04 ? 1 : 0), \
  (byte & 0x02 ? 1 : 0), \
  (byte & 0x01 ? 1 : 0) 

struct needle_work {
	char *sequence_set1;
	char *sequence_set2;
	unsigned int *pos1;
	unsigned int *pos2;
	int *score_matrix;
	unsigned int *pos_matrix;
	unsigned int max_pair_no;
	short penalty;
	unsigned int start;
	unsigned int batch_size;
};

void dataInput( char *sequence_set1, 
				char *sequence_set2, 
				unsigned int *pos1, 
				unsigned int *pos2, 
				unsigned int *pos_matrix,
				int no_pair);



#endif	//__NEEDLE_H__
