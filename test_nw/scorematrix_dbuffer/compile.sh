#!/bin/bash

EXTRAFLAGS="-DDUAL_BUFFERING -DUSE_PINNED_MEM " EXTRASUFFIX="dualbuffer_pinned" make
make clean
EXTRAFLAGS="-DUSE_PINNED_MEM " EXTRASUFFIX="pinned" make
make clean
EXTRAFLAGS=" " EXTRASUFFIX="regular" make
make clean
