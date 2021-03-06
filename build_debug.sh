#!/bin/bash

TARGET_DIR=debug
BLOCKS=4
MATRIX_WIDTH=16

if [ ! -f "$TARGET_DIR"/Makefile ] ; then
	#create build folder
	mkdir -p $TARGET_DIR || exit 1
	(
		cd $TARGET_DIR && cmake -DCMAKE_BUILD_TYPE=DEBUG -DBLOCKS=$BLOCKS -DMATRIX_WIDTH=$MATRIX_WIDTH -DMATRIX_HEIGHT=$MATRIX_WIDTH .. 
	) || exit 1
fi

cd $TARGET_DIR && make -j2
