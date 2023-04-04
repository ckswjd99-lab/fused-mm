SHELL := /bin/bash
CC = gcc
ARCH = arm64
LIBNAME = custom_blas

all:
	make -C src

clean:
	-rm custom_blas.a
	-make -C src ${@F}