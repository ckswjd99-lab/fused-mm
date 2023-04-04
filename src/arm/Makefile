SHELL := /bin/bash

all: kernel_micro/kernel_micro.a kernel_macro/kernel_macro.a gemm/gemm.a test.o utils.o
	gcc -g test.o gemm/gemm.a kernel_macro/kernel_macro.a kernel_micro/kernel_micro.a utils.o libopenblas_cortexa72-r0.3.21.dev.a -o tester.out

kernel_micro/kernel_micro.a:
	make -C kernel_micro

kernel_macro/kernel_macro.a:
	make -C kernel_macro

gemm/gemm.a:
	make -C gemm

clean:
	-rm *.out *.o *.s *.out.*
	-make -C kernel_micro $(@F)
	-make -C kernel_macro $(@F)
	-make -C gemm $(@F)

test.o:
	gcc -g -c test.c

utils.o:
	gcc -g -c utils.c

bench: kernel_micro/kernel_micro.a kernel_macro/kernel_macro.a gemm/gemm.a utils.o
	# gcc kernel_micro.c -S
	gcc -g bench_custom.c gemm/gemm.a kernel_macro/kernel_macro.a kernel_micro/kernel_micro.a utils.o libopenblas_cortexa72-r0.3.21.dev.a -o bench_custom.out
	gcc -g bench_openblas.c gemm/gemm.a kernel_macro/kernel_macro.a kernel_micro/kernel_micro.a utils.o libopenblas_cortexa72-r0.3.21.dev.a libopenblas_cortexa72-r0.3.21.dev.a -o bench_openblas.out
	@echo "BENCH CUSTOM"
	@time ./bench_custom.out
	@perf stat ./bench_custom.out
	@valgrind --tool=callgrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./bench_custom.out
	@echo "BENCH OPENBLAS"
	@time ./bench_openblas.out
	@perf stat ./bench_openblas.out
	@valgrind --tool=callgrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./bench_openblas.out
	@perf record -e cpu-clock ./bench_custom.out