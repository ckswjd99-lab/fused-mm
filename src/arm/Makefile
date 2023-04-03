SHELL := /bin/bash

clean:
	rm *.out *.o *.s *.out.*

tester:
	gcc -g gemm.c kernel_macro.c kernel_micro.c utils.c test.c libopenblas_cortexa72-r0.3.21.dev.a -o tester.out

bench: 
	gcc -g gemm.c kernel_macro.c kernel_micro.c utils.c bench_custom.c -o bench_custom.out
	gcc -g gemm.c kernel_macro.c kernel_micro.c utils.c bench_openblas.c libopenblas_cortexa72-r0.3.21.dev.a -o bench_openblas.out
	@echo "BENCH CUSTOM"
	@time ./bench_custom.out
	@perf stat ./bench_custom.out
	@valgrind --tool=callgrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./bench_custom.out
	@echo "BENCH OPENBLAS"
	@time ./bench_openblas.out
	@perf stat ./bench_openblas.out
	@valgrind --tool=callgrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./bench_openblas.out