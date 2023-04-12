SHELL := /bin/bash
CC = gcc -g

.PHONY: correct fumm_correct single_sgemm double_sgemm

correct:
	$(CC) correctness/test.c ../custom_blas.a ../libopenblas_cortexa72-r0.3.21.dev.a -o correct.out
	./correct.out

fumm_correct:
	$(CC) fumm_correctness/test.c ../custom_blas.a ../libopenblas_cortexa72-r0.3.21.dev.a -o fumm_correct.out
	./fumm_correct.out

single_sgemm:
	${CC} single_sgemm/bench_custom.c ../custom_blas.a -o bench_custom.out
	${CC} single_sgemm/bench_openblas.c ../libopenblas_cortexa72-r0.3.21.dev.a -o bench_openblas.out
	@echo "BENCH CUSTOM"
	@time ./bench_custom.out
	@perf stat ./bench_custom.out
	@valgrind --tool=callgrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./bench_custom.out
	@echo "BENCH OPENBLAS"
	@time ./bench_openblas.out
	@perf stat ./bench_openblas.out
	@valgrind --tool=callgrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./bench_openblas.out
	@perf record -e cpu-clock ./bench_custom.out

double_sgemm:
	${CC} double_sgemm/bench_custom_singles.c ../custom_blas.a -o bench_custom_singles.out
	${CC} double_sgemm/bench_custom_fumm.c ../custom_blas.a -o bench_custom_fumm.out
	${CC} double_sgemm/bench_openblas.c ../libopenblas_cortexa72-r0.3.21.dev.a -o bench_openblas.out
	@echo "BENCH CUSTOM SINGLES"
	@time ./bench_custom_singles.out
	@perf stat ./bench_custom_singles.out
	@valgrind --tool=callgrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./bench_custom_singles.out
	@echo "BENCH CUSTOM FUMM"
	@time ./bench_custom_fumm.out
	@perf stat ./bench_custom_fumm.out
	@valgrind --tool=callgrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./bench_custom_fumm.out
	@echo "BENCH OPENBLAS"
	@time ./bench_openblas.out
	@perf stat ./bench_openblas.out
	@valgrind --tool=callgrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./bench_openblas.out
	@perf record -e cpu-clock ./bench_custom_fumm.out

clean:
	rm *.out *.out.* perf.*