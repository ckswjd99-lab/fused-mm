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
	@valgrind --tool=callgrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./bench_custom_fumm.out 2> temp.txt
	@cat temp.txt
	@FUMM_PNUM=`cat ./temp.txt | grep -Po "[0-9.]{6}" | head -1`; callgrind_annotate ./callgrind.out.$$FUMM_PNUM > cgout_$$FUMM_PNUM.txt;
	@echo "BENCH OPENBLAS"
	@time ./bench_openblas.out
	@perf stat ./bench_openblas.out
	@valgrind --tool=callgrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./bench_openblas.out
	@perf record -e cpu-clock ./bench_custom_fumm.out
	@-rm temp.txt

double_sgemm_stat:
	${CC} double_sgemm/bench_custom_singles.c ../custom_blas.a -o bench_custom_singles.out
	${CC} double_sgemm/bench_custom_fumm.c ../custom_blas.a -o bench_custom_fumm.out
	${CC} double_sgemm/bench_openblas.c ../libopenblas_cortexa72-r0.3.21.dev.a -o bench_openblas.out
	@{ time ./bench_custom_singles.out ;} 2>&1 | grep user | grep -Po "[0-9]+s" | grep -Po "[0-9]+"
	@valgrind --tool=callgrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./bench_custom_singles.out 2> single.txt
	@cat single.txt | grep "I   refs" | grep -Po "[0-9,]+" | sed -n '2 p'
	@cat single.txt | grep "LLi misses" | grep -Po "[0-9,]+" | sed -n '2 p'
	@cat single.txt | grep "D   refs" | grep -Po "[0-9,]+" | sed -n '2 p'
	@cat single.txt | grep "D1  misses" | grep -Po "[0-9,]+" | sed -n '3 p'
	@cat single.txt | grep "LLd misses" | grep -Po "[0-9,]+" | sed -n '2 p'
	@{ time ./bench_custom_fumm.out ;} 2>&1 | grep user | grep -Po "[0-9]+s" | grep -Po "[0-9]+"
	@valgrind --tool=callgrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./bench_custom_fumm.out 2> fused.txt
	@cat fused.txt | grep "I   refs" | grep -Po "[0-9,]+" | sed -n '2 p'
	@cat fused.txt | grep "LLi misses" | grep -Po "[0-9,]+" | sed -n '2 p'
	@cat fused.txt | grep "D   refs" | grep -Po "[0-9,]+" | sed -n '2 p'
	@cat fused.txt | grep "D1  misses" | grep -Po "[0-9,]+" | sed -n '3 p'
	@cat fused.txt | grep "LLd misses" | grep -Po "[0-9,]+" | sed -n '2 p'
	@{ time ./bench_openblas.out ;} 2>&1 | grep user | grep -Po "[0-9]+s" | grep -Po "[0-9]+"
	@valgrind --tool=callgrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./bench_openblas.out 2> openblas.txt
	@cat openblas.txt | grep "I   refs" | grep -Po "[0-9,]+" | sed -n '2 p'
	@cat openblas.txt | grep "LLi misses" | grep -Po "[0-9,]+" | sed -n '2 p'
	@cat openblas.txt | grep "D   refs" | grep -Po "[0-9,]+" | sed -n '2 p'
	@cat openblas.txt | grep "D1  misses" | grep -Po "[0-9,]+" | sed -n '3 p'
	@cat openblas.txt | grep "LLd misses" | grep -Po "[0-9,]+" | sed -n '2 p'

clean:
	rm *.out *.out.* perf.* cgout_* temp.txt