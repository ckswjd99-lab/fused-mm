clean:
	rm *.out *.out.*

tester:
	gcc tester.c kernel_8x8.c block_l2.c gemm.c utils.c -lopenblas -g -o tester_custom.out
	gcc tester_openblas.c kernel_8x8.c block_l2.c utils.c -lopenblas -g -o tester_openblas.out

cachegrind:
	valgrind --tool=cachegrind --I1=49152,3,64 --D1=32768,2,64 --LL=1048576,16,64 ./a.out