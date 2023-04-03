clean:
	rm *.out *.o *.s

tester:
	gcc -g gemm.c kernel_macro.c kernel_micro.c utils.c test.c libopenblas_cortexa72-r0.3.21.dev.a -o tester.out