

all: gemm_naive.o gemm_neon_8x8.o gemm_neon_4x16.o
	ar cr gemm.a gemm_naive.o gemm_neon_8x8.o gemm_neon_4x16.o

gemm_naive.o:
	gcc -g -c gemm_naive.c

gemm_neon_8x8.o:
	gcc -g -c gemm_neon_8x8.c

gemm_neon_4x16.o:
	gcc -g -c gemm_neon_4x16.c

clean:
	rm *.o *.a