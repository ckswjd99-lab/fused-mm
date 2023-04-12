CC = gcc -g

all: gemm_naive.o gemm_neon_8x8.o gemm_neon_4x16.o fumm_neon_4x16.o
	ld -r -o gemm.o gemm_naive.o gemm_neon_8x8.o gemm_neon_4x16.o fumm_neon_4x16.o

gemm_naive.o:
	$(CC) -c gemm_naive.c

gemm_neon_8x8.o:
	$(CC) -c gemm_neon_8x8.c

gemm_neon_4x16.o:
	$(CC) -c gemm_neon_4x16.c

fumm_neon_4x16.o:
	$(CC) -c fumm_neon_4x16.c

clean:
	rm *.o *.a