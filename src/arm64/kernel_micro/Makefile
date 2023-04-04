CC = gcc

all: kernel_micro_naive.o kernel_micro_neon_8x8.o kernel_micro_neon_4x16.o
	ld -r -o kernel_micro.o kernel_micro_naive.o kernel_micro_neon_8x8.o kernel_micro_neon_4x16.o

kernel_micro_naive.o:
	$(CC) -c kernel_micro_naive.c 

kernel_micro_neon_8x8.o:
	$(CC) -c kernel_micro_neon_8x8.c

kernel_micro_neon_4x16.o:
	$(CC) -c kernel_micro_neon_4x16.s

clean:
	rm *.o *.a