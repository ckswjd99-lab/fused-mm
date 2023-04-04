

all: kernel_micro_naive.o kernel_micro_neon_8x8.o kernel_micro_neon_4x16.o
	ar cr kernel_micro.a kernel_micro_naive.o kernel_micro_neon_8x8.o kernel_micro_neon_4x16.o

kernel_micro_naive.o:
	gcc -g -c kernel_micro_naive.c

kernel_micro_neon_8x8.o:
	gcc -g -c kernel_micro_neon_8x8.c

kernel_micro_neon_4x16.o:
	gcc -g -c kernel_micro_neon_4x16.s

clean:
	rm *.o *.a