CC = gcc

all: kernel_macro_util.o kernel_macro_naive.o kernel_macro_neon_8x8.o kernel_macro_neon_4x16.o
	ld -r -o kernel_macro.o kernel_macro_util.o kernel_macro_naive.o kernel_macro_neon_8x8.o kernel_macro_neon_4x16.o

kernel_macro_util.o:
	$(CC) -c kernel_macro_util.c

kernel_macro_naive.o:
	$(CC) -c kernel_macro_naive.c

kernel_macro_neon_8x8.o:
	$(CC) -c kernel_macro_neon_8x8.c

kernel_macro_neon_4x16.o:
	$(CC) -c kernel_macro_neon_4x16.c

clean:
	rm *.o *.a