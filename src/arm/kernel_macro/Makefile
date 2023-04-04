

all: kernel_macro.o kernel_macro_naive.o kernel_macro_neon_8x8.o kernel_macro_neon_4x16.o
	ar cr kernel_macro.a kernel_macro.o kernel_macro_naive.o kernel_macro_neon_8x8.o kernel_macro_neon_4x16.o

kernel_macro.o:
	gcc -g -c kernel_macro.c

kernel_macro_naive.o:
	gcc -g -c kernel_macro_naive.c

kernel_macro_neon_8x8.o:
	gcc -g -c kernel_macro_neon_8x8.c

kernel_macro_neon_4x16.o:
	gcc -g -c kernel_macro_neon_4x16.c

clean:
	rm *.o *.a