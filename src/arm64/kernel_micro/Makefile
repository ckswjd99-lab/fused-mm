CC = gcc -g

all: kernel_micro_naive.o kernel_micro_neon_8x8.o kernel_micro_neon_4x16.o kernel_micro_neon_4x16_pack_colwise.o kernel_micro_neon_4x16_fused.o
	ld -r -o kernel_micro.o kernel_micro_naive.o kernel_micro_neon_8x8.o kernel_micro_neon_4x16.o kernel_micro_neon_4x16_pack_colwise.o kernel_micro_neon_4x16_fused.o

kernel_micro_naive.o:
	$(CC) -c kernel_micro_naive.c 

kernel_micro_neon_8x8.o:
	$(CC) -c kernel_micro_neon_8x8.c

kernel_micro_neon_4x16.o:
	$(CC) -c kernel_micro_neon_4x16.s

kernel_micro_neon_4x16_pack_colwise.o:
	$(CC) -c kernel_micro_neon_4x16_pack_colwise.s

kernel_micro_neon_4x16_fused.o:
	$(CC) -c kernel_micro_neon_4x16_fused.c

clean:
	rm *.o *.a