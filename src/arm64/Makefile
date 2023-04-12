

all: kernel_micro/kernel_micro.o kernel_macro/kernel_macro.o gemm/gemm.o utils.o
	ar cr ../../custom_blas.a utils.o kernel_micro/kernel_micro.o kernel_macro/kernel_macro.o gemm/gemm.o

kernel_micro/kernel_micro.o:
	make -C kernel_micro

kernel_macro/kernel_macro.o:
	make -C kernel_macro

gemm/gemm.o:
	make -C gemm

utils.o:
	${CC} -c utils.c

clean:
	-rm *.out *.o *.s *.out.*
	-make -C kernel_micro $(@F)
	-make -C kernel_macro $(@F)
	-make -C gemm $(@F)
