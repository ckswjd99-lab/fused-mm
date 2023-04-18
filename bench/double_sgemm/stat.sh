D_MODEL=320

L1_CACHE=49152
D1_CACHE=32768
LL_CACHE=1048576
# LL_CACHE=2097152

TOKEN_START=32
TOKEN_END=64
TOKEN_STEP=32

for i in {16..512..16}
do
    echo -e "#include \"../../params.h\"\n\n#define BENCH_M $i\n#define BENCH_N1 ($D_MODEL*4)\n#define BENCH_N2 $D_MODEL\n#define BENCH_K $D_MODEL" > bench.h
    gcc bench_custom_singles.c ../../custom_blas.a -o bench_custom_singles.out
    gcc bench_custom_fumm.c ../../custom_blas.a -o bench_custom_fumm.out
    gcc bench_openblas.c ../../libopenblas_cortexa72-r0.3.21.dev.a -o bench_openblas.out
    { time ./bench_custom_singles.out ;} 2>&1 | grep user | grep -Po "[0-9]+s" | grep -Po "[0-9]+"
    valgrind --tool=callgrind --I1=$L1_CACHE,3,64 --D1=$D1_CACHE,2,64 --LL=$LL_CACHE,16,64 ./bench_custom_singles.out 2> single.txt
    cat single.txt | grep "I   refs" | grep -Po "[0-9,]+" | sed -n '2 p'
    cat single.txt | grep "LLi misses" | grep -Po "[0-9,]+" | sed -n '2 p'
    cat single.txt | grep "D   refs" | grep -Po "[0-9,]+" | sed -n '2 p'
    cat single.txt | grep "D1  misses" | grep -Po "[0-9,]+" | sed -n '3 p'
    cat single.txt | grep "LLd misses" | grep -Po "[0-9,]+" | sed -n '2 p'
    { time ./bench_custom_fumm.out ;} 2>&1 | grep user | grep -Po "[0-9]+s" | grep -Po "[0-9]+"
    valgrind --tool=callgrind --I1=$L1_CACHE,3,64 --D1=$D1_CACHE,2,64 --LL=$LL_CACHE,16,64 ./bench_custom_fumm.out 2> fused.txt
    cat fused.txt | grep "I   refs" | grep -Po "[0-9,]+" | sed -n '2 p'
    cat fused.txt | grep "LLi misses" | grep -Po "[0-9,]+" | sed -n '2 p'
    cat fused.txt | grep "D   refs" | grep -Po "[0-9,]+" | sed -n '2 p'
    cat fused.txt | grep "D1  misses" | grep -Po "[0-9,]+" | sed -n '3 p'
    cat fused.txt | grep "LLd misses" | grep -Po "[0-9,]+" | sed -n '2 p'
    { time ./bench_openblas.out ;} 2>&1 | grep user | grep -Po "[0-9]+s" | grep -Po "[0-9]+"
    valgrind --tool=callgrind --I1=$L1_CACHE,3,64 --D1=$D1_CACHE,2,64 --LL=$LL_CACHE,16,64 ./bench_openblas.out 2> openblas.txt
    cat openblas.txt | grep "I   refs" | grep -Po "[0-9,]+" | sed -n '2 p'
    cat openblas.txt | grep "LLi misses" | grep -Po "[0-9,]+" | sed -n '2 p'
    cat openblas.txt | grep "D   refs" | grep -Po "[0-9,]+" | sed -n '2 p'
    cat openblas.txt | grep "D1  misses" | grep -Po "[0-9,]+" | sed -n '3 p'
    cat openblas.txt | grep "LLd misses" | grep -Po "[0-9,]+" | sed -n '2 p'
done > stat.txt


