	.arch armv8-a
	.file	"kernel_micro_neon_4x16.c"
	.text
	.align	2
	.global	sgemm_micro_kernel_neon_4x16
	.type	sgemm_micro_kernel_neon_4x16, %function
sgemm_micro_kernel_neon_4x16:
// FUNCTION SHAPE
// 	void sgemm_micro_kernel_neon_4x16
// 	(int kc, float alpha, const float *A, const float *B, float beta, float *C, int incRowC, int incColC)
//       w0           s0              x1              x2          s1        x3           w4           w5

// REGISTER USAGE
// 
// GEN REG
//
// 00 kc
// 01 A
// 02 B
// 03 C
// 04 incRowC
// 05 incColC
// 06 iter
// 07 tempA
// 08 tempB
// 09 tempC
// 10 kc
// 11 
// 12 
// 13 
// 14 
// 15 
// 16 
// 17 
// 18 
// 19 
// 20 
// 21 
// 22 
// 23 
// 24 
// 25 
// 26 
// 27 
// 28 
// 29 
// 30 
// 31 
//
// VEC REG
//
//

// kc .req w0
// pA .req x1
// poB .req x2
// poC .req x3
// incRowC .req w4
// incColC .req w5
iter 	.req w6
tempA 	.req x7
tempB 	.req x8
tempC 	.req x9
kc_w	.req w10
kc_x	.req x10


.LFB0:
	.cfi_startproc
// prepare space for arguments
	sub	sp, sp, #64
	.cfi_def_cfa_offset 64
	str	w0, [sp, 44]
	str	s0, [sp, 40]
	// str	x1, [sp, 32]
	// str	x2, [sp, 24]
	str	s1, [sp, 20]
	str	x3, [sp, 8]
	str	w4, [sp, 16]
	str	w5, [sp, 4]
// load C values
	mov tempC, x3				// tempC = C
	lsl	w5, w5, 2
	ld1 {v16.4S-v19.4S}, [tempC], x5
	ld1 {v20.4S-v23.4S}, [tempC], x5
	ld1 {v24.4S-v27.4S}, [tempC], x5
	ld1 {v28.4S-v31.4S}, [tempC], x5

// for ( iter = 0; iter < w0(=kc); iter++ )
	mov iter, wzr				// init: iter = 0
	mov tempA, x1				// tempA = A
	mov tempB, x2				// tempB = B
	mov kc_x, x0					// kc on reg x0
	ldr	w2, [sp, 40]			// load alpha
	dup v10.4S, w2				// v10.4S = alpha
	b	.LoopConditionCheck

.LoopBody:	
	ld1 {v8.4S-v9.4S}, [tempA], #32
	ld1 {v0.4S-v3.4S}, [tempB], #64
	ld1 {v4.4S-v7.4S}, [tempB], #64
	fmul v8.4S, v8.4S, v10.4S
	fmul v9.4S, v9.4S, v10.4S
	fmla v16.4S, v0.4S, v8.4S[0]
	fmla v17.4S, v1.4S, v8.4S[0]
	fmla v18.4S, v2.4S, v8.4S[0]
	fmla v19.4S, v3.4S, v8.4S[0]
	fmla v20.4S, v0.4S, v8.4S[1]
	fmla v21.4S, v1.4S, v8.4S[1]
	fmla v22.4S, v2.4S, v8.4S[1]
	fmla v23.4S, v3.4S, v8.4S[1]
	fmla v24.4S, v0.4S, v8.4S[2]
	fmla v25.4S, v1.4S, v8.4S[2]
	fmla v26.4S, v2.4S, v8.4S[2]
	fmla v27.4S, v3.4S, v8.4S[2]
	fmla v28.4S, v0.4S, v8.4S[3]
	fmla v29.4S, v1.4S, v8.4S[3]
	fmla v30.4S, v2.4S, v8.4S[3]
	fmla v31.4S, v3.4S, v8.4S[3]
	fmla v16.4S, v4.4S, v9.4S[0]
	fmla v17.4S, v5.4S, v9.4S[0]
	fmla v18.4S, v6.4S, v9.4S[0]
	fmla v19.4S, v7.4S, v9.4S[0]
	fmla v20.4S, v4.4S, v9.4S[1]
	fmla v21.4S, v5.4S, v9.4S[1]
	fmla v22.4S, v6.4S, v9.4S[1]
	fmla v23.4S, v7.4S, v9.4S[1]
	fmla v24.4S, v4.4S, v9.4S[2]
	fmla v25.4S, v5.4S, v9.4S[2]
	fmla v26.4S, v6.4S, v9.4S[2]
	fmla v27.4S, v7.4S, v9.4S[2]
	fmla v28.4S, v4.4S, v9.4S[3]
	fmla v29.4S, v5.4S, v9.4S[3]
	fmla v30.4S, v6.4S, v9.4S[3]
	fmla v31.4S, v7.4S, v9.4S[3]
	
// 0 "" 2
#NO_APP
	
	ldr	w2, [sp, 40]
#APP
// 94 "kernel_micro_neon_4x16.c" 1
	ld1 {v8.4S-v9.4S}, [tempA], #32
	ld1 {v0.4S-v3.4S}, [tempB], #64
	ld1 {v4.4S-v7.4S}, [tempB], #64
	dup v10.4S, w2
	fmul v8.4S, v8.4S, v10.4S
	fmul v9.4S, v9.4S, v10.4S
	fmla v16.4S, v0.4S, v8.4S[0]
	fmla v17.4S, v1.4S, v8.4S[0]
	fmla v18.4S, v2.4S, v8.4S[0]
	fmla v19.4S, v3.4S, v8.4S[0]
	fmla v20.4S, v0.4S, v8.4S[1]
	fmla v21.4S, v1.4S, v8.4S[1]
	fmla v22.4S, v2.4S, v8.4S[1]
	fmla v23.4S, v3.4S, v8.4S[1]
	fmla v24.4S, v0.4S, v8.4S[2]
	fmla v25.4S, v1.4S, v8.4S[2]
	fmla v26.4S, v2.4S, v8.4S[2]
	fmla v27.4S, v3.4S, v8.4S[2]
	fmla v28.4S, v0.4S, v8.4S[3]
	fmla v29.4S, v1.4S, v8.4S[3]
	fmla v30.4S, v2.4S, v8.4S[3]
	fmla v31.4S, v3.4S, v8.4S[3]
	fmla v16.4S, v4.4S, v9.4S[0]
	fmla v17.4S, v5.4S, v9.4S[0]
	fmla v18.4S, v6.4S, v9.4S[0]
	fmla v19.4S, v7.4S, v9.4S[0]
	fmla v20.4S, v4.4S, v9.4S[1]
	fmla v21.4S, v5.4S, v9.4S[1]
	fmla v22.4S, v6.4S, v9.4S[1]
	fmla v23.4S, v7.4S, v9.4S[1]
	fmla v24.4S, v4.4S, v9.4S[2]
	fmla v25.4S, v5.4S, v9.4S[2]
	fmla v26.4S, v6.4S, v9.4S[2]
	fmla v27.4S, v7.4S, v9.4S[2]
	fmla v28.4S, v4.4S, v9.4S[3]
	fmla v29.4S, v5.4S, v9.4S[3]
	fmla v30.4S, v6.4S, v9.4S[3]
	fmla v31.4S, v7.4S, v9.4S[3]
	
// 0 "" 2
#NO_APP
		
	ldr	w2, [sp, 40]
#APP
// 97 "kernel_micro_neon_4x16.c" 1
	ld1 {v8.4S-v9.4S}, [tempA], #32
	ld1 {v0.4S-v3.4S}, [tempB], #64
	ld1 {v4.4S-v7.4S}, [tempB], #64
	dup v10.4S, w2
	fmul v8.4S, v8.4S, v10.4S
	fmul v9.4S, v9.4S, v10.4S
	fmla v16.4S, v0.4S, v8.4S[0]
	fmla v17.4S, v1.4S, v8.4S[0]
	fmla v18.4S, v2.4S, v8.4S[0]
	fmla v19.4S, v3.4S, v8.4S[0]
	fmla v20.4S, v0.4S, v8.4S[1]
	fmla v21.4S, v1.4S, v8.4S[1]
	fmla v22.4S, v2.4S, v8.4S[1]
	fmla v23.4S, v3.4S, v8.4S[1]
	fmla v24.4S, v0.4S, v8.4S[2]
	fmla v25.4S, v1.4S, v8.4S[2]
	fmla v26.4S, v2.4S, v8.4S[2]
	fmla v27.4S, v3.4S, v8.4S[2]
	fmla v28.4S, v0.4S, v8.4S[3]
	fmla v29.4S, v1.4S, v8.4S[3]
	fmla v30.4S, v2.4S, v8.4S[3]
	fmla v31.4S, v3.4S, v8.4S[3]
	fmla v16.4S, v4.4S, v9.4S[0]
	fmla v17.4S, v5.4S, v9.4S[0]
	fmla v18.4S, v6.4S, v9.4S[0]
	fmla v19.4S, v7.4S, v9.4S[0]
	fmla v20.4S, v4.4S, v9.4S[1]
	fmla v21.4S, v5.4S, v9.4S[1]
	fmla v22.4S, v6.4S, v9.4S[1]
	fmla v23.4S, v7.4S, v9.4S[1]
	fmla v24.4S, v4.4S, v9.4S[2]
	fmla v25.4S, v5.4S, v9.4S[2]
	fmla v26.4S, v6.4S, v9.4S[2]
	fmla v27.4S, v7.4S, v9.4S[2]
	fmla v28.4S, v4.4S, v9.4S[3]
	fmla v29.4S, v5.4S, v9.4S[3]
	fmla v30.4S, v6.4S, v9.4S[3]
	fmla v31.4S, v7.4S, v9.4S[3]
	
// 0 "" 2
#NO_APP
	
	ldr	w2, [sp, 40]
#APP
// 100 "kernel_micro_neon_4x16.c" 1
	ld1 {v8.4S-v9.4S}, [tempA], #32
	ld1 {v0.4S-v3.4S}, [tempB], #64
	ld1 {v4.4S-v7.4S}, [tempB], #64
	dup v10.4S, w2
	fmul v8.4S, v8.4S, v10.4S
	fmul v9.4S, v9.4S, v10.4S
	fmla v16.4S, v0.4S, v8.4S[0]
	fmla v17.4S, v1.4S, v8.4S[0]
	fmla v18.4S, v2.4S, v8.4S[0]
	fmla v19.4S, v3.4S, v8.4S[0]
	fmla v20.4S, v0.4S, v8.4S[1]
	fmla v21.4S, v1.4S, v8.4S[1]
	fmla v22.4S, v2.4S, v8.4S[1]
	fmla v23.4S, v3.4S, v8.4S[1]
	fmla v24.4S, v0.4S, v8.4S[2]
	fmla v25.4S, v1.4S, v8.4S[2]
	fmla v26.4S, v2.4S, v8.4S[2]
	fmla v27.4S, v3.4S, v8.4S[2]
	fmla v28.4S, v0.4S, v8.4S[3]
	fmla v29.4S, v1.4S, v8.4S[3]
	fmla v30.4S, v2.4S, v8.4S[3]
	fmla v31.4S, v3.4S, v8.4S[3]
	fmla v16.4S, v4.4S, v9.4S[0]
	fmla v17.4S, v5.4S, v9.4S[0]
	fmla v18.4S, v6.4S, v9.4S[0]
	fmla v19.4S, v7.4S, v9.4S[0]
	fmla v20.4S, v4.4S, v9.4S[1]
	fmla v21.4S, v5.4S, v9.4S[1]
	fmla v22.4S, v6.4S, v9.4S[1]
	fmla v23.4S, v7.4S, v9.4S[1]
	fmla v24.4S, v4.4S, v9.4S[2]
	fmla v25.4S, v5.4S, v9.4S[2]
	fmla v26.4S, v6.4S, v9.4S[2]
	fmla v27.4S, v7.4S, v9.4S[2]
	fmla v28.4S, v4.4S, v9.4S[3]
	fmla v29.4S, v5.4S, v9.4S[3]
	fmla v30.4S, v6.4S, v9.4S[3]
	fmla v31.4S, v7.4S, v9.4S[3]
	
// 0 "" 2
#NO_APP
	add	iter, iter,	 1	// iter++ at for loop
.LoopConditionCheck:
	add	w1, kc_w, 7
	cmp	kc_w, 0
	csel	w0, w1, kc_w, lt
	asr	w0, w0, 3
	mov	w1, w0
	cmp	iter, w1		// for loop condition check
	blt	.LoopBody
// Loop End
// 	Save C
	mov tempC, x3
	st1 {v16.4S-v19.4S}, [tempC], x5
	st1 {v20.4S-v23.4S}, [tempC], x5
	st1 {v24.4S-v27.4S}, [tempC], x5
	st1 {v28.4S-v31.4S}, [tempC], x5

// END FUNCTION	
	add	sp, sp, 64
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE0:
	.size	sgemm_micro_kernel_neon_4x16, .-sgemm_micro_kernel_neon_4x16
	.ident	"GCC: (Debian 10.2.1-6) 10.2.1 20210110"
	.section	.note.GNU-stack,"",@progbits
