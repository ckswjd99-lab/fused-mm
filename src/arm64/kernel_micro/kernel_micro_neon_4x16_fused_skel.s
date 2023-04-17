	.arch armv8-a
	.file	"kernel_micro_neon_4x16_fused.c"
	.text
	.align	2
	.global	sfumm_micro_kernel_neon_4x16
	.type	sfumm_micro_kernel_neon_4x16, %function
sfumm_micro_kernel_neon_4x16:
// void sfumm_micro_kernel_neon_4x16(
//	 int kc, int n2, float alpha, const float *A, const float *B1, const float *B2, float beta, float *C, int incRowC, int incColC
//   w0      w1      s0           x2              x3               x4               s1          x5        w6           w7
// )
//
// GEN REGS
// 00 kc
// 01 n2
// 02 pA
// 03 pB1
// 04 pB2
// 05 pC
// 06 incRowC
// 07 incColC
// 08 i
// 09
// 10 
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
// FLT REGS
// 00 [alpha]				||	[C_set0_col_00]
// 01 [beta]				||	[C_set0_col_01]
// 02 [A_set0_col]			||	[C_set0_col_02]
// 03 [A_set1_col]			||	[C_set0_col_03]
// 04 [B1_set0_row0]		||	[C_set1_col_00]
// 05 [B1_set0_row1]		||	[C_set1_col_01]
// 06 [B1_set0_row2]		||	[C_set1_col_02]
// 07 [B1_set0_row3]		||	[C_set1_col_03]
// 08 [B1_set1_row0]		||	[B2_set0_row_0]
// 09 [B1_set1_row1]		||	[B2_set0_row_1]
// 10 [B1_set1_row2]		||	[B2_set1_row_0]
// 11 [B1_set1_row3]		||	[B2_set1_row_1]
// 12 [temp0]
// 13 [temp1]
// 14 [temp2]
// 15 [temp3]
// 16 [AB1_col_00]
// 17 [AB1_col_01]
// 18 [AB1_col_02]
// 19 [AB1_col_03]
// 20 [AB1_col_04]
// 21 [AB1_col_05]
// 22 [AB1_col_06]
// 23 [AB1_col_07]
// 24 [AB1_col_08]
// 25 [AB1_col_09]
// 26 [AB1_col_10]
// 27 [AB1_col_11]
// 28 [AB1_col_12]
// 29 [AB1_col_13]
// 30 [AB1_col_14]
// 31 [AB1_col_15]

.LFB0:
	.cfi_startproc
//
//	STORE PARAMS
//
	sub	sp, sp, #80
	.cfi_def_cfa_offset 80
	str	w0, [sp, 60]	// kc
	str	w1, [sp, 56]	// n2
	str	s0, [sp, 52]	// alpha
	str	x2, [sp, 40]	// pA
	str	x3, [sp, 32]	// pB1
	str	x4, [sp, 24]	// pB2
	str	s1, [sp, 48]	// beta
	str	x5, [sp, 16]	// pC
	str	w6, [sp, 12]	// incRowC
	str	w7, [sp, 8]		// incColC
//
//	INIT AB1 BUFFER
//	

//
//	CALC AB1 BUFFER
//

// LOOP INIT
	str	wzr, [sp, 76]
	b	.FUMM_MICRO_4x16_AB1LOOP_CONDITION

.FUMM_MICRO_4x16_AB1LOOP_BODY:
// LOOP BODY

// LOOP TAIL
	ldr	w0, [sp, 76]
	add	w0, w0, 1
	str	w0, [sp, 76]

.FUMM_MICRO_4x16_AB1LOOP_CONDITION:
// LOOP CONDITION
	ldr	w1, [sp, 76]
	ldr	w0, [sp, 60]
	cmp	w1, w0
	blt	.FUMM_MICRO_4x16_AB1LOOP_BODY

//
//	ACCUM C
//

// LOOP INIT
	str	wzr, [sp, 76]
	b	.FUMM_MICRO_4x16_ACCCLOOP_CONDITION
.FUMM_MICRO_4x16_ACCCLOOP_BODY:
// LOOP BODY

// LOOP TAIL
	ldr	w0, [sp, 76]
	add	w0, w0, 1
	str	w0, [sp, 76]

.FUMM_MICRO_4x16_ACCCLOOP_CONDITION:
// LOOP CONDITION
	ldr	w1, [sp, 76]
	ldr	w0, [sp, 56]
	cmp	w1, w0
	blt	.FUMM_MICRO_4x16_ACCCLOOP_BODY
	nop
	nop

// FUNC EPILOGUE
	add	sp, sp, 80
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE0:
	.size	sfumm_micro_kernel_neon_4x16, .-sfumm_micro_kernel_neon_4x16
	.ident	"GCC: (Debian 10.2.1-6) 10.2.1 20210110"
	.section	.note.GNU-stack,"",@progbits
