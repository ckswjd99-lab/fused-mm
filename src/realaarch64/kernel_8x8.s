	.arch armv8-a
	.file	"kernel_8x8.c"
	.text
	.align	2
	.global	sgemm_kernel_8x8_naive
	.type	sgemm_kernel_8x8_naive, %function
sgemm_kernel_8x8_naive:
.LFB0:
	.cfi_startproc
	sub	sp, sp, #320
	.cfi_def_cfa_offset 320
	str	w0, [sp, 44]
	str	s0, [sp, 40]
	str	x1, [sp, 32]
	str	x2, [sp, 24]
	str	s1, [sp, 20]
	str	x3, [sp, 8]
	str	w4, [sp, 16]
	str	w5, [sp, 4]
	str	wzr, [sp, 316]
	b	.L2
.L5:
	str	wzr, [sp, 312]
	b	.L3
.L4:
	ldr	w0, [sp, 312]
	lsl	w1, w0, 3
	ldr	w0, [sp, 316]
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	add	x1, sp, 48
	str	wzr, [x1, x0]
	ldr	w0, [sp, 312]
	add	w0, w0, 1
	str	w0, [sp, 312]
.L3:
	ldr	w0, [sp, 312]
	cmp	w0, 7
	ble	.L4
	ldr	w0, [sp, 316]
	add	w0, w0, 1
	str	w0, [sp, 316]
.L2:
	ldr	w0, [sp, 316]
	cmp	w0, 7
	ble	.L5
	str	wzr, [sp, 308]
	b	.L6
.L11:
	str	wzr, [sp, 316]
	b	.L7
.L10:
	str	wzr, [sp, 312]
	b	.L8
.L9:
	ldr	w0, [sp, 312]
	lsl	w1, w0, 3
	ldr	w0, [sp, 316]
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	add	x1, sp, 48
	ldr	s1, [x1, x0]
	ldr	w0, [sp, 308]
	lsl	w1, w0, 3
	ldr	w0, [sp, 316]
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 32]
	add	x0, x1, x0
	ldr	s2, [x0]
	ldr	w0, [sp, 308]
	lsl	w1, w0, 3
	ldr	w0, [sp, 312]
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 24]
	add	x0, x1, x0
	ldr	s0, [x0]
	fmul	s0, s2, s0
	ldr	w0, [sp, 312]
	lsl	w1, w0, 3
	ldr	w0, [sp, 316]
	add	w0, w1, w0
	fadd	s0, s1, s0
	sxtw	x0, w0
	lsl	x0, x0, 2
	add	x1, sp, 48
	str	s0, [x1, x0]
	ldr	w0, [sp, 312]
	add	w0, w0, 1
	str	w0, [sp, 312]
.L8:
	ldr	w0, [sp, 312]
	cmp	w0, 7
	ble	.L9
	ldr	w0, [sp, 316]
	add	w0, w0, 1
	str	w0, [sp, 316]
.L7:
	ldr	w0, [sp, 316]
	cmp	w0, 7
	ble	.L10
	ldr	w0, [sp, 308]
	add	w0, w0, 1
	str	w0, [sp, 308]
.L6:
	ldr	w1, [sp, 308]
	ldr	w0, [sp, 44]
	cmp	w1, w0
	blt	.L11
	ldr	s0, [sp, 20]
	fcmp	s0, #0.0
	bne	.L12
	str	wzr, [sp, 316]
	b	.L13
.L16:
	str	wzr, [sp, 312]
	b	.L14
.L15:
	ldr	w1, [sp, 316]
	ldr	w0, [sp, 16]
	mul	w1, w1, w0
	ldr	w2, [sp, 312]
	ldr	w0, [sp, 4]
	mul	w0, w2, w0
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 8]
	add	x0, x1, x0
	str	wzr, [x0]
	ldr	w0, [sp, 312]
	add	w0, w0, 1
	str	w0, [sp, 312]
.L14:
	ldr	w0, [sp, 312]
	cmp	w0, 7
	ble	.L15
	ldr	w0, [sp, 316]
	add	w0, w0, 1
	str	w0, [sp, 316]
.L13:
	ldr	w0, [sp, 316]
	cmp	w0, 7
	ble	.L16
	b	.L17
.L12:
	ldr	s1, [sp, 20]
	fmov	s0, 1.0e+0
	fcmp	s1, s0
	beq	.L17
	str	wzr, [sp, 316]
	b	.L18
.L21:
	str	wzr, [sp, 312]
	b	.L19
.L20:
	ldr	w1, [sp, 316]
	ldr	w0, [sp, 16]
	mul	w1, w1, w0
	ldr	w2, [sp, 312]
	ldr	w0, [sp, 4]
	mul	w0, w2, w0
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 8]
	add	x0, x1, x0
	ldr	s1, [x0]
	ldr	w1, [sp, 316]
	ldr	w0, [sp, 16]
	mul	w1, w1, w0
	ldr	w2, [sp, 312]
	ldr	w0, [sp, 4]
	mul	w0, w2, w0
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 8]
	add	x0, x1, x0
	ldr	s0, [sp, 20]
	fmul	s0, s1, s0
	str	s0, [x0]
	ldr	w0, [sp, 312]
	add	w0, w0, 1
	str	w0, [sp, 312]
.L19:
	ldr	w0, [sp, 312]
	cmp	w0, 7
	ble	.L20
	ldr	w0, [sp, 316]
	add	w0, w0, 1
	str	w0, [sp, 316]
.L18:
	ldr	w0, [sp, 316]
	cmp	w0, 7
	ble	.L21
.L17:
	ldr	s1, [sp, 40]
	fmov	s0, 1.0e+0
	fcmp	s1, s0
	bne	.L22
	str	wzr, [sp, 316]
	b	.L23
.L26:
	str	wzr, [sp, 312]
	b	.L24
.L25:
	ldr	w1, [sp, 316]
	ldr	w0, [sp, 16]
	mul	w1, w1, w0
	ldr	w2, [sp, 312]
	ldr	w0, [sp, 4]
	mul	w0, w2, w0
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 8]
	add	x0, x1, x0
	ldr	s1, [x0]
	ldr	w0, [sp, 312]
	lsl	w1, w0, 3
	ldr	w0, [sp, 316]
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	add	x1, sp, 48
	ldr	s0, [x1, x0]
	ldr	w1, [sp, 316]
	ldr	w0, [sp, 16]
	mul	w1, w1, w0
	ldr	w2, [sp, 312]
	ldr	w0, [sp, 4]
	mul	w0, w2, w0
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 8]
	add	x0, x1, x0
	fadd	s0, s1, s0
	str	s0, [x0]
	ldr	w0, [sp, 312]
	add	w0, w0, 1
	str	w0, [sp, 312]
.L24:
	ldr	w0, [sp, 312]
	cmp	w0, 7
	ble	.L25
	ldr	w0, [sp, 316]
	add	w0, w0, 1
	str	w0, [sp, 316]
.L23:
	ldr	w0, [sp, 316]
	cmp	w0, 7
	ble	.L26
	b	.L32
.L22:
	str	wzr, [sp, 316]
	b	.L28
.L31:
	str	wzr, [sp, 312]
	b	.L29
.L30:
	ldr	w1, [sp, 316]
	ldr	w0, [sp, 16]
	mul	w1, w1, w0
	ldr	w2, [sp, 312]
	ldr	w0, [sp, 4]
	mul	w0, w2, w0
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 8]
	add	x0, x1, x0
	ldr	s1, [x0]
	ldr	w0, [sp, 312]
	lsl	w1, w0, 3
	ldr	w0, [sp, 316]
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	add	x1, sp, 48
	ldr	s2, [x1, x0]
	ldr	s0, [sp, 40]
	fmul	s0, s2, s0
	ldr	w1, [sp, 316]
	ldr	w0, [sp, 16]
	mul	w1, w1, w0
	ldr	w2, [sp, 312]
	ldr	w0, [sp, 4]
	mul	w0, w2, w0
	add	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 8]
	add	x0, x1, x0
	fadd	s0, s1, s0
	str	s0, [x0]
	ldr	w0, [sp, 312]
	add	w0, w0, 1
	str	w0, [sp, 312]
.L29:
	ldr	w0, [sp, 312]
	cmp	w0, 7
	ble	.L30
	ldr	w0, [sp, 316]
	add	w0, w0, 1
	str	w0, [sp, 316]
.L28:
	ldr	w0, [sp, 316]
	cmp	w0, 7
	ble	.L31
.L32:
	nop
	add	sp, sp, 320
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE0:
	.size	sgemm_kernel_8x8_naive, .-sgemm_kernel_8x8_naive
	.align	2
	.global	sgemm_kernel_8x8_neon_fullyunroll
	.type	sgemm_kernel_8x8_neon_fullyunroll, %function
sgemm_kernel_8x8_neon_fullyunroll:
.LFB1:
	.cfi_startproc
	sub	sp, sp, #64
	.cfi_def_cfa_offset 64
	str	w0, [sp, 44]
	str	s0, [sp, 40]
	str	x1, [sp, 32]
	str	x2, [sp, 24]
	str	s1, [sp, 20]
	str	x3, [sp, 8]
	str	w4, [sp, 16]
	str	w5, [sp, 4]
	ldr	w0, [sp, 44]
	add	w1, w0, 7
	cmp	w0, 0
	csel	w0, w1, w0, lt
	asr	w0, w0, 3
	str	w0, [sp, 56]
	ldr	w0, [sp, 44]
	negs	w1, w0
	and	w0, w0, 7
	and	w1, w1, 7
	csneg	w0, w0, w1, mi
	str	w0, [sp, 52]
	ldr	w0, [sp, 4]
	lsl	w1, w0, 1
	ldr	x0, [sp, 8]
#APP
// 77 "kernel_8x8.c" 1
	ld1 {v16.4S-v17.4S}, [x0], x1
	ld1 {v18.4S-v19.4S}, [x0], x1
	ld1 {v20.4S-v21.4S}, [x0], x1
	ld1 {v22.4S-v23.4S}, [x0], x1
	ld1 {v24.4S-v25.4S}, [x0], x1
	ld1 {v26.4S-v27.4S}, [x0], x1
	ld1 {v28.4S-v29.4S}, [x0], x1
	ld1 {v30.4S-v31.4S}, [x0], x1
	
// 0 "" 2
#NO_APP
	str	wzr, [sp, 60]
	b	.L34
.L35:
	ldr	x1, [sp, 32]
	ldr	x2, [sp, 24]
	ldr	w3, [sp, 40]
#APP
// 93 "kernel_8x8.c" 1
	ld1 {v0.4S-v1.4S}, [x1]
	ld1 {v2.4S-v3.4S}, [x2]
	dup v4.4S, w3
	fmul v0.4S, v0.4S, v4.4S
	fmul v1.4S, v1.4S, v4.4S
	fmla v16.4S, v0.4S, v2.4S[0]
	fmla v17.4S, v1.4S, v2.4S[0]
	fmla v18.4S, v0.4S, v2.4S[1]
	fmla v19.4S, v1.4S, v2.4S[1]
	fmla v20.4S, v0.4S, v2.4S[2]
	fmla v21.4S, v1.4S, v2.4S[2]
	fmla v22.4S, v0.4S, v2.4S[3]
	fmla v23.4S, v1.4S, v2.4S[3]
	fmla v24.4S, v0.4S, v3.4S[0]
	fmla v25.4S, v1.4S, v3.4S[0]
	fmla v26.4S, v0.4S, v3.4S[1]
	fmla v27.4S, v1.4S, v3.4S[1]
	fmla v28.4S, v0.4S, v3.4S[2]
	fmla v29.4S, v1.4S, v3.4S[2]
	fmla v30.4S, v0.4S, v3.4S[3]
	fmla v31.4S, v1.4S, v3.4S[3]
	
// 0 "" 2
#NO_APP
	ldr	x0, [sp, 32]
	add	x0, x0, 32
	str	x0, [sp, 32]
	ldr	x0, [sp, 24]
	add	x0, x0, 32
	str	x0, [sp, 24]
	ldr	w0, [sp, 60]
	add	w0, w0, 1
	str	w0, [sp, 60]
.L34:
	ldr	w1, [sp, 60]
	ldr	w0, [sp, 44]
	cmp	w1, w0
	blt	.L35
	ldr	w0, [sp, 4]
	lsl	w1, w0, 1
	ldr	x0, [sp, 8]
#APP
// 129 "kernel_8x8.c" 1
	st1 {v16.4S-v17.4S}, [x0], x1
	st1 {v18.4S-v19.4S}, [x0], x1
	st1 {v20.4S-v21.4S}, [x0], x1
	st1 {v22.4S-v23.4S}, [x0], x1
	st1 {v24.4S-v25.4S}, [x0], x1
	st1 {v26.4S-v27.4S}, [x0], x1
	st1 {v28.4S-v29.4S}, [x0], x1
	st1 {v30.4S-v31.4S}, [x0], x1
	
// 0 "" 2
#NO_APP
	nop
	add	sp, sp, 64
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE1:
	.size	sgemm_kernel_8x8_neon_fullyunroll, .-sgemm_kernel_8x8_neon_fullyunroll
	.ident	"GCC: (Debian 10.2.1-6) 10.2.1 20210110"
	.section	.note.GNU-stack,"",@progbits
