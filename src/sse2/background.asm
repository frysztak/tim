section .rodata
    align 16
    bgrMask: db 0xFF,0xFF,0xFF,0,0,0,0,0,0,0,0,0,0,0,0,0
    bgrMask2: db 0xFF,0,0,0,0xFF,0,0,0,0xFF,0,0,0,0xFF,0,0,0
    const_6.25: times 4 dd 6.25
    const_1:    times 4 dd 1.0
    const_eta:  times 4 dd 1.57496099457e+01
    const_exp0: times 4 dd 9.94663531855e-01
    const_exp1: times 4 dd 9.30963170380e-01
    const_exp2: times 4 dd 3.08826533369e-01
    const_log0: times 4 dd 3.674002733
    const_log1: times 4 dd 6.69321654748e-01
    const_log2: times 4 dd -3.72382626847e-02
    const_0.5: times 4 dd 0.5
    const_minus0.5: times 4 dd -0.5

section .text
global processPixels_SSE2

; this code is buggy and incomplete
; foreground mask contains much more background pixels than it should
; background image is not updated
; background stddev is not updated
; performance-wise, it's about 2 fps slower than fully-implemented intrinsic version
; today's lesson is that compiler is better than me
; but overall, it's been interesting experience writing vector code in asm

processPixels_SSE2:
    ; RDI -> frame
    ; RSI -> gaussian
    ; RDX -> currentBackground
    ; RCX -> currentStdDev
    ; XMM0 -> learningRate
    ; XMM1 -> initalVariance
    ; XMM2 -> initialWeight
    ; XMM3 -> foregroundThreshold
    ; EAX -> returned value
    ; we can't afford to keep parameters in XMM registers, 
    ; so we'll push them on the stack

    push rbp
    mov rbp, rsp
    sub rsp, 0x50 ; reserve 5*16 = 80 bytes 
    shufps xmm0, xmm0, 0 ; 'broadcast' first value to all floats in register
    shufps xmm1, xmm1, 0
    shufps xmm2, xmm2, 0
    shufps xmm3, xmm3, 0
    movaps [rsp+0x00], xmm0
    movaps [rsp+0x10], xmm1
    movaps [rsp+0x20], xmm2
    movaps [rsp+0x30], xmm3

    movdqu xmm0, [rdi]
    movdqa xmm1, [bgrMask]
    movdqa xmm2, xmm1
    movdqa xmm3, xmm1
    movdqa xmm4, xmm1
    pslldq xmm2, 3
    pslldq xmm3, 6
    pslldq xmm4, 9
    pand xmm1, xmm0 ; b1g1r1
    pand xmm2, xmm0 ; b2g2r2
    pand xmm3, xmm0 ; b3g3r3
    pand xmm4, xmm0 ; b4g4r4
    pslldq xmm2, 1
    pslldq xmm3, 2
    pslldq xmm4, 3
    por xmm1, xmm2
    por xmm3, xmm4
    por xmm3, xmm1 
    ; xmm3 now contains B1G1R10 B2G2R20 B3G3R30 B4G4R40
    movdqa xmm4, [bgrMask2]
    movdqa xmm5, xmm4
    movdqa xmm6, xmm4
    pslldq xmm5, 1
    pslldq xmm6, 2
    ; extract B1B2B3B4 and so on
    pand xmm4, xmm3 ; B 
    pand xmm5, xmm3 ; G
    pand xmm6, xmm3 ; R
    psrldq xmm5, 1
    psrldq xmm6, 2
    ; convert to float
    cvtdq2ps xmm0, xmm4
    cvtdq2ps xmm1, xmm5
    cvtdq2ps xmm2, xmm6
    ; xmm0 now contains float B1B2B3B4, xmm1 -- G, xmm2 -- R
    ; xmm3 -> matched, zero it out
    pxor xmm3, xmm3
    ; use rax as loop counter
    xor rax, rax

per_gaussian_loop:
    ; xmm4 -> meanB
    ; xmm5 -> meanG
    ; xmm6 -> meanR
    ; xmm7 -> variance
    ; xmm8 -> weight
    ; xmm9 -> distance
    pxor xmm9, xmm9
    movaps xmm4, [rsi + rax + 0x00] ; meanB
    movaps xmm5, [rsi + rax + 0x30] ; meanG
    movaps xmm6, [rsi + rax + 0x60] ; meanR
    movaps xmm7, [rsi + rax + 0x90] ; variance
    movaps xmm8, [rsi + rax + 0xC0] ; weight
    ; process meanB
    movaps xmm10, xmm4
    subps xmm10, xmm0 ; meanB - B
    mulps xmm10, xmm10 ; square
    addps xmm9, xmm10 ; add to distance
    ; process meanG
    movaps xmm10, xmm5
    subps xmm10, xmm1 ; meanG - G
    mulps xmm10, xmm10 ; square
    addps xmm9, xmm10 ; add to distance
    ; process meanR
    movaps xmm10, xmm6
    subps xmm10, xmm2 ; meanR - R
    mulps xmm10, xmm10 ; square
    addps xmm9, xmm10 ; add to distance
    ; xmm10 -> 6.25 * variance
    movaps xmm10, xmm7
    mulps xmm10, [const_6.25]
    cmpps xmm10, xmm9, 14 ; greater-than (requires AVX I think)
    ; xmm10 now contains 'matched' mask
    movaps xmm11, xmm3
    andnps xmm11, xmm10    
    ; xmm11 -> mask
    orps xmm3, xmm11
    ; calculate exponent
    ; copy distance to xmm10
    movaps xmm10, xmm9
    mulps xmm10, [const_minus0.5]
    rcpps xmm12, xmm7 ; reciprocal of variance
    mulps xmm10, xmm12
    ; exponent is now in xmm10
    ; calculate approximate exp(x)
    movaps xmm12, [const_exp0]
    movaps xmm13, xmm10
    mulps xmm13, [const_exp1]
    addps xmm12, xmm13
    movaps xmm13, xmm10
    mulps xmm13, xmm13
    mulps xmm13, [const_exp2]
    addps xmm12, xmm13
    ; exp(x) is now in xmm12
    movaps xmm13, xmm7
    mulps xmm13, [const_eta]
    rcpps xmm13, xmm13
    mulps xmm12, xmm13 ; eta
    mulps xmm12, [rsp] ; multiply by learning rate
    ; xmm12 -> rho
    movaps xmm10, [const_1]
    subps xmm10, xmm12
    ; register to preserve:
    ; xmm0..6 -> BGR, matched mask, meanBGR
    ; xmm7 -> variance
    ; xmm8 -> weight
    ; xmm9 -> distance
    ; xmm10 -> 1.0 - rho
    ; xmm11 -> mask
    ; xmm12 -> rho

    ; I'd like to get rid of 'distance' (xmm9),
    ; but it's needed for newVariance, so calculate that first
    ; xmm13 -> variance
    movaps xmm13, xmm7
    mulps xmm13, xmm10
    mulps xmm9, xmm12
    addps xmm13, xmm9
    andps xmm13, xmm11
    movaps xmm14, xmm11 ; copy 'mask'
    andnps xmm14, xmm7
    orps xmm13, xmm14
    movaps [rsi + rax + 0x90], xmm13 ; out with it
    ; we're done with variance
    ; register to preserve:
    ; xmm0..6 -> BGR, matched mask, meanBGR
    ; xmm8 -> weight
    ; xmm10 -> 1.0 - rho
    ; xmm11 -> mask
    ; xmm12 -> rho
    
    ; weight
    movaps xmm9, xmm8
    movaps xmm13, [const_1]
    subps xmm13, [rsp] ; xmm13 -> 1.0 - learningRate
    mulps xmm9, xmm13 ; xmm9 -> newWeight
    ; at this point in intrinsic-based code I invert 'mask'
    ; but I've gotten more clever and now I see it's not necessary
    movaps xmm14, xmm11
    andnps xmm14, xmm9
    andps xmm8, xmm11
    orps xmm8, xmm14
    movaps [rsi + rax + 0xC0], xmm8 
    ; register to preserve:
    ; xmm0..6 -> BGR, matched mask, meanBGR
    ; xmm10 -> 1.0 - rho
    ; xmm11 -> mask
    ; xmm12 -> rho

    ; meanB
    movaps xmm7, xmm4 
    mulps xmm7, xmm10 ; xmm7 -> newMeanB
    movaps xmm8, xmm0 ; xmm8 -> B
    mulps xmm8, xmm12
    addps xmm7, xmm8
    andps xmm7, xmm11
    movaps xmm13, xmm11
    andnps xmm13, xmm4
    orps xmm13, xmm7
    movaps [rsi + rax], xmm13
   
    ; meanG
    movaps xmm7, xmm5 
    mulps xmm7, xmm10 ; xmm7 -> newMeanG
    movaps xmm8, xmm1 ; xmm8 -> G
    mulps xmm8, xmm12
    addps xmm7, xmm8
    andps xmm7, xmm11
    movaps xmm13, xmm11
    andnps xmm13, xmm4
    orps xmm13, xmm7
    movaps [rsi + rax + 0x30], xmm13

    ; meanR
    movaps xmm7, xmm6 
    mulps xmm7, xmm10 ; xmm7 -> newMeanR
    movaps xmm8, xmm2 ; xmm8 -> R
    mulps xmm8, xmm12
    addps xmm7, xmm8
    andps xmm7, xmm11
    movaps xmm13, xmm11
    andnps xmm13, xmm4
    orps xmm13, xmm7
    movaps [rsi + rax + 0x60], xmm13

    ; loop maintenance
    add rax, 16
    cmp rax, 48
    jne per_gaussian_loop
    ; end per_gaussian_loop

    ; register to preserve:
    ; xmm0..3 -> BGR, matched mask

    ; now, in case some Gaussian still didn't match, we gotta update (essentialy reset)
    ; the least probable Gaussian.
    ; to determine which Gaussian is the least probable, we should divide its weight by variance.
    ; but that'd be slow, so let's just use weight alone as the criterion.

    ; load all weights
    movaps xmm4, [rsi + 0xC0]
    movaps xmm5, [rsi + 16 + 0xC0]
    movaps xmm6, [rsi + 32 + 0xC0]
    ; find minimal weight in all of Gaussians
    movaps xmm7, xmm4
    minps xmm7, xmm5
    minps xmm7, xmm6 ; xmm7 -> min weights
    ; it's probably most effient to invert 'matched mask' in xmm3
    pcmpeqb xmm8, xmm8 ; xmm8 -> all ones
    xorps xmm3, xmm8 ; xmm3 -> inverted matched mask

; check if first Gaussian is the the least probable one
    movaps xmm8, xmm4
    cmpeqps xmm8, xmm7
    andps xmm8, xmm3 ; xmm8 -> is minimal? mask

    ; modify meanB
    movaps xmm9, [rsi]
    movaps xmm10, xmm8
    andps xmm10, xmm0
    movaps xmm11, xmm8
    andnps xmm11, xmm9 
    orps xmm11, xmm10
    movaps [rsi], xmm11

    ; modify meanG
    movaps xmm9, [rsi + 0x30]
    movaps xmm10, xmm8
    andps xmm10, xmm1
    movaps xmm11, xmm8
    andnps xmm11, xmm9 
    orps xmm11, xmm10
    movaps [rsi + 0x30], xmm11

    ; modify meanG
    movaps xmm9, [rsi + 0x60]
    movaps xmm10, xmm8
    andps xmm10, xmm2
    movaps xmm11, xmm8
    andnps xmm11, xmm9 
    orps xmm11, xmm10
    movaps [rsi + 0x60], xmm11

    ; modify variance
    movaps xmm9, [rsi + 0x90]
    movaps xmm10, xmm8
    andps xmm10, [rsp + 0x10] ; initial variance
    movaps xmm11, xmm8
    andnps xmm11, xmm9
    orps xmm11, xmm10
    movaps [rsi + 0x90], xmm11

    ; modify weight, but only locally
    movaps xmm9, xmm8
    andps xmm9, [rsp + 0x20] ; initial weight
    andnps xmm8, xmm4 ; at this point we can overwrite isMin mask, we won't need it anymore
    orps xmm8, xmm9
    movaps xmm4, xmm8

; second Gaussian 
    movaps xmm8, xmm5
    cmpeqps xmm8, xmm7
    andps xmm8, xmm3 ; xmm8 -> is minimal? mask

    ; modify meanB
    movaps xmm9, [rsi + 16]
    movaps xmm10, xmm8
    andps xmm10, xmm0
    movaps xmm11, xmm8
    andnps xmm11, xmm9 
    orps xmm11, xmm10
    movaps [rsi + 16], xmm11

    ; modify meanG
    movaps xmm9, [rsi + 16 + 0x30]
    movaps xmm10, xmm8
    andps xmm10, xmm1
    movaps xmm11, xmm8
    andnps xmm11, xmm9 
    orps xmm11, xmm10
    movaps [rsi + 16 + 0x30], xmm11

    ; modify meanG
    movaps xmm9, [rsi + 16 + 0x60]
    movaps xmm10, xmm8
    andps xmm10, xmm2
    movaps xmm11, xmm8
    andnps xmm11, xmm9 
    orps xmm11, xmm10
    movaps [rsi + 16 + 0x60], xmm11

    ; modify variance
    movaps xmm9, [rsi + 16 + 0x90]
    movaps xmm10, xmm8
    andps xmm10, [rsp + 0x10] ; initial variance
    movaps xmm11, xmm8
    andnps xmm11, xmm9
    orps xmm11, xmm10
    movaps [rsi + 16 + 0x90], xmm11

    ; modify weight, but only locally
    movaps xmm9, xmm8
    andps xmm9, [rsp + 0x20] ; initial weight
    andnps xmm8, xmm5 ; at this point we can overwrite isMin mask, we won't need it anymore
    orps xmm8, xmm9
    movaps xmm5, xmm8

; third Gaussian 
    movaps xmm8, xmm6
    cmpeqps xmm8, xmm7
    andps xmm8, xmm3 ; xmm8 -> is minimal? mask

    ; modify meanB
    movaps xmm9, [rsi + 32]
    movaps xmm10, xmm8
    andps xmm10, xmm0
    movaps xmm11, xmm8
    andnps xmm11, xmm9 
    orps xmm11, xmm10
    movaps [rsi + 32], xmm11

    ; modify meanG
    movaps xmm9, [rsi + 32 + 0x30]
    movaps xmm10, xmm8
    andps xmm10, xmm1
    movaps xmm11, xmm8
    andnps xmm11, xmm9 
    orps xmm11, xmm10
    movaps [rsi + 32 + 0x30], xmm11

    ; modify meanG
    movaps xmm9, [rsi + 32 + 0x60]
    movaps xmm10, xmm8
    andps xmm10, xmm2
    movaps xmm11, xmm8
    andnps xmm11, xmm9 
    orps xmm11, xmm10
    movaps [rsi + 32 + 0x60], xmm11

    ; modify variance
    movaps xmm9, [rsi + 32 + 0x90]
    movaps xmm10, xmm8
    andps xmm10, [rsp + 0x10] ; initial variance
    movaps xmm11, xmm8
    andnps xmm11, xmm9
    orps xmm11, xmm10
    movaps [rsi + 32 + 0x90], xmm11

    ; modify weight, but only locally
    movaps xmm9, xmm8
    andps xmm9, [rsp + 0x20] ; initial weight
    andnps xmm8, xmm6 ; at this point we can overwrite isMin mask, we won't need it anymore
    orps xmm8, xmm9
    movaps xmm6, xmm8

    ; make sum of weights equal to one. save weights. 
    movaps xmm7, xmm4
    addps xmm7, xmm5
    addps xmm7, xmm6
    rcpps xmm7, xmm7
    mulps xmm4, xmm7
    mulps xmm5, xmm7
    mulps xmm6, xmm7
    movaps [rsi + 0xC0], xmm4
    movaps [rsi + 16 + 0xC0], xmm5
    movaps [rsi + 32 + 0xC0], xmm6

    ; we're done updating Gaussians.
    ; now, let's calculate foreground mask and update current background image
    movaps xmm7, xmm4
    maxps xmm7, xmm5
    maxps xmm7, xmm6 ; xmm7 -> max weight
    xorps xmm3, xmm3 ; xmm3 -> fg mask
    xorps xmm8, xmm8 ; xmm8 -> bgB
    xorps xmm9, xmm9 ; xmm9 -> bgG
    xorps xmm10, xmm10 ; xmm10 -> bgR
    xorps xmm11, xmm11 ; xmm11 -> bgVariance

    ; weights won't be needed anymore, 
    ; so we'll turn them into isMax masks
; first Gaussian
    cmpeqps xmm4, xmm7    
    ; load variance
    movaps xmm12, [rsi + 0x90]
    ; update bgVariance
    movaps xmm13, xmm4
    andps xmm13, xmm12
    movaps xmm14, xmm4
    andnps xmm14, xmm11
    orps xmm14, xmm13
    movaps xmm11, xmm14
    ; calculate epsilonBg
    movaps xmm13, [const_log0]
    movaps xmm14, xmm12
    mulps xmm14, [const_log1]
    addps xmm13, xmm14
    movaps xmm14, xmm12
    mulps xmm14, xmm14
    mulps xmm14, [const_log2]
    addps xmm13, xmm14 ; xmm13 -> epsilonBg
    ; from now on we won't need variance, but rather reciprocal of variance
    rcpps xmm12, xmm12

    ; load meanB
    movaps xmm14, [rsi]
    ; first update bgB
    ; we're out of registers, push xmm3 onto stack
    movaps [rsp+0x40], xmm3
    movaps xmm15, xmm4
    andps xmm15, xmm14
    movaps xmm3, xmm4 
    andnps xmm3, xmm8
    orps xmm15, xmm3 
    movaps xmm8, xmm15
    ; add to epsilonBg
    subps xmm14, xmm0 ; meanB - B
    mulps xmm14, xmm14
    mulps xmm14, [const_0.5]
    mulps xmm14, xmm12
    addps xmm13, xmm14

    ; load meanG
    movaps xmm14, [rsi + 0x30]
    ; first update bgG
    movaps xmm15, xmm5
    andps xmm15, xmm14
    movaps xmm3, xmm5 
    andnps xmm3, xmm8
    orps xmm15, xmm3
    movaps xmm9, xmm15
    ; add to epsilonBg
    subps xmm14, xmm1 ; meanG - G
    mulps xmm14, xmm14
    mulps xmm14, [const_0.5]
    mulps xmm14, xmm12
    addps xmm13, xmm14

    ; load meanR
    movaps xmm14, [rsi + 0x60]
    ; first update bgR
    movaps xmm15, xmm6
    andps xmm15, xmm14
    movaps xmm3, xmm6 
    andnps xmm3, xmm8
    orps xmm15, xmm3
    movaps xmm10, xmm15
    ; add to epsilonBg
    subps xmm14, xmm2 ; meanR - R
    mulps xmm14, xmm14
    mulps xmm14, [const_0.5]
    mulps xmm14, xmm12
    addps xmm13, xmm14
    movaps xmm3, [rsp+0x40] ; restore

    cmpps xmm13, [rsp + 0x30], 14 ; greater-than
    ; update fg mask
    movaps xmm14, xmm4
    andps xmm14, xmm13
    movaps xmm15, xmm4
    andnps xmm15, xmm3
    orps xmm15, xmm14
    movaps xmm3, xmm15
; second Gaussian
    cmpeqps xmm5, xmm7    
    ; load variance
    movaps xmm12, [rsi + 16 + 0x90]
    ; update bgVariance
    movaps xmm13, xmm5
    andps xmm13, xmm12
    movaps xmm14, xmm5
    andnps xmm14, xmm11
    orps xmm14, xmm13
    movaps xmm11, xmm14
    ; calculate epsilonBg
    movaps xmm13, [const_log0]
    movaps xmm14, xmm12
    mulps xmm14, [const_log1]
    addps xmm13, xmm14
    movaps xmm14, xmm12
    mulps xmm14, xmm14
    mulps xmm14, [const_log2]
    addps xmm13, xmm14 ; xmm13 -> epsilonBg
    ; from now on we won't need variance, but rather reciprocal of variance
    rcpps xmm12, xmm12

    ; load meanB
    movaps xmm14, [rsi + 16]
    ; first update bgB
    ; we're out of registers, push xmm3 onto stack
    movaps [rsp+0x40], xmm3
    movaps xmm15, xmm5
    andps xmm15, xmm14
    movaps xmm3, xmm5 
    andnps xmm3, xmm8
    orps xmm15, xmm3 
    movaps xmm8, xmm15
    ; add to epsilonBg
    subps xmm14, xmm0 ; meanB - B
    mulps xmm14, xmm14
    mulps xmm14, [const_0.5]
    mulps xmm14, xmm12
    addps xmm13, xmm14

    ; load meanG
    movaps xmm14, [rsi + 16 + 0x30]
    ; first update bgG
    movaps xmm15, xmm5
    andps xmm15, xmm14
    movaps xmm3, xmm5 
    andnps xmm3, xmm8
    orps xmm15, xmm3
    movaps xmm9, xmm15
    ; add to epsilonBg
    subps xmm14, xmm1 ; meanG - G
    mulps xmm14, xmm14
    mulps xmm14, [const_0.5]
    mulps xmm14, xmm12
    addps xmm13, xmm14

    ; load meanR
    movaps xmm14, [rsi + 16 + 0x60]
    ; first update bgR
    movaps xmm15, xmm6
    andps xmm15, xmm14
    movaps xmm3, xmm6 
    andnps xmm3, xmm8
    orps xmm15, xmm3
    movaps xmm10, xmm15
    ; add to epsilonBg
    subps xmm14, xmm2 ; meanR - R
    mulps xmm14, xmm14
    mulps xmm14, [const_0.5]
    mulps xmm14, xmm12
    addps xmm13, xmm14
    movaps xmm3, [rsp+0x40] ; restore

    cmpps xmm13, [rsp + 0x30], 14 ; greater-than
    ; update fg mask
    movaps xmm14, xmm5
    andps xmm14, xmm13
    movaps xmm15, xmm5
    andnps xmm15, xmm3
    orps xmm15, xmm14
    movaps xmm3, xmm15
; third Gaussian
    cmpeqps xmm6, xmm7    
    ; load variance
    movaps xmm12, [rsi + 32 + 0x90]
    ; update bgVariance
    movaps xmm13, xmm6
    andps xmm13, xmm12
    movaps xmm14, xmm6
    andnps xmm14, xmm11
    orps xmm14, xmm13
    movaps xmm11, xmm14
    ; calculate epsilonBg
    movaps xmm13, [const_log0]
    movaps xmm14, xmm12
    mulps xmm14, [const_log1]
    addps xmm13, xmm14
    movaps xmm14, xmm12
    mulps xmm14, xmm14
    mulps xmm14, [const_log2]
    addps xmm13, xmm14 ; xmm13 -> epsilonBg
    ; from now on we won't need variance, but rather reciprocal of variance
    rcpps xmm12, xmm12

    ; load meanB
    movaps xmm14, [rsi + 32]
    ; first update bgB
    ; we're out of registers, push xmm3 onto stack
    movaps [rsp+0x40], xmm3
    movaps xmm15, xmm6
    andps xmm15, xmm14
    movaps xmm3, xmm6 
    andnps xmm3, xmm8
    orps xmm15, xmm3 
    movaps xmm8, xmm15
    ; add to epsilonBg
    subps xmm14, xmm0 ; meanB - B
    mulps xmm14, xmm14
    mulps xmm14, [const_0.5]
    mulps xmm14, xmm12
    addps xmm13, xmm14

    ; load meanG
    movaps xmm14, [rsi + 32 + 0x30]
    ; first update bgG
    movaps xmm15, xmm6
    andps xmm15, xmm14
    movaps xmm3, xmm6 
    andnps xmm3, xmm8
    orps xmm15, xmm3
    movaps xmm9, xmm15
    ; add to epsilonBg
    subps xmm14, xmm1 ; meanG - G
    mulps xmm14, xmm14
    mulps xmm14, [const_0.5]
    mulps xmm14, xmm12
    addps xmm13, xmm14

    ; load meanR
    movaps xmm14, [rsi + 32 + 0x60]
    ; first update bgR
    movaps xmm15, xmm6
    andps xmm15, xmm14
    movaps xmm3, xmm6 
    andnps xmm3, xmm8
    orps xmm15, xmm3
    movaps xmm10, xmm15
    ; add to epsilonBg
    subps xmm14, xmm2 ; meanR - R
    mulps xmm14, xmm14
    mulps xmm14, [const_0.5]
    mulps xmm14, xmm12
    addps xmm13, xmm14
    movaps xmm3, [rsp+0x40] ; restore

    cmpps xmm13, [rsp + 0x30], 14 ; greater-than
    ; update fg mask
    movaps xmm14, xmm6
    andps xmm14, xmm13
    movaps xmm15, xmm6
    andnps xmm15, xmm3
    orps xmm15, xmm14
    ; TODO: update background image, update background stddev

    ; set function's return value
    movmskps edx, xmm15
    mov eax, edx
    mov r8d, edx
    and eax, 1 
    and r8d, 2
    mov r9d, edx
    mov r10d, edx
    and r9d, 4
    and r10d, 8
    shl r8d, 8
    shl r9d, 16
    shl r10d, 24
    or eax, r8d
    or eax, r9d
    or eax, r10d    

    mov rsp, rbp
    pop rbp
    ret

; vim: set ft=asm syntax=nasm ts=4 sw=4 sts=4 tw=0 fenc=utf-8 et: 
