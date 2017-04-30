# calculate C($a0,$a1)
nchoosek:
	# prologue
	addiu $sp, $sp, -16
	sw	$ra, 0($sp) 
	sw	$s1, 4($sp)
	sw	$s0, 8($sp)
	sw      $s2, 12($sp)

    
	beq	$a1, $0, return1
	beq	$a0, $a1, return1
	beq	$a0, $0, return0
	blt	$a0, $a1, return0

	addiu	$a0, $a0, -1
	addu	$s0, $a0, $0
	addu	$s1, $a1, $0
	jal	nchoosek
	addu	$s2, $v0, $0
	addu	$a0, $s0, $0
	addiu	$a1, $s1, -1
	jal	nchoosek
	addu	$v0, $v0, $s2
	j	return
return0:
	addu	$v0, $0, $0
	j	return
return1:
	addiu	$v0, $0, 1

return:
	# epilogue
	### YOUR CODE HERE ###
	lw      $s2, 12($sp)
	lw	$s0, 8($sp)
	lw	$s1, 4($sp)
	lw	$ra, 0($sp)
	addiu	$sp, $sp, 16
	jr	$ra
