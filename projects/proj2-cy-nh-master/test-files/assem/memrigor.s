#This program tests the mem component of the CPU.
	# Initialize $sp
	lui		$sp, 0x0fff
	ori		$sp, $sp, 0xfffc

	# Initialize $ra
	lui     $ra, 0x0010
	addiu	$s0, $0, 123
	addiu	$s1, $0, 456
	addiu	$s2, $0, -9999
	# Store $ra and other variables.
	addiu   $sp, $sp, -16
	sw		$ra, 0($sp)
	sw		$s0, 4($sp)
	sw		$s1, 8($sp)
	sw		$s2, 12($sp)

	addiu	$a0, $0, 3
	jal		quadruple

	addiu	$s0, $v0, 0
	addiu	$a0, $0, 4
	jal		quadruple

	addiu	$s1, $a0, 0
	addu	$a0, $s0, $s1
	jal		quadruple

	#should set $v0 to ((3*4)+(4*4))*4 = 112

	lw		$s2, 12($sp)
	lw		$s1, 8($sp)
	lw		$s0, 4($sp)
	lw		$ra, 0($sp)
	addiu   $sp, $sp, 16



	jr $ra

#This function returns ($a0)*4.
quadruple:
	addu	$a0, $a0, $a0
	addu	$a0, $a0, $a0
	addiu   $v0, $a0, 0
	jr $ra