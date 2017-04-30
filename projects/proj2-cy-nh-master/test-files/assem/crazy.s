start:
		addiu   $t3, $0, 5
crazy:
        beq     $0, $0, so_crazy

        # YOU BETTER NOT EXECUTE THE NEXT FEW LINES!

		addiu	$s0, $s0, 31
		addiu	$ra, $ra, 193
		addiu	$sp, $sp, -5
		sw		$t0, -1245($t0)

############ COMMENTS ##############
############ COMMENTS ##############
############ COMMENTS ##############
just_kidding:
		j      start
############ COMMENTS ##############
############ COMMENTS ##############

		# YOU BETTER NOT EXECUTE THE NEXT FEW LINES!

		addiu	$s0, $s0, 31
		addiu	$ra, $ra, 193
		addiu	$sp, $sp, -5
		sw		$t0, -1245($t0)

so_crazy: 
		addiu   $t3, $t3, 2
		addiu   $t4, $0, 54
		slt		$t0, $t3, $t4
		bne     $t0, $0, crazy
		slt		$t0, $t4, $t3
		bne     $t4, $t3, craziness_over

		# YOU BETTER NOT EXECUTE THE NEXT FEW LINES!

		addiu	$s0, $s0, 31
		addiu	$ra, $ra, 193
		addiu	$sp, $sp, -5
		sw		$t0, -1245($t0)

craziness_over:
		j       just_kidding

		# YOU BETTER NOT EXECUTE THE NEXT FEW LINES!

		addiu	$s0, $s0, 31
		addiu	$ra, $ra, 193
		addiu	$sp, $sp, -5
		sw		$t0, -1245($t0)

# MOAR COMMENTS
# MOAR COMMENTS
# MOAR COMMENTS
# MOAR COMMENTS