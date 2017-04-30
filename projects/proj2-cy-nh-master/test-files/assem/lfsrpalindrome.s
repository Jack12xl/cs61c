LsfrPalindrome:
	# Comment out the next line in order to de-hardcode $a0.
	#addiu		$a0, $0, 1

	addiu		$v0, $a0, 0
	addiu       $t0, $0, 0
LfsrLoop:
	lfsr        $v0, $v0
	bitpal      $t1, $v0
	beq			$v0, $a0, End
	beq         $t0, $t1, LfsrLoop
End:
	# Either: bitpal stored a 1 if $v0 was a palindrome, so return it.
	# Or: bitpal reached the seed value again before reaching a palindrome, so return it.
	jr          $ra 