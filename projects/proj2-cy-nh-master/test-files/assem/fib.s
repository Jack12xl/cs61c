#Calculates the Nth Fibonacci number.
first:
	# Change the immediate here to input an N.
	addiu		$a0, $0, 10
	
	
	addiu		$t0, $0, 0
	addiu    	$t1, $0, 1
fib:
	beq 		$a0, $0, finish
	add 		$t2, $t1, $t0

	addiu 		$t0, $t1, 0
	add 		$t1, $t2, $0
	addiu 		$a0, $a0, -1
	j			fib
finish:
	addiu		$v0, $t0, 0
	jr 			$ra
