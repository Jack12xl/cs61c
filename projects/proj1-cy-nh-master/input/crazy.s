start:
		addiu   $t3, $0, 5
crazy:
        beq     $0, $0, so_crazy

############ COMMENTS ##############
############ COMMENTS ##############
############ COMMENTS ##############
just_kidding:
		j      start
############ COMMENTS ##############
############ COMMENTS ##############

so_crazy: 
		addiu   $t3, $t3, 2
		addiu   $t4, $0, 54
		blt     $t3, $t4, crazy
		blt     $t4, $t3, craziness_over

craziness_over:
		j       just_kidding

# MOAR COMMENTS
# MOAR COMMENTS
# MOAR COMMENTS
# MOAR COMMENTS