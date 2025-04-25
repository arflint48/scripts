#!/bin/bash

me=$(finger $USER | grep "Name" | awk '{print $4}')

if [[ -z $1 ]];
then
	echo "ERROR: no .out file passed"
	exit
fi

OUT=$1


if grep -q "PROGRAM SYSTEM MOLPRO" $OUT; then
	echo "***reading from a Molpro output file***"
	prog=1
elif grep -q "Gaussian(R)" $OUT; then
	echo "***reading from a Gaussian output file***"
	prog=2
else
	echo "ERROR: program used to generate $OUT not supported"
fi

#echo $prog
if [ $prog == 1 ]; then
	lin=$(grep -c "Molecule type: Linear" $OUT)
	if [ $lin -eq 0 ]; then
		sub=6
	else
		sub=5
	fi

	beg=$(grep -i -n "geometry=" $OUT | cut -f1 -d:)
	en=$(grep -n -m 1 "}" $OUT | cut -f1 -d:)
	beg=$((beg+1))
	en=$((en-1))
	dum=$(sed -n "${beg},${en}p" $OUT | grep -c -i "x")
	tot=$((en-beg-dum+1))
	COUNT=$((3*tot-sub+1-low-imag))
	
	molpro_xyz () {
	xyzblk=$((tot+3))
	#echo $xyzblk
	grep -A$xyzblk "Current geometry" $OUT | tail -n +5
	}
	
	molpro_vib () {
		grep -B$COUNT --line-buffered "CALCULATION OF NORMAL MODES FOR" $OUT | head -n -2 | awk '{print $2}'
	} 
	
	vibetc () {
	ivchk=$(grep "Imaginary Vibration" $OUT)
	if [[ -n $ivchk ]]; then
		ivline=$(grep -i -n "Imaginary Vibration  Wavenumber" $OUT | cut -f1 -d:)
		blank1=$((ivline+2))
		empty1=$(sed -n "${blank1}p" $OUT)
		if [[ -n $empty1 ]];
		then
			imag=0
			while [[ -n $empty1 ]];
			do
				sed -n "${blank1}p" $OUT | awk '{print "-"$2}'
				let "blank1=blank1+1"
				empty1=$(sed -n "${blank1}p" $OUT)
				let "imag=imag+1"
			done
		fi
	fi
	
	#Low Vibration
	lvline=$(grep -i -n "Low Vibration      Wavenumber" $OUT | cut -f1 -d:)
	blank2=$((lvline+sub+2))
	empty2=$(sed -n "${blank2}p" $OUT)
	if [[ -n $empty2 ]];
	then
		low=0
		while [[ -n $empty2 ]];
		do
			sed -n "${blank2}p" $OUT | awk '{print $2}'
			let "blank2=blank2+1"
			empty2=$(sed -n "${blank2}p" $OUT)
			let "low=low+1"
		done
	fi
	}
	
	molpro_zpe () {
		zpe=$(grep "Zero point energy" $OUT | awk '{print $4}')
		echo "Zero-Point Vibrational Energy (Hartrees): $zpe"
	}
	
	#Rotational Constants
	rotcon () {
	for i in {3..5}; do
		grep -i "Rotational constants" $OUT | tail -n 1 | awk '{print $'$i'}'
	done
	}

	molpro_energy () {
	m_ee=$(tail -n $((6)) $OUT | head -n 1 | awk '{print $3}')
	echo "Electronic Energy (Hartrees): $m_ee"
	msum=$(echo "scale = 8; $m_ee+$zpe" | bc)
	echo "Electronic Energy + Zero-Point Vibrational Energy (Hartrees): $msum"
	mkcal=$(echo "scale = 2; $msum * 627.509" | bc)
	echo "Electronic Energy + Zero-Point Vibrational Energy (kcal mol$^{-1}$): $mkcal"
	}

	#molpro_xyz
	#vibetc
	#molpro_vib
	#molpro_zpe
	#rotcon
fi

if [ $prog == 2 ]; then
	gaussian_xyz () {
	beg=$(grep -i -n -E "Redundant internal coordinates found in file.|No Z-Matrix found in chk file; cartesian coordinates used." $OUT | cut -f1 -d:)
	beg=$((beg+1))
	en=$(grep -i -n "Recover connectivity data" $OUT | cut -f1 -d:)
	en=$((en-1))
	cart=$(grep "No Z-matrix variables" $OUT)
	if [[ -z "$cart" ]]; then
		sep=","
		col=3
	else
		sep=" "
		col=2
	fi
	awk "NR==$beg, NR==$en" $OUT | awk -F"$sep" -v OFS='\t' '{print $1, $'$col', $'$((col+1))', $'$((col+2))'}'
	}
	
	gauss_vib () {
	count=$(grep -c "Frequencies --" $OUT)
	for ((i=1;i<=count;i++)); do
		for j in {3..5}; do
			grep "Frequencies --" $OUT | sed -n ''$i'p' | awk '{print $'$j'}'
		done
	done
	}

	gauss_zpe () {
	zpe=$(grep "Zero-point correction=" $OUT | awk '{print $3}')
	echo "Zero-Point Vibrational Energy (Hartrees): $zpe"
	}
	
	gauss_rot () {
	#Rotational Constants
	for i in {4..6}; do
		grep -i "Rotational constants" $OUT | tail -n 1 | awk '{print $'$i'}'
	done
	}

	gauss_eng () {
	g_ee=$(find . -maxdepth 2 -type f -print0 -name '*.out' -exec grep "energy=   " {} ';' | awk '{print $4}')
	if [[ -z $g_ee ]]; then
		echo "I couldn't find any Molpro single-point energy calculation for this molecule nearby. Are you looking for the energy from the Gaussian output?"
		select yn in "Yes" "No, I wanted the Molpro SPE"; do
			case $yn in
				"Yes" ) ezpe=$((grep "Sum of electronic and zero-point Energies" $OUT | awk '{print $7}'))
					g_ee=$(echo "scale = 3; $ezpe-$(gauss_zpe | awk '{print $5}')" | bc)
					break;;
				"No, I wanted the Molpro SPE" ) echo "Too bad!"
				        g_ee="thumbsdownemoji.png"
					break;;	
			esac
		done
	else
		echo "Electronic Energy (Hartrees): $g_ee"
		ezpe=$(echo "scale = 3; $g_ee+$(gauss_zpe | awk '{print $5}')" | bc)
		
	fi
	echo "Electronic Energy + Zero-Point Vibrational Energy (Hartrees): $ezpe"
	gkcal=$(echo "scale=2; $ezpe * 627.509" | bc)
	echo "Electronic Energy + Zero-Point Vibrational Energy (kcal mol$^{-1}$): $gkcal"
	}
	
	#gaussian_xyz
	#gauss_vib
	#gauss_zpe
	#gauss_rot
fi

xyztag () {
echo "Optimized Geometry (Cartesian coordinates in \r{A})"
}

xyz () {
if [ $prog == 1 ]; then
	molpro_xyz
elif [ $prog == 2 ]; then
	gaussian_xyz
fi
}

vibtag () {
echo "Vibrational Frequencies (cm$^{-1}$)"
}

vib () {
if [ $prog == 1 ]; then
	vibetc
	molpro_vib
elif [ $prog == 2 ]; then
	gauss_vib
fi
}

zpe () {
if [ $prog == 1 ]; then
	molpro_zpe
elif [ $prog == 2 ]; then
	gauss_zpe
fi
}

rottag () {
	echo "Rotational Constants (GHz)"
}

rot () {
if [ $prog == 1 ]; then
	rotcon
elif [ $prog == 2 ]; then
	gauss_rot
fi
}

eng () {
if [ $prog == 1 ]; then
	molpro_energy
elif [ $prog == 2 ]; then
	gauss_eng
fi
}


echo "Hi, $me. What data are you looking for from this output file?"
select yn in "The optimized geometry" "The vibrational frequencies" "The zero-point and electronic energies" "All rovibrational data" "All of the above"; do
	case $yn in
		"The optimized geometry" ) xyztag
		        xyz	
			break;;
		"The vibrational frequencies" ) vibtag
			vib
			break;;
		"The zero-point and electronic energies" ) eng
			zpe
			break;;
		"All rovibrational data" ) vibtag
			vib
			zpe
			rottag
			rot
			break;;
		"All of the above" ) xyztag
			xyz
			vibtag
			vib
			eng
			zpe
			rottag
			rot
			break;;
	esac
done

echo "Would you like to make an SI table with the data from this output file?"
select yn in "Yes" "No"; do
	case $yn in
		Yes ) "Sure thing $me!"
			break;;
		No ) exit;;
	esac
done

echo "Here you will select the basis set used for the calculation. Please choose the one that was used in the frequency calculation."
select yn in "cc-pVTZ-F12" "cc-pVDZ-F12" "aug-cc-pVTZ" "aug-cc-pVDZ" "Other"; do
	case $yn in
		"cc-pVTZ-F12" ) basis="cc-pVTZ-F12"
	 			break;;		
		"cc-pVDZ-F12" ) basis="cc-pVDZ-F12"	       
	 			break;;		
		"aug-cc-pVTZ" ) basis="aug-cc-pVTZ"	       
	 			break;;		
		"aug-cc-pVDZ" ) basis="aug-cc-pVDZ"	       
	 			break;;
		"Other" ) echo "Please enter the basis set manually."
			read basis 	
			break;;
	esac
done

echo "Here you will select the method used for the calculation. Please choose the one that was used in the frequency calculation."
select yn in "CCSD(T)-F12" "B3LYP" "MP2" "CCSD" "Other"; do
	case $yn in
		"CCSD(T)-F12" ) method="CCSD(T)-F12"
			break;;
		"B3LYP" ) method="B3LYP"
			break;;
		"MP2" ) method="MP2"
			break;;
		"CCSD" ) method="CCSD"
			break;;
		"Other" ) echo "Please enter the method manually."
			read method
			break;;
	esac
done

echo "Here you need to input what you want your molecule to be called. Please input a molecular formula in proper LaTeX formatting or the molecule's name/abbreviation/shorthand name."
read molecule

echo "Is it okay to use what you just input as the table label? Be aware that you can't use the same table label twice in the same document."
select yn in "Yes" "No, I want to choose a different label"; do
	case $yn in
		"Yes" ) label="$molecule"
			break;;
		"No, I want to choose a different label" ) echo "Please input the table label."
			read label
			break;;
	esac
done

theory="${method}""/""${basis}"
caption () {
	echo "\caption{Optimized geometry, vibrational frequencies, rotational constants, and T1 and D1 diagnostic values for "$molecule". Cartesian coordinates in \r{A}.} \label{"$label"}"
}

echo "Do you want to include the T1 and D1 diagnostic values in your SI table? You should, but if you're one of those people that run 100 calculations in the same directory, or string a bunch of subdirectories together, this could get messy..."
select yn in "Yes" "...never mind"; do
	case $yn in
		"Yes" ) diags=1
			break;;
		"...never mind" ) diags=0
			break;;
	esac
done

get_diags () {

#gaussian
if [[ $prog -eq 2 ]]; then
	t1=$(find . -maxdepth 2 -type f -print0 -name '*.out' -exec grep "T1 diagnostic" {} ';' | tail -1 | awk '{print $11}')
	d1=$(find . -maxdepth 2 -type f -print0 -name '*.out' -exec grep "D1 diagnostic" {} ';' | tail -1 | awk '{print $4}')
	if [[ -z $t1 ]]; then
		if [[ "$method" == *"ccsd"* ]]; then
			echo "Unable to locate file for T1 diagnostic retrieval"
		else
			echo "T1 diagnostic not printable for a non-coupled-cluster method"
		fi
	fi
fi


#molpro
if [[ $prog -eq 1 ]]; then
	shopt -s nocasematch
	if [[ "$method" == *"f12"* ]]; then
		t1=$(grep "T1 diagnostic" *.log | tail -1 | awk '{print $11}')
		d1=$(grep "D1 diagnostic" *.log | tail -1 | awk '{print $4}')
	else
		t1=$(grep "T1 diagnostic" *.out | tail -1 | awk '{print $11}')
		d1=$(grep "D1 diagnostic" *.out | tail -1 | awk '{print $4}')
	fi
fi
}

diags () {
	if [[ "$t1" != "0" ]]; then
		echo "T1 Diagnostic & "$t1" & & \\\\"
	fi
	if [[ "$d1" != "0" ]]; then
		echo "D1 Diagnostic & "$d1" & & \\\\"
	fi 
}

freq_format () {
#please=$(echo "$(vib)" | sed -u '/^-/ s/$/$i$/')
header="\multicolumn{4}{c}{"${theory}" Harmonic Frequencies (cm$^{-1}$)} \\\\"
echo "${header}"
viblines=$(wc -l <<< "$(vibs)")
echo $viblines
if (( $viblines % 3 == 0 )); then
	mod=0
elif (( $viblines % 3 != 0 )); then
	viblines=$((viblines-1))
	if (( $viblines % 3 == 0 )); then
		mod=1
		freq1=$(vib | tail -n 2 | head -n 1)
	elif (( $viblines % 3 != 0 )); then
		viblines=$((viblines-1))
		mod=2
		freq2=$(vib | tail -n 1)
	fi
fi

#imags=$(vib | grep -c "-")
#line=$(grep -hn "-" | cut -f1 -d:)

#if [[ $imags != 0 ]]; then
#	for ((i=1;i<=viblines;i++)); do
#		line=$(vib | tail -n$((viblines-i+1)) | head -n 1 | sed -e '/^-/ s/$/$i$/' -e 's/-//g')
#		if [[ $i == 0 ]]; then
			
	
for ((i=1;i<=viblines;i+=3)); do
	echo "& $(vib | tail -n$((viblines-i+1)) | head -n 1) & $(vib | tail -n$((viblines-i)) | head -n 1) & $(vib | tail -n$((viblines-i-1)) | head -n 1) \\\\"
done
if [[ $mod == 1 ]]; then
	echo "& $freq1  & & \\\\"
elif [[ $mod == 2 ]]; then
	echo "& $freq1 & $freq2  & \\\\"
fi

getzpe=$(zpe | awk '{print $5}')
cmzpe=$(echo "scale = 2; $getzpe * 219474.63" | bc)
echo "\multicolumn{4}{c}{Zero-Point Vibrational Energy (cm$^{-1}$): $getzpe} \\\\"
}

xyz_format () {
xyzlines=$(wc -l <<< "$(xyz)")
for ((i=1;i<xyzlines;i++)); do
	line=$(xyz | tail -n$((xyzlines-i+1)) | head -n 1)
	echo "$(awk '{print $1}' <<< $line) & $(awk '{print $2}' <<< $line) & $(awk '{print $3}' <<< $line) & $(awk '{print $4}' <<< $line) \\\\"
done
}

rot_format () {
rottag
echo "& $(rot | head -n 1) & $(rot | tail -n2 | head -n 1) & $(rot | tail -n1) \\\\"
}

echo "\begin{table}[ht!]"
echo "\centering"
echo "\begin{tabular}{crrr}"
echo "\hline"
echo "Cartesian Coordinates (\r{A})"
echo "\hline"
xyz_format
echo "\hline"
get_diags
diags
echo "\hline"
freq_format
echo "\hline"
rot_format
echo "\hline"
echo "\end{tabular}"
caption
echo "\end{table}"
	

