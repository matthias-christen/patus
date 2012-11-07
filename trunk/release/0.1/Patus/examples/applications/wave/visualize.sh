#!/bin/bash
echo "set zrange [-1:1]" > waves.gp
echo "set cbrange [-1:1]" >> waves.gp
echo "set pm3d" >> waves.gp
for f in ????.txt
do
	echo "splot \"$f\" matrix with pm3d" >> waves.gp
done
gnuplot waves.gp
