#!/bin/bash
#SBATCH -n 64
#SBATCH -N 2
#SBATCH -t 12:00:00
#SBATCH -p Long
#SBATCH -J SabbirJob
#SBATCH -o is_cool.out
#SBATCH -e sabbir_is_notcool.err
#for ((i = 1; i < 2; i+=256))
#do
#	(( last_= (($i - 1) + 256 )))
	#echo $last_
mpirun python data_collection.py 4096 6092 32
#done
#mpiexec --mca opal_set_max_sys_limits 64 -np 60 python graph_based_data_formatting.py 
