#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment
#SBATCH --partition=knl

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=ipol_10            #Set the job name to
#SBATCH --time=01:00:00               #Set the wall clock limit to 6hr and 30min
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=72         #Request 8 tasks/cores per node
#SBATCH --mem=80GB                     #Request 8GB per node
#SBATCH --output=ipol_10.%j            #Send stdout/err to "Example2Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=ALL                     #Send email on all job events
#SBATCH --mail-user=sharkmanhmz@tamu.edu    #Send all emails

#First Executable Line
module load Python/3.9.6-GCCcore-11.2.0
python ipol_10.py 2> ipol_10_out
