#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment
#SBATCH --partition=knl

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=ipol_31            #Set the job name to
#SBATCH --time=2-02:00:00               #Set the wall clock limit to 6hr and 30min
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=72         #Request 8 tasks/cores per node
#SBATCH --mem=80GB                     #Request 8GB per node
#SBATCH --output=ipol_31.%j            #Send stdout/err to "Example2Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=ALL                     #Send email on all job events
#SBATCH --mail-user=sharkmanhmz@tamu.edu    #Send all emails

#First Executable Line
module load Anaconda3/2021.05

python ipol_31.py 
