#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment
#SBATCH --partition=bigmem
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=latentcor_py_ipol            #Set the job name to
#SBATCH --time=01:00:00               #Set the wall clock limit to 6hr and 30min
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=80         #Request 8 tasks/cores per node
#SBATCH --mem=320GB                     #Request 8GB per node
#SBATCH --output=latentcor_py_ipol.%j            #Send stdout/err to "Example2Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=ALL                     #Send email on all job events
#SBATCH --mail-user=sharkmanhmz@tamu.edu    #Send all emails

#First Executable Line
module load Anaconda3/2021.05

python ipol_10.py 
python ipol_11.py
python ipol_20.py
python ipol_21.py
python ipol_22.py
python ipol_30.py
