import argparse
import os
import subprocess
import math
import time
import getpass
import secrets
import stat

parser = argparse.ArgumentParser(description="Dispatch a list of python jobs from a given file to a PBS cluster")
parser.add_argument("file", type=argparse.FileType(),
                    help="path to python commands file to run")
parser.add_argument("--conda", default="pymarl", type=str, help="Name of the targeted anaconda environment")
parser.add_argument("--email", default="kcorder@udel.edu", type=str)
parser.add_argument("--queue", default="standard", type=str, choices=["standard","debug","interactive"])
parser.add_argument("--name", default=None, type=str, help="Name of the job. Default: file name")
parser.add_argument("--gpus", default=0, type=int, help="Requested GPUs PER job")
parser.add_argument("--jobs-per-node", default=8, type=int, help="how many runs can fit on a single node")
parser.add_argument("--timelimit", default=72, type=int, help="Requested hour limit PER job")
# parser.add_argument("--exclude", default=None, type=str, help="Exclude malfunctioning nodes. Should be a node name.")
parser.add_argument(
    "--modules", default=[], nargs="+", type=str, help="An optional list of strings of module names to be loaded"
)
parser.add_argument("--cleanup-temp-files", default=False, action='store_true',
                    help="remove temporary files created by this script after submitting jobs")
args = parser.parse_args()


# Parse and validate input:
if args.name is None:
    dispatch_name = args.file.name
else:
    dispatch_name = args.name

# 1) Strip the file_list of comments and blank lines
content = args.file.readlines()
jobs = [c.strip().split("#", 1)[0] for c in content if c[0] != "#"]  # Also allow non-python launches
jobs = [job for job in jobs if len(job) - job.count(" ") > 0]  # Filter empty lines


print(f"Detected {len(jobs)} jobs.")
if len(jobs) < 1:
    raise ValueError("Detected no valid jobs in given file!")

# Write the clean file list
authkey = secrets.token_urlsafe(5)
job_list_filename = f".qsub_job_list_{authkey}.temp.sh"
with open(job_list_filename, "w") as file:
    file.writelines(chr(10).join(job for job in jobs))
    file.write("\n")


# Write the aprun line script (may already exist, doesn't change):
aprun_line_filename = ".aprun_line.sh"
if not os.path.exists(aprun_line_filename) or not os.path.isfile(aprun_line_filename):
    aprun_line_contents = """#!/bin/sh 
    
cmds_file=$1 
line_num=$ALPS_APP_PE
line_num=$((line_num+1)) # ALPS_APP_PE is 0-indexed; add 1 for line num 
echo $line_num 

line=`head -$line_num $cmds_file | tail -1` 
echo $line 
eval "$line" 

wait 
"""
    with open(aprun_line_filename, "w") as fd:
        fd.write(aprun_line_contents)
    file_mode = os.stat(aprun_line_filename).st_mode
    os.chmod(aprun_line_filename, file_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)  # chmod a+x


# 3) Prepare environment
if not os.path.exists("log"):
    os.makedirs("log")

# 4) Construct the sbatch launch file
MAX_ARRAY_SIZE = 500
num_template_to_launch = 1 + len(jobs) // MAX_ARRAY_SIZE

num_nodes = math.ceil(len(jobs) / args.jobs_per_node)


def _prototype(template_idx):
    if num_template_to_launch > 1:
        job_name = f"{''.join(e for e in dispatch_name if e.isalnum())}_{template_idx}"
    else:
        job_name = "".join(e for e in dispatch_name if e.isalnum())
    SBATCH_PROTOTYPE = f"""#!/bin/bash
#PBS -A ARLAP00581830 
#PBS -q {args.queue}
#PBS -N {job_name}
#PBS -j oe 
#PBS -M {args.email}
#PBS -m abe 
#PBS -l walltime={args.timelimit}:00:00

#PBS -l select={num_nodes}:{"ncpus=22:mpiprocs=1:ngpus="+str(args.gpus) if args.gpus else "ncpus=44:mpiprocs=1"}

source $HOME/anaconda3/bin/activate {args.conda}
{f"module load {' '.join(args.modules)}" if len(args.modules) > 0 else ""}

cd $PBS_O_WORKDIR  # where file submitted (current directory) 
echo "Executing from directory: $PWD"  
aprun -n {len(jobs)} -N {min(len(jobs), args.jobs_per_node)} $PWD/{aprun_line_filename} $PWD/{job_list_filename}
mv {job_name}.o$PBS_JOBID log/
"""
    return SBATCH_PROTOTYPE




# 5) Write launch commands to file(s) and print info
qsub_launch_filename = lambda idx: f".qsub_launch_{authkey}.{idx}.temp.sh"
print("Launch prototype is ...")
for idx in range(num_template_to_launch):
    SBATCH_PROTOTYPE = _prototype(idx)
    print("---------------")
    print(SBATCH_PROTOTYPE)
    print("---------------")

    with open(qsub_launch_filename(idx), "w") as file:
        file.write(SBATCH_PROTOTYPE)

print("---------------")
print("Valid jobs will be ...")
print("\n".join("qsub " + job for job in jobs))
print("---------------")

print(f"Preparing {len(jobs)} jobs as user {getpass.getuser()}" f" for launch in 10 seconds...")
print("Terminate if necessary ...")
for _ in range(10):
    time.sleep(1)

# 6) Launch

# Execute file(s) with sbatch
for idx in range(num_template_to_launch):
    subprocess.run(["qsub", qsub_launch_filename(idx)])
print("Subprocess launched ...")
time.sleep(3)


# cleanup:
if args.cleanup_temp_files:
    subprocess.run(["rm", "-f", job_list_filename])
    subprocess.run(["rm", "-f", aprun_line_filename])
    for idx in range(num_template_to_launch):
        subprocess.run(["rm", "-f", qsub_launch_filename(idx)])


if __name__ == '__main__':
    pass
