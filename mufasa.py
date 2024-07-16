import os
import subprocess

def detect_aval_cpus():
    """
    Detects the number of available CPUs.
    """
    try:
        currentjobid = os.environ["SLURM_JOB_ID"]
        currentjobid = int(currentjobid)
        command = f"squeue --Format=JobID,cpus-per-task | grep {currentjobid}"
        # Run the command as a subprocess and capture the output
        output = subprocess.check_output(command, shell=True)[5:-4].replace(b" ", b"")
        cpus = output.decode("utf-8")
        cpus2 = len(os.sched_getaffinity(0))
        cpus = min(int(cpus), cpus2)
    except:
        cpus = os.cpu_count()
    return cpus