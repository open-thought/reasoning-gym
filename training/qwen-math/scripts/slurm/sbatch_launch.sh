#!/bin/bash
# use by running `bash sbatch_launch.sh <script.slurm>`

cleanup() {
    echo "Script interrupted. Cleaning up..."
    scancel "$job_id" 2>/dev/null
    echo "Job $job_id has been canceled."
    exit 1
}
trap cleanup SIGINT

# launch the slurm script
SLURM_FILE=$1
echo "Launching $SLURM_FILE ..."
job_id=$(sbatch $SLURM_FILE | awk '{print $4}')
echo "Submitted job with ID: $job_id"

# Wait until the job is running
while true; do
    job_status=$(squeue -j "$job_id" -h -o "%T")
    if [ "$job_status" == "RUNNING" ]; then
        echo "Job $job_id is now running."
        sleep 5
        break
    elif [ -z "$job_status" ]; then
        echo "Job $job_id has finished or failed before reaching running state."
        exit 1
    else
        echo "Job $job_id is still in $job_status state. Checking again in 10 seconds..."
        sleep 10
    fi
done

# Plot the real-time output
output_file=$(scontrol show job "$job_id" | awk -F= '/StdOut/ {print $2}' | sed "s/%A/${job_id}/g" | sed "s/%a/1/g")
echo "Tailing output file: $output_file"
tail -f "$output_file"