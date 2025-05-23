#!/bin/bash

LOG_FILE="system_utilization_log_$(date +%Y%m%d_%H%M%S).csv"
DURATION_MINUTES=6
INTERVAL_SECONDS=10
ITERATIONS=$(( (DURATION_MINUTES * 60) / INTERVAL_SECONDS ))

# Check for nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi command not found. Please ensure NVIDIA drivers are installed and nvidia-smi is in your PATH."
    exit 1
fi

# Check for vmstat
if ! command -v vmstat &> /dev/null; then
    echo "Error: vmstat command not found. Please install procps (or equivalent package for vmstat) and ensure vmstat is in your PATH."
    exit 1
fi

# Determine number of GPUs and construct header
# Ensure nvidia-smi output is parsed correctly even if it has extra spaces or lines.
NUM_GPUS_RAW=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)
NUM_GPUS=$(echo "$NUM_GPUS_RAW" | awk '{print $1}' | head -n 1) # Take first field of first line

HEADER_FIELDS="Timestamp,CPU_Util_%,SysMem_Used_MiB,SysMem_Total_MiB"
if [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] && [ "$NUM_GPUS" -gt 0 ]; then
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        HEADER_FIELDS="${HEADER_FIELDS},GPU${i}_Util_%,GPU${i}_Mem_Used_MiB,GPU${i}_Mem_Total_MiB"
    done
else
    # If NUM_GPUS is not a positive integer, treat as 0.
    # This can happen if nvidia-smi --query-gpu=count returns unexpected output or no GPUs are found.
    echo "Warning: Could not reliably determine GPU count or no GPUs found by nvidia-smi. GPU utilization will not be logged or header might be incomplete for GPUs."
    NUM_GPUS=0
fi
echo "$HEADER_FIELDS" > "$LOG_FILE"

echo "Logging to $LOG_FILE for $DURATION_MINUTES minutes at $INTERVAL_SECONDS second intervals..."
echo "This will take approximately $DURATION_MINUTES minutes. Press Ctrl+C to stop logging early."

# Trap Ctrl+C for graceful exit
trap "echo; echo 'Logging interrupted by user. Log file: $LOG_FILE'; exit" SIGINT

for i in $(seq 1 $ITERATIONS); do
    CURRENT_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # CPU Utilization (100 - idle percentage from vmstat)
    # vmstat 1 2: samples for 1 second, outputs 2 lines (avg since boot, then current interval)
    # We take the second line (current interval) and column 15 (idle 'id' field).
    CPU_IDLE=$(vmstat 1 2 | tail -n1 | awk '{print $15}')
    CPU_UTIL="Error" # Default in case of issues
    if [[ "$CPU_IDLE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then # Check if CPU_IDLE is a number
        CPU_UTIL=$(awk -v idle="$CPU_IDLE" 'BEGIN { printf "%.2f", 100 - idle }')
    else
        echo "Warning: Failed to parse CPU idle value ($CPU_IDLE) at $(date '+%Y-%m-%d %H:%M:%S')"
    fi

    # System Memory Utilization (from free -m)
    SYS_MEM_INFO=$(free -m | grep '^Mem:')
    SYS_MEM_TOTAL="Error"
    SYS_MEM_USED="Error"
    if [[ -n "$SYS_MEM_INFO" ]]; then
        SYS_MEM_TOTAL=$(echo "$SYS_MEM_INFO" | awk '{print $2}')
        SYS_MEM_USED=$(echo "$SYS_MEM_INFO" | awk '{print $3}')
    else
        echo "Warning: Failed to parse system memory info at $(date '+%Y-%m-%d %H:%M:%S')"
    fi

    LOG_LINE="${CURRENT_TIMESTAMP},${CPU_UTIL},${SYS_MEM_USED},${SYS_MEM_TOTAL}"

    # GPU Utilization and Memory
    if [ "$NUM_GPUS" -gt 0 ]; then
        # Get utilization and memory for all GPUs, one line per GPU: util.gpu,memory.used,memory.total
        # Example line: 30, 1024, 40960
        GPU_DATA_RAW=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)

        if [ -n "$GPU_DATA_RAW" ]; then
            # Process each GPU's data line
            # Use a loop to handle multi-line output from nvidia-smi if NUM_GPUS > 1
            # Convert to string of comma separated values
            GPU_DATA_FLAT=$(echo "$GPU_DATA_RAW" | awk '{gsub(/ MiB/, ""); printf "%s,", $0}' | sed 's/,$//')
            LOG_LINE="${LOG_LINE},${GPU_DATA_FLAT}"
        else
            echo "Warning: Failed to get GPU utilization/memory data at $(date '+%Y-%m-%d %H:%M:%S')"
            # Add empty placeholders if GPU data couldn't be fetched but header expects them
            for j in $(seq 0 $((NUM_GPUS - 1))); do
                LOG_LINE="${LOG_LINE},Error,Error,Error" # For util, mem.used, mem.total
            done
        fi
    fi

    echo "$LOG_LINE" >> "$LOG_FILE"

    # Sleep until the next interval, but not for the last iteration
    if [ "$i" -lt "$ITERATIONS" ]; then
        sleep "$INTERVAL_SECONDS"
    fi
done

echo "Logging complete. Log file: $LOG_FILE" 