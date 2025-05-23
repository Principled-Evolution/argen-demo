#!/bin/bash

# Usage:
#  1) ./wait_and_shutdown.sh
#     - Prompts for process pattern, lets you select and confirm
#     - Then backgrounds monitoring via nohup and exits
#  2) ./wait_and_shutdown.sh --monitor PID
#     - Internal mode: monitors the given PID, syncs, and shuts down

LOG_FILE="./wait_and_shutdown.log"
SLEEP_INTERVAL=10 # seconds to wait between checks

# Determine mode
if [ "$1" != "--monitor" ]; then
    # Interactive selection mode
    DEFAULT_PATTERN="accelerate launch"
    read -p "Enter process search pattern [${DEFAULT_PATTERN}]: " PATTERN
    PATTERN=${PATTERN:-$DEFAULT_PATTERN}

    # Find matching processes
    mapfile -t lines < <(pgrep -af -- "$PATTERN")

    if [ ${#lines[@]} -eq 0 ]; then
        echo "No process matching '$PATTERN' found."
        exit 1
    elif [ ${#lines[@]} -gt 1 ]; then
        echo "Multiple processes found matching '$PATTERN':"
        for i in "${!lines[@]}"; do
            echo "$i) ${lines[$i]}"
        done
        read -p "Enter the number of the process you want to monitor: " choice
        selected_line="${lines[$choice]}"
    else
        selected_line="${lines[0]}"
    fi

    PROCESS_PID=$(echo "$selected_line" | awk '{print $1}')
    START_INFO=$(ps -p "$PROCESS_PID" -o lstart=,args=)
    echo "Selected process: $START_INFO"
    read -p "Is this the process you want to monitor? (y/n) " answer
    case "$answer" in
        [Yy]*) ;; 
        *) echo "Aborting."; exit 1 ;;
    esac

    echo "Starting background monitor for PID $PROCESS_PID. Logging to $LOG_FILE."
    nohup "$0" --monitor "$PROCESS_PID" > "$LOG_FILE" 2>&1 &
    exit 0
fi

# Monitor mode
PROCESS_PID="$2"
if [ -z "$PROCESS_PID" ]; then
    echo "No PID provided for monitor mode."
    exit 1
fi

# Logging function: append timestamped messages to log
log_message() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" >> "$LOG_FILE"
}

log_message "Starting monitoring script for PID $PROCESS_PID"

# Verify process exists
if ! ps -p "$PROCESS_PID" > /dev/null; then
    log_message "Error: Process with PID $PROCESS_PID does not exist. Exiting."
    exit 1
fi

log_message "PID $PROCESS_PID found. Monitoring..."

ITERATIONS_PER_HOUR=$((3600 / SLEEP_INTERVAL))
check_count=0

while ps -p "$PROCESS_PID" > /dev/null; do
    sleep "$SLEEP_INTERVAL"
    check_count=$((check_count + 1))
    if [ $((check_count % ITERATIONS_PER_HOUR)) -eq 0 ]; then
        log_message "PID $PROCESS_PID is still running. Check count: $check_count"
    fi
done

log_message "Process $PROCESS_PID finished."
log_message "Syncing disks..."
sync
log_message "Disk sync complete."
log_message "Shutting down the VM now."

sudo shutdown now