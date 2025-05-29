#!/bin/bash

# Enhanced wait_and_shutdown.sh - Monitor processes and shutdown when they complete
# Avoids shutdown when processes are manually killed (Ctrl+C, kill commands)

LOG_FILE="./wait_and_shutdown.log"
PID_FILE="./wait_and_shutdown.pid"
SLEEP_INTERVAL=10 # seconds to wait between checks

# Show help information
show_help() {
    cat << EOF
wait_and_shutdown.sh - Monitor processes and shutdown system when they complete naturally

USAGE:
    $0 [OPTION]

OPTIONS:
    (no args)       Interactive mode: select a process to monitor, then start background monitoring
    --help, -h      Show this help message
    --status        Show current monitoring status
    --cancel        Cancel any active monitoring and remove background job
    --monitor PID   Internal mode: monitor specific PID (used by background process)

EXAMPLES:
    # Start monitoring (interactive)
    $0

    # Check if monitoring is active
    $0 --status

    # Cancel active monitoring
    $0 --cancel

    # Show help
    $0 --help

BEHAVIOR:
    - When a monitored process ends naturally (completion/crash): System will shutdown
    - When a monitored process is manually killed (Ctrl+C, kill): System stays running
    - All activity is logged to: $LOG_FILE
    - Background monitor PID is tracked in: $PID_FILE

FILES:
    $LOG_FILE    - Activity log with timestamps
    $PID_FILE    - Contains PID of background monitor process (when active)

EOF
}

# Show current monitoring status
show_status() {
    if [ -f "$PID_FILE" ]; then
        MONITOR_PID=$(cat "$PID_FILE")
        if ps -p "$MONITOR_PID" > /dev/null 2>&1; then
            # Get the target PID being monitored
            TARGET_PID=$(ps -p "$MONITOR_PID" -o args= | grep -o -- '--monitor [0-9]*' | awk '{print $2}')
            if [ -n "$TARGET_PID" ]; then
                if ps -p "$TARGET_PID" > /dev/null 2>&1; then
                    TARGET_INFO=$(ps -p "$TARGET_PID" -o pid,comm,args= | tail -1)
                    echo "✓ Monitoring is ACTIVE"
                    echo "  Monitor PID: $MONITOR_PID"
                    echo "  Target PID:  $TARGET_PID"
                    echo "  Target process: $TARGET_INFO"
                    echo "  Log file: $LOG_FILE"
                    if [ -f "$LOG_FILE" ]; then
                        echo "  Last log entry: $(tail -1 "$LOG_FILE")"
                    fi
                else
                    echo "⚠ Monitor is running (PID $MONITOR_PID) but target process $TARGET_PID has ended"
                    echo "  The monitor should terminate soon..."
                fi
            else
                echo "⚠ Monitor is running (PID $MONITOR_PID) but cannot determine target PID"
            fi
        else
            echo "✗ Monitoring is INACTIVE (stale PID file found)"
            echo "  Cleaning up stale PID file: $PID_FILE"
            rm -f "$PID_FILE"
        fi
    else
        echo "✗ Monitoring is INACTIVE (no PID file found)"
    fi
}

# Cancel active monitoring
cancel_monitoring() {
    if [ -f "$PID_FILE" ]; then
        MONITOR_PID=$(cat "$PID_FILE")
        if ps -p "$MONITOR_PID" > /dev/null 2>&1; then
            echo "Cancelling monitoring (PID $MONITOR_PID)..."
            kill "$MONITOR_PID"
            sleep 1
            if ps -p "$MONITOR_PID" > /dev/null 2>&1; then
                echo "Process didn't respond to SIGTERM, using SIGKILL..."
                kill -9 "$MONITOR_PID"
                sleep 1
            fi
            if ! ps -p "$MONITOR_PID" > /dev/null 2>&1; then
                echo "✓ Monitoring cancelled successfully"
                rm -f "$PID_FILE"
                echo "[$(date +"%Y-%m-%d %H:%M:%S")] Monitoring cancelled by user" >> "$LOG_FILE"
            else
                echo "✗ Failed to cancel monitoring process"
                exit 1
            fi
        else
            echo "✗ No active monitoring found (stale PID file)"
            rm -f "$PID_FILE"
        fi
    else
        echo "✗ No active monitoring found (no PID file)"
    fi
}

# Parse command line arguments
case "$1" in
    --help|-h)
        show_help
        exit 0
        ;;
    --status)
        show_status
        exit 0
        ;;
    --cancel)
        cancel_monitoring
        exit 0
        ;;
    --monitor)
        # Monitor mode - continue to existing logic
        ;;
    "")
        # Interactive mode - continue to existing logic
        ;;
    *)
        echo "Error: Unknown option '$1'"
        echo "Use --help for usage information"
        exit 1
        ;;
esac

# Determine mode
if [ "$1" != "--monitor" ]; then
    # Interactive selection mode

    # Check if monitoring is already active
    if [ -f "$PID_FILE" ]; then
        EXISTING_PID=$(cat "$PID_FILE")
        if ps -p "$EXISTING_PID" > /dev/null 2>&1; then
            echo "⚠ Monitoring is already active (PID $EXISTING_PID)"
            echo "Use '$0 --status' to see details or '$0 --cancel' to stop it first"
            exit 1
        else
            echo "Cleaning up stale PID file..."
            rm -f "$PID_FILE"
        fi
    fi

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
    MONITOR_PID=$!
    echo "$MONITOR_PID" > "$PID_FILE"
    echo "✓ Background monitoring started (Monitor PID: $MONITOR_PID)"
    echo "  Use '$0 --status' to check status"
    echo "  Use '$0 --cancel' to cancel monitoring"
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

# Cleanup function to remove PID file on exit
cleanup_on_exit() {
    log_message "Monitor process exiting, cleaning up PID file"
    rm -f "$PID_FILE"
}

# Set up cleanup trap for various exit conditions
trap cleanup_on_exit EXIT INT TERM

# Verify process exists
if ! ps -p "$PROCESS_PID" > /dev/null; then
    log_message "Error: Process with PID $PROCESS_PID does not exist. Exiting."
    exit 1
fi

log_message "PID $PROCESS_PID found. Monitoring..."

ITERATIONS_PER_HOUR=$((3600 / SLEEP_INTERVAL))
check_count=0

# Monitor the process and capture its exit status
while ps -p "$PROCESS_PID" > /dev/null; do
    sleep "$SLEEP_INTERVAL"
    check_count=$((check_count + 1))
    if [ $((check_count % ITERATIONS_PER_HOUR)) -eq 0 ]; then
        log_message "PID $PROCESS_PID is still running. Check count: $check_count"
    fi
done

log_message "Process $PROCESS_PID finished."

# Try to get the exit status of the process
# Note: This won't work for processes we didn't start, so we'll use a different approach
# Check if the process was killed by examining system logs or using alternative detection

# Alternative approach: Check if any kill signals were sent recently
# We'll look for manual termination signals (SIGTERM=15, SIGINT=2, SIGKILL=9)
MANUAL_KILL_DETECTED=false

# Check recent system logs for kill signals to our PID (last 1 minute)
if command -v journalctl > /dev/null 2>&1; then
    # Check for kill signals in systemd journal
    if journalctl --since "1 minute ago" --no-pager -q | grep -q "signal.*$PROCESS_PID"; then
        log_message "Detected potential manual termination signal in system logs"
        MANUAL_KILL_DETECTED=true
    fi
fi

# Additional check: Look for recent kill commands in bash history targeting our PID
# This is a heuristic approach since we can't directly get exit status of unrelated processes
if [ -f ~/.bash_history ]; then
    # Check last few commands for kill targeting our PID
    if tail -20 ~/.bash_history 2>/dev/null | grep -q "kill.*$PROCESS_PID"; then
        log_message "Detected recent kill command for PID $PROCESS_PID in bash history"
        MANUAL_KILL_DETECTED=true
    fi
fi

# Check if the process directory still exists in /proc (it disappears after natural exit)
# and if parent process is still around (manual kills often leave parent running)
if [ -d "/proc/$PROCESS_PID" ]; then
    log_message "Process directory still exists, indicating recent termination"
    # This is inconclusive, so we won't use it to determine manual kill
fi

if [ "$MANUAL_KILL_DETECTED" = true ]; then
    log_message "Manual termination detected. Skipping shutdown to preserve user session."
    log_message "System will remain running as user appears to be actively using it."
    exit 0
else
    log_message "Process appears to have ended naturally (completion or crash)."
    log_message "Proceeding with planned shutdown sequence."
    log_message "Syncing disks..."
    sync
    log_message "Disk sync complete."
    log_message "Shutting down the VM now."

    sudo shutdown now
fi