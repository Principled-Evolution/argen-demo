#!/bin/bash
# Script to run all debug tests

# Create debug logs directory
mkdir -p debug_logs

# Set up logging
LOG_FILE="debug_logs/debug_tests.log"
echo "Starting debug tests at $(date)" > $LOG_FILE

# Function to log messages
log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1" | tee -a $LOG_FILE
}

# Function to run a test and log the result
run_test() {
    TEST_NAME=$1
    TEST_CMD=$2
    
    log "Running test: $TEST_NAME"
    log "Command: $TEST_CMD"
    
    # Run the test and capture output
    OUTPUT_FILE="debug_logs/${TEST_NAME}_output.log"
    if eval "$TEST_CMD" > "$OUTPUT_FILE" 2>&1; then
        log "Test $TEST_NAME completed successfully"
    else
        log "Test $TEST_NAME failed with exit code $?"
    fi
    
    log "Output saved to $OUTPUT_FILE"
    log "----------------------------------------"
}

# Test 1: Chat template test
log "Starting Chat Template Test"
run_test "chat_template" "python test_chat_template.py"

# Test 2: Compare TRL versions
log "Starting TRL Version Comparison"
run_test "trl_comparison" "python compare_trl_versions.py"

# Test 3: Patch TRL and run debug script
log "Patching TRL and running debug script"
run_test "trl_patch" "python trl_debug_patch.py"
run_test "debug_grpo" "python debug_grpo_generation.py"

# Test 4: Run with original train_grpo.py using a small dataset
log "Running minimal test with train_grpo.py"
run_test "train_grpo_minimal" "python examples/train_grpo.py --model meta-llama/Llama-3.2-1B-Instruct --scenarios data/debug-data-set/debug-garbled-chat-template-response.jsonl --output_dir debug_grpo_output --max_steps 5 --logging_steps 1 --save_steps 5 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 --group_size 2"

log "All debug tests completed"
echo "Debug tests completed at $(date)" >> $LOG_FILE
