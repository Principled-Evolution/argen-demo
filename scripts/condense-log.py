import sys
import re
import os

def is_metric_line(line):
    """Checks if a line contains metric or reward information."""
    # Look for patterns like 'key': value or metrics/ rewards paths
    return re.search(r"'[^']+'\s*:\s*[^,\}]+", line) or \
           re.search(r"metrics\/|rewards\/", line) or \
           re.search(r"loss|grad_norm|learning_rate|kl|clip_ratio|num_tokens|completions\/|reward", line)

def get_comparable_message_part(line_text):
    """
    Extracts the part of a log line to be used for comparison, ignoring the timestamp.
    This includes the module, log level, and message.
    """
    # Regex to capture everything after the initial timestamp and ' - '
    # Example match.group(1): 'src.reward_functions.openai_rewards - INFO - Evaluating Ahimsa with OpenAI...'
    match = re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - (.*)", line_text)
    if match:
        return match.group(1).strip()  # Return everything after the timestamp and strip whitespace
    return line_text.strip()  # Fallback: return the whole line and strip whitespace if it doesn't match the pattern

def is_error_warning_line(line):
    """Checks if a line is an error or warning."""
    return re.search(r"ERROR|WARNING", line, re.IGNORECASE) is not None

def is_wandb_line(line):
    """Checks if a line is a wandb message."""
    return line.strip().startswith("wandb:")

def is_progress_bar_line(line):
    """Checks if a line is a progress bar."""
    # Matches lines like "  0%|          | 0/24000 [00:00<?, ?it/s]"
    return re.match(r"^\s*\d+%\|.*\[.*it\/s.*\]", line) is not None

def process_log_file(input_filepath):
    """Reads a log file and writes a condensed version."""
    output_filepath = input_filepath.replace('.log', '-condensed.log')
    if output_filepath == input_filepath:
         output_filepath = input_filepath + '-condensed'


    with open(input_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
        current_info_block = []
        current_progress_block = []
        # Stores the comparable message part of the first line in current_info_block
        block_representative_message = None

        line_num = 0 # For debugging

        def flush_info_block():
            if current_info_block:
                print(f"DEBUG: Flushing info block. Size: {len(current_info_block)}") # Debug print
                if len(current_info_block) > 1:
                    # Condense block
                    first_line = current_info_block[0]
                    # Attempt to extract timestamp and message part
                    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - [A-Za-z0-9\._]+ - (INFO|DEBUG|CRITICAL)) - (.*)", first_line)
                    if match:
                        timestamp_part = match.group(1)
                        message_part = match.group(3)
                        outfile.write(f"{timestamp_part} - Repeated {len(current_info_block)} times: {message_part}\n")
                    else:
                         # Fallback if regex doesn't match expected format
                         # Use get_comparable_message_part for fallback consistency
                         fallback_message = get_comparable_message_part(current_info_block[0])
                         outfile.write(f"Repeated {len(current_info_block)} times: {fallback_message}\n")
                else:
                    # Single line, just write it
                    print("DEBUG: Info block size 1, writing single line.") # Debug print
                    outfile.write(current_info_block[0] + '\n')
                current_info_block.clear()
                print("DEBUG: Info block cleared.") # Debug print

        def flush_progress_block():
            if current_progress_block:
                # Only write the last line of the progress block
                outfile.write(current_progress_block[-1] + '\n')
                current_progress_block.clear()

        for line in infile:
            line = line.rstrip() # Remove trailing newline
            line_num += 1 # For debugging
            print(f"DEBUG: Processing line {line_num}: {line}") # Debug print

            if is_metric_line(line) or is_error_warning_line(line) or is_wandb_line(line):
                # These lines break any ongoing blocks and are kept
                print(f"DEBUG: Line {line_num} is salient (metric/error/warning/wandb).") # Debug print
                flush_info_block()
                block_representative_message = None # Reset representative message
                flush_progress_block()
                outfile.write(line + '\n')
            elif is_progress_bar_line(line):
                # Progress bars
                print(f"DEBUG: Line {line_num} is a progress bar.") # Debug print
                flush_info_block() # Progress bars break info blocks
                block_representative_message = None # Reset representative message
                current_progress_block.append(line) # Add to progress block, will keep only last later
            else:
                # Potential standard info line
                print(f"DEBUG: Line {line_num} is a potential info line.") # Debug print
                flush_progress_block() # Info lines break progress blocks
                
                current_line_comparable_part = get_comparable_message_part(line)
                print(f"DEBUG: Comparable part for line {line_num}: {current_line_comparable_part}") # Debug print

                if not current_info_block:
                    # Start a new info block
                    print(f"DEBUG: Starting new info block with line {line_num}.") # Debug print
                    current_info_block.append(line)
                    block_representative_message = current_line_comparable_part
                    print(f"DEBUG: Representative message set to: {block_representative_message}") # Debug print
                elif current_line_comparable_part == block_representative_message:
                    # Current line's message content matches the block's representative message
                    print(f"DEBUG: Line {line_num} matches representative message.") # Debug print
                    current_info_block.append(line)
                else:
                    # Line is different, flush current block and start a new one
                    print(f"DEBUG: Line {line_num} does NOT match representative message. Flushing block.") # Debug print
                    flush_info_block()
                    current_info_block.append(line)
                    block_representative_message = current_line_comparable_part
                    print(f"DEBUG: Starting new info block with line {line_num}. Representative message set to: {block_representative_message}") # Debug print

        # Flush any remaining blocks at the end of the file
        print("DEBUG: End of file reached. Flushing remaining blocks.") # Debug print
        flush_info_block()
        flush_progress_block()

    print(f"Condensed log written to: {output_filepath}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <input_log_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print(f"Error: File not found at '{input_file}'")
        sys.exit(1)

    process_log_file(input_file)
