#!/bin/bash

OUTPUT_FILE="runtime_results.txt"

echo "Loop Value, Execution Time (s)" > $OUTPUT_FILE

# loop through values 50 to 230 with a step of 10
for i in $(seq 100 10 230); do
    echo "Running script with value: $i"
    START_TIME=$(date +%s.%N)
    
    python3 main.py --gender "test_of_speed" --quantity "$i"
    
    END_TIME=$(date +%s.%N)
    
    # Calculate elapsed time
    EXECUTION_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    
    # Save to output file
    echo "$i, $EXECUTION_TIME" >> $OUTPUT_FILE
    
    echo "Completed iteration for $i in $EXECUTION_TIME seconds"
done

echo "All runs completed. Results saved to $OUTPUT_FILE"