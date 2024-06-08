#!/bin/bash

# Check if the number of arguments is less than 2
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_file> <output_directory>"
    exit 1
fi

# Assign input arguments to variables
INPUT_FILE=$1
OUTPUT_DIR=$2

# Ensure the output directory exists
mkdir -p $OUTPUT_DIR

# Define quantization types
QUANT_TYPES=("Q2_K" "Q4_K_M" "Q5_K_M" "Q3_K_M" "Q8_0" "Q6_K")

# Loop through each quantization type and run the quantization command
for TYPE in "${QUANT_TYPES[@]}"; do
    OUTPUT_FILE="$OUTPUT_DIR/$(basename ${INPUT_FILE%.f32.gguf}).${TYPE}.gguf"
    echo "Running quantization with type $TYPE..."
    ./quantize $INPUT_FILE $OUTPUT_FILE $TYPE
    
    if [ $? -eq 0 ]; then
        echo "Quantization with type $TYPE completed successfully!"
    else
        echo "Quantization with type $TYPE failed!"
    fi
done

