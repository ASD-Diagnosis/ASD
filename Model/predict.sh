#!/bin/bash

# Set parameters
START_ID=50000
END_ID=50070
MODEL_NAME="rf"
DATA_DIR="./data/cc200_data"  # Adjust path for your system

# Iterate through subject IDs and run predict.py
for SUBJECT_ID in $(seq $START_ID $END_ID); do
    FILE_PATH="${DATA_DIR}/Pitt_00${SUBJECT_ID}_rois_cc200.1D"

    if [ -f "$FILE_PATH" ]; then
        echo "Processing Subject ID: $SUBJECT_ID"
        PYTHONWARNINGS="ignore" python3 predict.py "$MODEL_NAME" "$FILE_PATH"
    else
        echo "Skipping: File $FILE_PATH does not exist."
    fi
done