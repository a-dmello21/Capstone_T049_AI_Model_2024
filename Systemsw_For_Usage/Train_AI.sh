#!/bin/bash
if [ ! -f "training_data.xlsx" ]; then
  echo "Error: 'training_data.xlsx' not found. Please add the dataset in the current directory."
  exit 1
fi

pip install tensorflow pandas scikit-learn openpyxl

echo "Starting the model training process..."

python  AI_Model_V2_Multiple.py