#!/bin/bash

# Script to run the improved graph-based LSTM model 5 times with specific seeds
# For use with the modified short_script.py that supports --seed and --results_file

# Define common parameters
DATA_PATH="../../resource/transformed_http_1m_7d.csv"
WINDOW_MINUTES=12
PREDICTION_MINUTES=1
HIDDEN_DIM=64
BATCH_SIZE=16
EPOCHS=200
LR=0.001
PATIENCE=15
OUTPUT_DIR="outputs"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Define specific seeds
SEEDS=(13 23 42 1337 69)

# Run the model with each seed
for i in {0..4}; do
    SEED=${SEEDS[$i]}
    echo "========================================================"
    echo "Starting run $((i+1)) with seed $SEED"
    echo "========================================================"

    # Define output directory for this run
    RUN_OUTPUT_DIR="${OUTPUT_DIR}/run_${SEED}"

    # Create run-specific output directory
    mkdir -p $RUN_OUTPUT_DIR

    # Define model and results file paths INSIDE the output directory
    MODEL_PATH="${RUN_OUTPUT_DIR}/improved_model_seed${SEED}.pt"
    RESULTS_FILE="${RUN_OUTPUT_DIR}/model_results_seed${SEED}.json"

    # Run the model with the specified seed using short_script.py
    python model.py \
        --data_path $DATA_PATH \
        --window_minutes $WINDOW_MINUTES \
        --prediction_minutes $PREDICTION_MINUTES \
        --hidden_dim $HIDDEN_DIM \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --patience $PATIENCE \
        --model_path $MODEL_PATH \
        --results_file $RESULTS_FILE \
        --output_dir $RUN_OUTPUT_DIR \
        --seed $SEED

    echo "Finished run $((i+1))"
    echo ""
done

echo "All runs completed!"
echo "Model files saved as:"
for SEED in ${SEEDS[@]}; do
    echo "  - ${OUTPUT_DIR}/run_${SEED}/improved_model_seed${SEED}.pt"
    echo "  - ${OUTPUT_DIR}/run_${SEED}/model_results_seed${SEED}.json"
done

# Create a summary of all results
echo "Creating summary of all results..."
python << 'EOF'
import json
import numpy as np
import os

output_dir = "outputs"
seeds = [13, 23, 42, 2, 69]
results = {}

for seed in seeds:
    try:
        results_path = os.path.join(output_dir, f'run_{seed}', f'model_results_seed{seed}.json')
        with open(results_path, 'r') as f:
            data = json.load(f)
            results[f'seed_{seed}'] = data
            print(f'Successfully loaded results for seed {seed}')
    except FileNotFoundError:
        print(f'Warning: Results file for seed {seed} not found at {results_path}')
    except json.JSONDecodeError:
        print(f'Warning: Could not parse JSON for seed {seed}')

if results:
    # Calculate averages - using the correct key names from the improved model
    all_mae = [r['average']['mae'] for r in results.values() if r.get('average', {}).get('mae') is not None]
    all_rmse = [r['average']['rmse'] for r in results.values() if r.get('average', {}).get('rmse') is not None]
    all_smape = [r['average']['smape'] for r in results.values() if r.get('average', {}).get('smape') is not None]
    all_mape = [r['average']['mape'] for r in results.values() if r.get('average', {}).get('mape') is not None and not np.isnan(r['average']['mape']) and r['average']['mape'] is not None]

    summary = {
        'avg_mae': np.mean(all_mae) if all_mae else None,
        'std_mae': np.std(all_mae) if all_mae else None,
        'avg_rmse': np.mean(all_rmse) if all_rmse else None,
        'std_rmse': np.std(all_rmse) if all_rmse else None,
        'avg_smape': np.mean(all_smape) if all_smape else None,
        'std_smape': np.std(all_smape) if all_smape else None,
        'avg_mape': np.mean(all_mape) if all_mape else None,
        'std_mape': np.std(all_mape) if all_mape else None,
        'num_successful_runs': len(all_mae),
        'individual_results': results
    }

    # Save summary in the outputs directory
    summary_path = os.path.join(output_dir, 'model_results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nSummary saved to: {summary_path}')
    print('\nSummary statistics:')
    if summary['avg_mae'] is not None:
        print(f'  Average MAE: {summary["avg_mae"]:.4f} ± {summary["std_mae"]:.4f}')
        print(f'  Average RMSE: {summary["avg_rmse"]:.4f} ± {summary["std_rmse"]:.4f}')
        print(f'  Average SMAPE: {summary["avg_smape"]:.2f}% ± {summary["std_smape"]:.2f}%')
        if summary["avg_mape"] is not None:
            print(f'  Average MAPE: {summary["avg_mape"]:.2f}% ± {summary["std_mape"]:.2f}%')
        print(f'  Successful runs: {summary["num_successful_runs"]}/{len(seeds)}')
    else:
        print('  No successful runs to summarize')
else:
    print('No results files found to summarize')
EOF

echo "Script completed!"
echo "Check the ${OUTPUT_DIR} directory for all results"