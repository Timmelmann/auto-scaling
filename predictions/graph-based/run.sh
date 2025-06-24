#!/bin/bash

# Script to run the improved graph-based LSTM model 5 times with specific seeds
# For use with the modified test3.py that supports command line arguments

# Define common parameters
DATA_PATH="../../resource/transformed_http_1m_7d.csv"
WINDOW_MINUTES=24
PREDICTION_MINUTES=1
HIDDEN_DIM=64
BATCH_SIZE=32  # Increased for anti-overfitting
EPOCHS=150     # Reduced for anti-overfitting
LR=0.001
PATIENCE=15
OUTPUT_DIR="outputs"

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data file '$DATA_PATH' not found!"
    echo "Please make sure the data file exists in the current directory."
    exit 1
fi

# Check if test3.py exists
if [ ! -f "graph_based_model.py" ]; then
    echo "❌ Error: test3.py not found!"
    echo "Please make sure test3.py is in the current directory."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Define specific seeds for reproducibility
SEEDS=(13 23 42 2 69)

echo "🚀 Starting multi-seed training runs..."
echo "📊 Configuration:"
echo "  - Data: $DATA_PATH"
echo "  - Window: ${WINDOW_MINUTES} minutes"
echo "  - Prediction: ${PREDICTION_MINUTES} minutes"
echo "  - Hidden dim: $HIDDEN_DIM"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Epochs: $EPOCHS"
echo "  - Learning rate: $LR"
echo "  - Patience: $PATIENCE"
echo "  - Seeds: ${SEEDS[@]}"
echo ""

# Run the model with each seed
for i in {0..4}; do
    SEED=${SEEDS[$i]}
    echo "========================================================"
    echo "🎲 Starting run $((i+1))/5 with seed $SEED"
    echo "========================================================"

    # Define output directory for this run
    RUN_OUTPUT_DIR="${OUTPUT_DIR}/run_${SEED}"

    # Create run-specific output directory
    mkdir -p $RUN_OUTPUT_DIR

    # Define model and results file paths INSIDE the output directory
    MODEL_PATH="${RUN_OUTPUT_DIR}/improved_model_seed${SEED}.pt"
    RESULTS_FILE="${RUN_OUTPUT_DIR}/model_results_seed${SEED}.json"

    # Run the model with the specified seed
    echo "📝 Command: python test3.py with seed $SEED"

    python graph_based_model.py \
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

    # Check if the run was successful
    if [ $? -eq 0 ]; then
        echo "✅ Run $((i+1)) completed successfully"

        # Check if results file was created
        if [ -f "$RESULTS_FILE" ]; then
            echo "📋 Results saved to: $RESULTS_FILE"
        else
            echo "⚠️  Warning: Results file not found at $RESULTS_FILE"
        fi

        # Check if model file was created
        if [ -f "$MODEL_PATH" ]; then
            echo "💾 Model saved to: $MODEL_PATH"
        else
            echo "⚠️  Warning: Model file not found at $MODEL_PATH"
        fi
    else
        echo "❌ Run $((i+1)) failed with exit code $?"
    fi

    echo ""
done

echo "🎉 All runs completed!"
echo ""
echo "📁 Model files saved as:"
for SEED in ${SEEDS[@]}; do
    MODEL_FILE="${OUTPUT_DIR}/run_${SEED}/improved_model_seed${SEED}.pt"
    RESULTS_FILE="${OUTPUT_DIR}/run_${SEED}/model_results_seed${SEED}.json"

    if [ -f "$MODEL_FILE" ]; then
        echo "  ✅ ${MODEL_FILE}"
    else
        echo "  ❌ ${MODEL_FILE} (missing)"
    fi

    if [ -f "$RESULTS_FILE" ]; then
        echo "  ✅ ${RESULTS_FILE}"
    else
        echo "  ❌ ${RESULTS_FILE} (missing)"
    fi
done

echo ""
echo "📊 Creating summary of all results..."

# Create a summary of all results
python << 'EOF'
import json
import numpy as np
import os
import sys

output_dir = "outputs"
seeds = [13, 23, 42, 1337, 69]
results = {}
successful_runs = 0

print("🔍 Analyzing results from all runs...")

for seed in seeds:
    try:
        results_path = os.path.join(output_dir, f'run_{seed}', f'model_results_seed{seed}.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                data = json.load(f)
                results[f'seed_{seed}'] = data
                successful_runs += 1
                print(f'  ✅ Successfully loaded results for seed {seed}')
        else:
            print(f'  ❌ Results file for seed {seed} not found at {results_path}')
    except json.JSONDecodeError as e:
        print(f'  ❌ Could not parse JSON for seed {seed}: {e}')
    except Exception as e:
        print(f'  ❌ Error loading results for seed {seed}: {e}')

if results:
    print(f"\n📊 Processing {successful_runs}/{len(seeds)} successful runs...")

    # Calculate averages - using the correct key names from the improved model
    all_mae = []
    all_rmse = []
    all_smape = []
    all_mape = []
    all_directional_acc = []

    for seed_key, result_data in results.items():
        if 'average' in result_data:
            avg_data = result_data['average']
            if avg_data.get('mae') is not None:
                all_mae.append(avg_data['mae'])
            if avg_data.get('rmse') is not None:
                all_rmse.append(avg_data['rmse'])
            if avg_data.get('smape') is not None:
                all_smape.append(avg_data['smape'])
            if avg_data.get('mape') is not None and not np.isnan(avg_data['mape']):
                all_mape.append(avg_data['mape'])
            if avg_data.get('directional_accuracy') is not None:
                all_directional_acc.append(avg_data['directional_accuracy'])

    summary = {
        'total_runs': len(seeds),
        'successful_runs': successful_runs,
        'failed_runs': len(seeds) - successful_runs,
        'avg_mae': np.mean(all_mae) if all_mae else None,
        'std_mae': np.std(all_mae) if all_mae else None,
        'avg_rmse': np.mean(all_rmse) if all_rmse else None,
        'std_rmse': np.std(all_rmse) if all_rmse else None,
        'avg_smape': np.mean(all_smape) if all_smape else None,
        'std_smape': np.std(all_smape) if all_smape else None,
        'avg_mape': np.mean(all_mape) if all_mape else None,
        'std_mape': np.std(all_mape) if all_mape else None,
        'avg_directional_accuracy': np.mean(all_directional_acc) if all_directional_acc else None,
        'std_directional_accuracy': np.std(all_directional_acc) if all_directional_acc else None,
        'individual_results': results
    }

    # Save summary in the outputs directory
    summary_path = os.path.join(output_dir, 'model_results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n💾 Summary saved to: {summary_path}')
    print('\n📈 FINAL SUMMARY STATISTICS:')
    print('='*50)

    if summary['avg_mae'] is not None:
        print(f'  📊 Average MAE: {summary["avg_mae"]:.4f} ± {summary["std_mae"]:.4f}')
        print(f'  📊 Average RMSE: {summary["avg_rmse"]:.4f} ± {summary["std_rmse"]:.4f}')
        print(f'  📊 Average SMAPE: {summary["avg_smape"]:.2f}% ± {summary["std_smape"]:.2f}%')
        if summary["avg_mape"] is not None:
            print(f'  📊 Average MAPE: {summary["avg_mape"]:.2f}% ± {summary["std_mape"]:.2f}%')
        if summary["avg_directional_accuracy"] is not None:
            print(f'  📊 Average Directional Accuracy: {summary["avg_directional_accuracy"]:.1f}% ± {summary["std_directional_accuracy"]:.1f}%')
        print(f'  📊 Successful runs: {summary["successful_runs"]}/{summary["total_runs"]}')

        # Performance assessment
        print('\n🎯 PERFORMANCE ASSESSMENT:')
        if summary["avg_directional_accuracy"] is not None:
            if summary["avg_directional_accuracy"] > 75:
                print('  ⚠️  High directional accuracy - possible overfitting')
            elif summary["avg_directional_accuracy"] < 45:
                print('  ⚠️  Low directional accuracy - possible underfitting')
            else:
                print('  ✅ Good directional accuracy - healthy model')

        if summary["avg_mape"] is not None:
            if summary["avg_mape"] < 10:
                print('  ✅ Excellent MAPE performance (<10%)')
            elif summary["avg_mape"] < 20:
                print('  ✅ Good MAPE performance (10-20%)')
            elif summary["avg_mape"] < 30:
                print('  ⚠️  Moderate MAPE performance (20-30%)')
            else:
                print('  ❌ Poor MAPE performance (>30%)')
    else:
        print('  ❌ No successful runs to summarize')
        sys.exit(1)

else:
    print('❌ No results files found to summarize')
    sys.exit(1)
EOF

# Check if summary was created successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Script completed successfully!"
    echo "📁 Check the ${OUTPUT_DIR} directory for all results:"
    echo "   - Individual run directories: ${OUTPUT_DIR}/run_<seed>/"
    echo "   - Overall summary: ${OUTPUT_DIR}/model_results_summary.json"
else
    echo ""
    echo "❌ Script completed with errors during summary creation"
    echo "📁 Check individual run directories in ${OUTPUT_DIR}/ for partial results"
fi