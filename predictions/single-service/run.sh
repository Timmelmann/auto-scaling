#!/bin/bash

# Array of seeds to use
SEEDS=(13 23 42 1337 69)

# Path to the CSV file
CSV_PATH="transformed_http_1m_7d.csv"

# Base output directory
OUTPUT_DIR="lstm_results_multi_seed_adaptive"

# Other parameters
SEQUENCE_LENGTH=24
EPOCHS=150
BATCH_SIZE=32
PATIENCE=25
FUTURE_STEPS=48

# Training mode: "all" for all services, "adaptive" for problematic services only
TRAINING_MODE="all"

# Specific destinations (optional - leave empty to train all)
SPECIFIC_DESTINATIONS=""

# Create the base output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Log file to track progress
LOG_FILE="$OUTPUT_DIR/training_log.txt"
echo "Starting adaptive multi-seed training at $(date)" > $LOG_FILE
echo "Training mode: $TRAINING_MODE" >> $LOG_FILE
echo "Seeds: ${SEEDS[*]}" >> $LOG_FILE
echo "Epochs: $EPOCHS, Patience: $PATIENCE" >> $LOG_FILE

# Train models for each seed
for seed in "${SEEDS[@]}"; do
    echo "Starting adaptive training with seed $seed at $(date)" | tee -a $LOG_FILE

    # Build the command based on training mode
    CMD="python single-service-lstm.py \
        --csv_path $CSV_PATH \
        --output_dir $OUTPUT_DIR \
        --sequence_length $SEQUENCE_LENGTH \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --patience $PATIENCE \
        --future_steps $FUTURE_STEPS \
        --seed $seed"

    # Add specific destinations if provided
    if [ ! -z "$SPECIFIC_DESTINATIONS" ]; then
        CMD="$CMD --specific_destinations $SPECIFIC_DESTINATIONS"
    fi

    # Add adaptive-only flag if in adaptive mode
    if [ "$TRAINING_MODE" = "adaptive" ]; then
        CMD="$CMD --adaptive_only"
    fi

    # Execute the command
    echo "Executing: $CMD" | tee -a $LOG_FILE
    eval $CMD

    echo "Completed training with seed $seed at $(date)" | tee -a $LOG_FILE
done

echo "All adaptive training completed at $(date)" | tee -a $LOG_FILE

# Generate a summary of metrics across all seeds
echo "Generating metrics summary..." | tee -a $LOG_FILE
python - << EOF
import pandas as pd
import numpy as np
import os
import glob

output_dir = "$OUTPUT_DIR"
seed_dirs = glob.glob(os.path.join(output_dir, "seed_*"))

print(f"Looking for seed directories in: {output_dir}")
print(f"Found seed directories: {seed_dirs}")

# Collect all metrics files
all_metrics_dfs = []
for seed_dir in seed_dirs:
    seed = os.path.basename(seed_dir).split("_")[1]
    metrics_file = os.path.join(seed_dir, f"all_models_metrics_seed{seed}.csv")
    print(f"Checking for metrics file: {metrics_file}")

    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        print(f"Loaded {len(df)} rows from {metrics_file}")
        all_metrics_dfs.append(df)
    else:
        print(f"Metrics file not found: {metrics_file}")

if all_metrics_dfs:
    # Combine all metrics
    combined_df = pd.concat(all_metrics_dfs, ignore_index=True)
    print(f"Combined {len(combined_df)} total rows from {len(all_metrics_dfs)} files")

    # Save the combined metrics
    combined_file = os.path.join(output_dir, "all_seeds_combined_metrics.csv")
    combined_df.to_csv(combined_file, index=False)
    print(f"Combined metrics saved to {combined_file}")

    # Calculate summary statistics by destination
    print("\nCalculating summary statistics...")
    summary_stats = combined_df.groupby('destination').agg({
        'rmse': ['mean', 'std', 'min', 'max'],
        'mae': ['mean', 'std', 'min', 'max'],
        'r_squared': ['mean', 'std', 'min', 'max'],
        'mape': ['mean', 'std', 'min', 'max'],
        'seed': 'count'
    }).round(4)

    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats = summary_stats.rename(columns={'seed_count': 'num_runs'})

    # Save detailed summary
    summary_file = os.path.join(output_dir, "detailed_metrics_summary_by_destination.csv")
    summary_stats.to_csv(summary_file)
    print(f"Detailed metrics summary saved to {summary_file}")

    # Create a more readable summary
    print("\nCreating readable summary...")
    readable_summary = []
    for dest in combined_df['destination'].unique():
        dest_data = combined_df[combined_df['destination'] == dest]

        # Get the most common approach for this destination
        most_common_approach = dest_data['approach'].mode()[0] if 'approach' in dest_data.columns else 'unknown'

        dest_summary = {
            'Destination': dest,
            'Approach': most_common_approach,
            'Number of Runs': len(dest_data),
            'R² (mean ± std)': f"{dest_data['r_squared'].mean():.4f} ± {dest_data['r_squared'].std():.4f}",
            'R² Range': f"[{dest_data['r_squared'].min():.4f}, {dest_data['r_squared'].max():.4f}]",
            'MAPE (mean ± std)': f"{dest_data['mape'].mean():.2f} ± {dest_data['mape'].std():.2f}%",
            'RMSE (mean ± std)': f"{dest_data['rmse'].mean():.2f} ± {dest_data['rmse'].std():.2f}",
            'MAE (mean ± std)': f"{dest_data['mae'].mean():.2f} ± {dest_data['mae'].std():.2f}"
        }
        readable_summary.append(dest_summary)

    readable_df = pd.DataFrame(readable_summary)

    # Sort by mean R²
    readable_df['r2_mean'] = combined_df.groupby('destination')['r_squared'].mean().values
    readable_df = readable_df.sort_values('r2_mean', ascending=False).drop('r2_mean', axis=1)

    readable_file = os.path.join(output_dir, "readable_metrics_summary.csv")
    readable_df.to_csv(readable_file, index=False)
    print(f"Readable metrics summary saved to {readable_file}")

    # Print the readable summary to console
    print("\n" + "="*120)
    print("MULTI-SEED ADAPTIVE LSTM RESULTS SUMMARY")
    print("="*120)
    print(readable_df.to_string(index=False))
    print("="*120)

    # Print best and worst performers
    print(f"\nBEST PERFORMERS (by mean R²):")
    top_3 = readable_df.head(3)
    for _, row in top_3.iterrows():
        print(f"  {row['Destination']:<20} R² = {row['R² (mean ± std)']:<20} ({row['Approach']} approach)")

    print(f"\nWORST PERFORMERS (by mean R²):")
    bottom_3 = readable_df.tail(3)
    for _, row in bottom_3.iterrows():
        print(f"  {row['Destination']:<20} R² = {row['R² (mean ± std)']:<20} ({row['Approach']} approach)")

    # Calculate overall statistics
    overall_mean_r2 = combined_df['r_squared'].mean()
    overall_std_r2 = combined_df['r_squared'].std()
    services_above_05 = (combined_df.groupby('destination')['r_squared'].mean() > 0.5).sum()
    total_services = combined_df['destination'].nunique()

    print(f"\nOVERALL STATISTICS:")
    print(f"  Overall mean R²: {overall_mean_r2:.4f} ± {overall_std_r2:.4f}")
    print(f"  Services with R² > 0.5: {services_above_05}/{total_services} ({services_above_05/total_services*100:.1f}%)")
    print(f"  Total training runs: {len(combined_df)}")
    print(f"  Seeds used: {sorted(combined_df['seed'].unique())}")

else:
    print("No metrics files found. Please check if training completed successfully.")
    print(f"Expected to find files like: seed_*/all_models_metrics_seed*.csv")
EOF

echo "All processing completed at $(date)" | tee -a $LOG_FILE

# Create a final report
echo "Creating final report..." | tee -a $LOG_FILE
cat > "$OUTPUT_DIR/experiment_report.md" << EOL
# Adaptive LSTM Multi-Seed Training Report

## Experiment Configuration
- **Training Script**: single-service-lstm.py
- **Training Mode**: $TRAINING_MODE
- **Seeds Used**: ${SEEDS[*]}
- **Epochs**: $EPOCHS
- **Patience**: $PATIENCE
- **Batch Size**: $BATCH_SIZE
- **Sequence Length**: $SEQUENCE_LENGTH
- **Dataset**: $CSV_PATH
- **Start Time**: $(head -1 $LOG_FILE | cut -d' ' -f6-)
- **End Time**: $(date)

## Output Files
- **Combined Metrics**: all_seeds_combined_metrics.csv
- **Readable Summary**: readable_metrics_summary.csv
- **Detailed Summary**: detailed_metrics_summary_by_destination.csv
- **Training Log**: training_log.txt

## Key Features
- Adaptive approach selection based on service characteristics
- Automatic handling of problematic services (currencyservice, paymentservice, etc.)
- Robust scaling and outlier detection
- Multi-seed evaluation for statistical reliability

## Next Steps
1. Review the readable_metrics_summary.csv for overall performance
2. Check individual seed directories for detailed training logs
3. Analyze which approaches work best for different service types
4. Consider ensemble methods for services with high variance across seeds

EOL

echo "Final report saved to: $OUTPUT_DIR/experiment_report.md" | tee -a $LOG_FILE
echo "Experiment completed successfully!" | tee -a $LOG_FILE