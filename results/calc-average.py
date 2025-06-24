import pandas as pd
import numpy as np


def analyze_service_metrics(csv_file_path='model_results.csv'):
    df = pd.read_csv(csv_file_path)

    print(f"Data shape: {df.shape}")
    print(f"Number of unique services: {df['Service'].nunique()}")
    print(f"Number of unique seeds: {df['Seed'].nunique()}")

    service_averages = df.groupby('Service').agg({
        'MAE': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'MAPE': ['mean', 'std']
    }).round(2)

    service_averages.columns = ['_'.join(col).strip() for col in service_averages.columns.values]

    service_averages = service_averages.reset_index()

    service_averages = service_averages.sort_values('RMSE_mean')

    return service_averages


def display_results(results_df):
    """
    Display the results in a formatted table.
    """
    print("\n" + "=" * 100)
    print("AVERAGE PERFORMANCE METRICS BY SERVICE (across all seeds)")
    print("=" * 100)

    print(f"{'Service':<25} {'MAE (avg±std)':<20} {'RMSE (avg±std)':<20} {'MAPE (avg±std)':<15}")
    print("-" * 100)

    for _, row in results_df.iterrows():
        service = row['Service']
        mae_avg = f"{row['MAE_mean']:,.0f}"
        mae_std = f"{row['MAE_std']:,.0f}"
        rmse_avg = f"{row['RMSE_mean']:,.0f}"
        rmse_std = f"{row['RMSE_std']:,.0f}"
        mape_avg = f"{row['MAPE_mean']:.1f}%"
        mape_std = f"{row['MAPE_std']:.1f}%"

        print(f"{service:<25} {mae_avg}±{mae_std:<20} {rmse_avg}±{rmse_std:<20} {mape_avg}±{mape_std:<15}")


def save_summary(results_df, output_file='service_averages.csv'):
    """
    Save the summary results to a CSV file.
    """
    results_df.to_csv(output_file, index=False)
    print(f"\nSummary results saved to: {output_file}")


def main():
    """
    Main function to run the analysis.
    """
    try:
        results = analyze_service_metrics("evaluation_results.csv")

        display_results(results)

        save_summary(results)

        print("\n" + "=" * 60)
        print("ADDITIONAL STATISTICS")
        print("=" * 60)

        best_service = results.iloc[0]['Service']
        worst_service = results.iloc[-1]['Service']

        print(f"Best performing service (lowest avg RMSE): {best_service}")
        print(f"Worst performing service (highest avg RMSE): {worst_service}")

        print(f"\nOverall average RMSE: {results['RMSE_mean'].mean():,.0f}")
        print(f"Overall average MAE: {results['MAE_mean'].mean():,.0f}")
        print(f"Overall average MAPE: {results['MAPE_mean'].mean():.1f}%")

        high_variance_threshold = results['RMSE_std'].quantile(0.75)
        high_variance_services = results[results['RMSE_std'] > high_variance_threshold]['Service'].tolist()

        print(f"\nServices with high performance variance across seeds:")
        for service in high_variance_services:
            print(f"  - {service}")

    except FileNotFoundError:
        print("Error: CSV file not found. Please make sure 'model_results.csv' exists in the current directory.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()