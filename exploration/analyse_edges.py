#!/usr/bin/env python3
import pandas as pd
import argparse
import sys


def analyze_http_traffic(csv_file):
    print(f"Analyzing HTTP traffic data from: {csv_file}")

    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

    required_cols = ['source', 'destination', 'timestamp', 'value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {', '.join(missing_cols)}")
        print(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)

    df['source_destination'] = df['source'] + ' -> ' + df['destination']

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek 

    df['day_of_week'] = (df['day_of_week'] + 1) % 7

    pairs = df['source_destination'].unique()
    print(f"Found {len(pairs)} unique source-destination pairs")

    print("\n" + "=" * 80)
    print("BASIC STATISTICS BY SOURCE-DESTINATION PAIR")
    print("=" * 80)

    stats_by_pair = {}
    for pair in pairs:
        pair_data = df[df['source_destination'] == pair]['value']
        stats = {
            'count': len(pair_data),
            'min': pair_data.min(),
            'max': pair_data.max(),
            'mean': pair_data.mean(),
            'variance': pair_data.var(),
            'std_dev': pair_data.std(),
            'cv': pair_data.std() / pair_data.mean() if pair_data.mean() > 0 else 0,
            'percentile_95': pair_data.quantile(0.95)
        }
        stats_by_pair[pair] = stats

        print(f"\n{pair}:")
        print(f"  Count: {stats['count']:,}")
        print(f"  Min: {stats['min']:,}")
        print(f"  Max: {stats['max']:,}")
        print(f"  Mean: {stats['mean']:,.0f}")
        print(f"  95th Percentile: {stats['percentile_95']:,.0f}")
        print(f"  Variance: {stats['variance']:,.0f}")
        print(f"  Standard Deviation: {stats['std_dev']:,.0f}")
        print(f"  Coefficient of Variation: {stats['cv']:.4f}")

    print("\n" + "=" * 80)
    print("CORRELATIONS BETWEEN SOURCE-DESTINATION PAIRS")
    print("=" * 80)

    hourly_data = df.groupby(['source_destination', pd.Grouper(key='datetime', freq='1H')])['value'].sum().reset_index()

    pivot_df = hourly_data.pivot(index='datetime', columns='source_destination', values='value')

    if len(pairs) > 20:
        pair_totals = df.groupby('source_destination')['value'].sum().sort_values(ascending=False)
        top_pairs = pair_totals.head(20).index.tolist()
        print(f"\nLimiting correlation analysis to top 20 pairs by traffic volume")
        pivot_df = pivot_df[top_pairs]
        pairs_for_correlation = top_pairs
    else:
        pairs_for_correlation = pairs

    corr_matrix = pivot_df.corr()

    print("\nCorrelation Matrix:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(corr_matrix.round(2))

    print("\nStrong Correlations (|r| > 0.7):")
    correlation_found = False
    for i, pair1 in enumerate(pairs_for_correlation):
        for j, pair2 in enumerate(pairs_for_correlation):
            if i < j and pair1 in corr_matrix.index and pair2 in corr_matrix.columns: 
                try:
                    corr = corr_matrix.loc[pair1, pair2]
                    if abs(corr) > 0.7:
                        print(f"{pair1} <-> {pair2}: {corr:.4f}")
                        correlation_found = True
                except KeyError:
                    continue

    if not correlation_found:
        print("No strong correlations found.")

    print("\n" + "=" * 80)
    print("DAILY TRAFFIC PATTERNS (HOUR OF DAY)")
    print("=" * 80)

    if len(pairs) > 10:
        pair_totals = df.groupby('source_destination')['value'].sum().sort_values(ascending=False)
        top_pairs = pair_totals.head(10).index.tolist()
        print(f"\nAnalyzing daily patterns for top 10 pairs by traffic volume")
        pairs_for_hourly = top_pairs
    else:
        pairs_for_hourly = pairs

    daily_patterns = {}
    for pair in pairs_for_hourly:
        pair_df = df[df['source_destination'] == pair]
        hourly_avg = pair_df.groupby('hour')['value'].mean()
        daily_patterns[pair] = hourly_avg

        peak_hour = hourly_avg.idxmax()
        low_hour = hourly_avg.idxmin()

        print(f"\n{pair}:")
        print(f"  Peak Hour: {peak_hour}:00 ({hourly_avg[peak_hour]:,.0f})")
        print(f"  Low Hour: {low_hour}:00 ({hourly_avg[low_hour]:,.0f})")

    print("\n" + "=" * 80)
    print("WEEKLY TRAFFIC PATTERNS (DAY OF WEEK)")
    print("=" * 80)

    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    weekly_patterns = {}
    for pair in pairs_for_hourly: 
        pair_df = df[df['source_destination'] == pair]
        daily_avg = pair_df.groupby('day_of_week')['value'].mean()
        weekly_patterns[pair] = daily_avg

        peak_day = daily_avg.idxmax()
        low_day = daily_avg.idxmin()

        print(f"\n{pair}:")
        print(f"  Peak Day: {day_names[peak_day]} ({daily_avg[peak_day]:,.0f})")
        print(f"  Low Day: {day_names[low_day]} ({daily_avg[low_day]:,.0f})")

    print("\n" + "=" * 80)
    print("PATTERNS OF INTEREST")
    print("=" * 80)

    print("\n1. Source-Destination Traffic Distribution Analysis:")
    total_by_pair = {pair: stats['count'] * stats['mean'] for pair, stats in stats_by_pair.items()}
    sorted_pairs = sorted(total_by_pair.items(), key=lambda x: x[1], reverse=True)

    print("Source-Destination Pairs by Traffic Volume (Top 20, Descending):")
    for i, (pair, total) in enumerate(sorted_pairs[:20]):
        stats = stats_by_pair[pair]
        print(
            f"{i + 1}. {pair}: {total:,.0f} (Mean: {stats['mean']:,.0f}, 95th Percentile: {stats['percentile_95']:,.0f})")

    print("\n2. Traffic Variability Analysis (CV = std/mean):")
    cv_by_pair = {pair: stats['cv'] for pair, stats in stats_by_pair.items()}
    sorted_by_cv = sorted(cv_by_pair.items(), key=lambda x: x[1], reverse=True)

    print("Source-Destination Pairs by Coefficient of Variation (Top 20, Most Variable First):")
    for i, (pair, cv) in enumerate(sorted_by_cv[:20]):
        stats = stats_by_pair[pair]
        print(f"{i + 1}. {pair}: CV = {cv:.4f} (95th Percentile: {stats['percentile_95']:,.0f})")

    print("\n3. Source-Destination Pairs with Unusual Day/Night Ratios:")
    day_night_ratios = {}
    for pair in pairs_for_hourly: 
        if pair in daily_patterns:
            pattern = daily_patterns[pair]
            day_hours = pattern[8:20]  
            night_hours = pd.concat([pattern[:8], pattern[20:]])  

            avg_day = day_hours.mean()
            avg_night = night_hours.mean()

            day_night_ratios[pair] = avg_day / avg_night if avg_night > 0 else float('inf')

    sorted_by_ratio = sorted(day_night_ratios.items(), key=lambda x: x[1], reverse=True)
    print("Source-Destination Pairs by Day/Night Traffic Ratio (Highest First):")
    for i, (pair, ratio) in enumerate(sorted_by_ratio):
        stats = stats_by_pair[pair]
        print(f"{i + 1}. {pair}: Day/Night Ratio = {ratio:.2f} (95th Percentile: {stats['percentile_95']:,.0f})")

    print("\n4. Top Sources by Total Traffic:")
    source_traffic = df.groupby('source')['value'].sum().sort_values(ascending=False)
    for i, (source, traffic) in enumerate(source_traffic.head(10).items()):
        source_data = df[df['source'] == source]['value']
        source_95th = source_data.quantile(0.95)
        print(f"{i + 1}. {source}: {traffic:,.0f} (95th Percentile: {source_95th:,.0f})")

    print("\n5. Top Destinations by Total Traffic:")
    dest_traffic = df.groupby('destination')['value'].sum().sort_values(ascending=False)
    for i, (dest, traffic) in enumerate(dest_traffic.head(10).items()):
        dest_data = df[df['destination'] == dest]['value']
        dest_95th = dest_data.quantile(0.95)
        print(f"{i + 1}. {dest}: {traffic:,.0f} (95th Percentile: {dest_95th:,.0f})")

    print("\n" + "=" * 80)
    print("95TH PERCENTILE ANALYSIS")
    print("=" * 80)

    print("\n6. Source-Destination Pairs by 95th Percentile (Top 20, Highest First):")
    percentile_by_pair = {pair: stats['percentile_95'] for pair, stats in stats_by_pair.items()}
    sorted_by_percentile = sorted(percentile_by_pair.items(), key=lambda x: x[1], reverse=True)

    for i, (pair, percentile_95) in enumerate(sorted_by_percentile[:20]):
        stats = stats_by_pair[pair]
        ratio_to_mean = percentile_95 / stats['mean'] if stats['mean'] > 0 else 0
        print(f"{i + 1}. {pair}: {percentile_95:,.0f} ({ratio_to_mean:.2f}x mean)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze HTTP traffic data from a CSV file')
    parser.add_argument('csv_file', nargs='?', default="../datasets/transformed_http_1m_7d.csv",
                        help='Path to the CSV file containing HTTP traffic data')
    parser.add_argument('--top', type=int, default=20, help='Number of top pairs to analyze (default: 20)')
    args = parser.parse_args()

    analyze_http_traffic(args.csv_file)