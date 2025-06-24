import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import networkx as nx
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path):
    print("Loading data from:", file_path)
    df = pd.read_csv(file_path)

    print("\nData overview:")
    print(f"- Total records: {len(df)}")
    print(
        f"- Time range: {pd.to_datetime(df['timestamp'], unit='s').min()} to {pd.to_datetime(df['timestamp'], unit='s').max()}")
    print(f"- Unique sources: {df['source'].nunique()}")
    print(f"- Unique destinations: {df['destination'].nunique()}")
    print(f"- Unique reporters: {df['reporter'].nunique()}")

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    return df

def create_time_series_by_destination(df):
    destinations = df['destination'].unique()
    time_series_dict = {}

    for dest in destinations:
        dest_data = df[df['destination'] == dest].copy()

        dest_data = dest_data.sort_values('timestamp')
        dest_data.set_index('datetime', inplace=True)

        resampled = dest_data['value'].resample('5min').mean()
        resampled = resampled.fillna(method='ffill').fillna(method='bfill')
        time_series_dict[dest] = resampled
        print(f"- {dest}: {len(resampled)} data points")

    return time_series_dict


def statistical_analysis(time_series_dict):

    print("\nStatistical Analysis:")
    stats_results = {}

    for dest, ts in time_series_dict.items():
        stats_dict = {
            'count': len(ts),
            'mean': ts.mean(),
            'std': ts.std(),
            'min': ts.min(),
            'q25': ts.quantile(0.25),
            'median': ts.median(),
            'q75': ts.quantile(0.75),
            'max': ts.max(),
            'skewness': stats.skew(ts.dropna()),
            'kurtosis': stats.kurtosis(ts.dropna())
        }

        adf_result = adfuller(ts.dropna())
        stats_dict['adf_statistic'] = adf_result[0]
        stats_dict['adf_pvalue'] = adf_result[1]
        stats_dict['is_stationary'] = adf_result[1] < 0.05

        stats_results[dest] = stats_dict

        print(f"\n{dest}:")
        print(f"- Count: {stats_dict['count']}")
        print(f"- Mean: {stats_dict['mean']:.2f}")
        print(f"- Std Dev: {stats_dict['std']:.2f}")
        print(f"- Min/Max: {stats_dict['min']:.2f}/{stats_dict['max']:.2f}")
        print(f"- Median: {stats_dict['median']:.2f}")
        print(f"- Skewness: {stats_dict['skewness']:.2f}")
        print(f"- Kurtosis: {stats_dict['kurtosis']:.2f}")
        print(f"- Is Stationary: {stats_dict['is_stationary']} (p-value: {stats_dict['adf_pvalue']:.4f})")

    return stats_results

def dependency_graph(causality_results, ccf_result, output_dir='./plots'):
    """
    Create a dependency graph based on causality and correlation
    """
    if not causality_results:
        print("No causality results available for dependency graph")
        return None

    print("\nService Dependency Analysis:")

    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    G = nx.DiGraph()

    services = list(causality_results.keys())
    G.add_nodes_from(services)

    edge_labels = {}
    for service1, results in causality_results.items():
        for service2, result in results.items():
            if result['is_causal']:
                try:
                    corr = ccf_result['correlation_matrix'].loc[service1, service2]
                    G.add_edge(
                        service1,
                        service2,
                        weight=abs(corr),
                        lag=result['min_p_lag']
                    )
                    edge_labels[(service1, service2)] = f"{corr:.2f}"
                except:
                    G.add_edge(
                        service1,
                        service2,
                        weight=0.5,
                        lag=result['min_p_lag']
                    )
                    edge_labels[(service1, service2)] = f"lag: {result['min_p_lag']}"

    if len(G.edges()) == 0:
        print("- No significant dependencies found")
        return None


    try:
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        betweenness = nx.betweenness_centrality(G)

        # Print important services
        print("\nService Importance Metrics:")
        print("- Services with high in-degree (depend on many):")
        for service, degree in sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:3]:
            if degree > 0:
                print(f"  {service}: {degree}")

        print("- Services with high out-degree (influence many):")
        for service, degree in sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:3]:
            if degree > 0:
                print(f"  {service}: {degree}")

        print("- Services with high betweenness centrality (critical paths):")
        for service, cent in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]:
            if cent > 0:
                print(f"  {service}: {cent:.4f}")

        plt.figure(figsize=(12, 10))

        pos = nx.spring_layout(G, seed=42)

        node_size = [betweenness[n] * 5000 + 500 for n in G.nodes()]

        node_color = [in_degree[n] for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, cmap=plt.cm.Blues)

        edge_width = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.7, edge_color='gray')

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        plt.title('Service Dependency Graph')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dependency_graph.png")
        plt.close()

        return {
            'graph': G,
            'in_degree': in_degree,
            'out_degree': out_degree,
            'betweenness': betweenness
        }

    except Exception as e:
        print(f"- Error in dependency graph: {e}")
        return None

def analyze_correlations(df, output_dir='./plots'):
    """
    Analyze correlations by source and destination
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\nAnalyzing correlations by source and destination...")

    pivot = df.pivot_table(
        index='source',
        columns='destination',
        values='value',
        aggfunc='mean'
    )

    corr_matrix = pivot.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Destination Correlation Matrix (Based on Source Patterns)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/destination_correlation_heatmap.png")

    print(f"Correlation analysis plot saved to {output_dir}")


def calculate_performance_metrics(df):
    """
    Calculate service performance metrics like reliability and response patterns
    """
    print("\nService Performance Metrics:")

    performance_metrics = {}

    for dest in df['destination'].unique():
        dest_data = df[df['destination'] == dest]

        try:
            total_requests = len(dest_data)
            values = dest_data['value']
            mean_value = values.mean()
            std_value = values.std()

            high_value_threshold = mean_value + 2 * std_value
            high_latency_count = len(values[values > high_value_threshold])

            p50 = values.quantile(0.5)
            p90 = values.quantile(0.9)
            p99 = values.quantile(0.99)

            performance_metrics[dest] = {
                'total_requests': total_requests,
                'mean_value': mean_value,
                'median_value': p50,
                'p90_value': p90,
                'p99_value': p99,
                'high_latency_count': high_latency_count,
                'high_latency_percentage': (high_latency_count / total_requests) * 100
            }

            print(f"\n{dest}:")
            print(f"- Total Requests: {total_requests}")
            print(f"- Mean Value: {mean_value:.2f}")
            print(f"- P50/P90/P99: {p50:.2f}/{p90:.2f}/{p99:.2f}")
            print(f"- High Latency Percentage: {(high_latency_count / total_requests) * 100:.2f}%")

        except Exception as e:
            print(f"- Error calculating performance metrics for {dest}: {e}")

    return performance_metrics

def generate_report(file_path, output_dir='./plots'):
    """
    Run the full analysis and generate a report
    """
    print("=" * 60)
    print("HTTP TIME SERIES ANALYSIS REPORT")
    print("=" * 60)

    df = load_and_prepare_data(file_path)

    time_series_dict = create_time_series_by_destination(df)

    stats_results = statistical_analysis(time_series_dict)

    analyze_correlations(df, output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    non_stationary = []
    for dest, stats in stats_results.items():
        if not stats['is_stationary']:
            non_stationary.append(dest)

    if non_stationary:
        print(f"\nNon-stationary services: {', '.join(non_stationary)}")
    else:
        print("\nAll services exhibit stationary behavior")

if __name__ == "__main__":
    file_path = "../datasets/transformed_http_1m_7d.csv"
    output_dir = "./plots"

    generate_report(file_path, output_dir)

    df = pd.read_csv(file_path)
    performance_metrics = calculate_performance_metrics(df)