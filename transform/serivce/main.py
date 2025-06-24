import pandas as pd


def transform_requests_pivot(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    df['value_diff'] = df.groupby('destination')['value'].diff().fillna(0)

    df['edge_source'] = df.apply(
        lambda row: row['source'] if row['reporter'] == 'source' else row['destination'],
        axis=1
    )
    df['edge_destination'] = df.apply(
        lambda row: row['destination'] if row['reporter'] == 'source' else row['source'],
        axis=1
    )

    grouped = (
        df.groupby(['timestamp', 'edge_source', 'edge_destination'], as_index=False)
        .agg({'value_diff': 'sum'})
    )

    grouped['connection'] = grouped['edge_source'] + '_' + grouped['edge_destination']

    df_pivot = grouped.pivot(index='timestamp', columns='connection', values='value_diff')

    df_pivot = df_pivot.fillna(0)

    df_pivot = df_pivot.reset_index()

    df_pivot.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    input_path = "../../datasets/http_30s.csv"
    output_path = "../../datasets/http_30s_transformed.csv"

    transform_requests_pivot(input_path, output_path)
    print(f"Pivotierte CSV wurde in {output_path} geschrieben.")
