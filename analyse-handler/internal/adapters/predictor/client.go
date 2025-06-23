package predictor

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"timmelmann/analyse-handler/internal/core/domain"
)

type Client struct {
	baseURL      string
	httpClient   *http.Client
}

func NewClient(baseURL string) *Client {
	return &Client{
		baseURL:    baseURL,
		httpClient: &http.Client{},
	}
}

func (c *Client) PredictMetric(metrics []domain.Metric) (*domain.ScalingPrediction, error) {

	body, err := json.Marshal(flattenMetrics(metrics))
	if err != nil {
		return nil, fmt.Errorf("error marshaling metrics: %w", err)
	}

	req, err := http.NewRequest("POST", c.baseURL+"/predict", bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error sending request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var prediction domain.ScalingPrediction
	if err := json.NewDecoder(resp.Body).Decode(&prediction); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &prediction, nil
}

func (c *Client) UpdateConfig(PredictorUrl string) {
	c.baseURL = PredictorUrl
}

func flattenMetrics(metrics []domain.Metric) []map[string]interface{} {

	flattened := make([]map[string]interface{}, 0, len(metrics))

	for _, metric := range metrics {
		t := time.Unix(metric.Timestamp, 0)
		entry := map[string]interface{}{
			"timestamp": t,
			"value":     metric.Value,
		}
		for k, v := range metric.Labels {
			entry[k] = v
		}

		flattened = append(flattened, entry)
	}

	return flattened
}
