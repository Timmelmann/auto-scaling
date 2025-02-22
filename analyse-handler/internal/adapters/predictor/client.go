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
	MetricWindow time.Duration
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

	// Create request
	req, err := http.NewRequest("POST", c.baseURL+"/predict", bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Send request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error sending request: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	fmt.Printf("Response message: %s", fmt.Sprint(resp.Body))
	// Parse response
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
		// Convert your Unix timestamp to a string
		t := time.Unix(metric.Timestamp, 0)

		// Build a generic map for this entry
		entry := map[string]interface{}{
			"timestamp": t,
			"value":     metric.Value,
		}

		// Merge all Labels into the top-level map
		for k, v := range metric.Labels {
			entry[k] = v
		}

		flattened = append(flattened, entry)
	}

	return flattened
}
