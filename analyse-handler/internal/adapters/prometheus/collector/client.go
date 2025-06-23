package collector

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/prometheus/client_golang/api"
	v1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/model"

	"timmelmann/analyse-handler/internal/core/domain"
)

type ClientConfig struct {
	Url             string
	MetricWindow    string
	StepDuration    string
	Query           string
	Namespace       string
	Labels          []string
	Services        []domain.Service
}

type Client struct {
	api          v1.API
	metricWindow time.Duration
	step         time.Duration
	query        string
	labels       []string
}

func NewClient(clientConfig ClientConfig) (*Client, error) {
	config := api.Config{
		Address: clientConfig.Url,
	}

	client, err := api.NewClient(config)
	if err != nil {
		return nil, err
	}

	window, err := time.ParseDuration(clientConfig.MetricWindow)
	if err != nil {
		return nil, err
	}

	step, err := time.ParseDuration(clientConfig.StepDuration)
	if err != nil {
		return nil, err
	}

	var serviceNames []string
	for _, service := range clientConfig.Services {
		serviceNames = append(serviceNames, service.Name)
	}
	joinedServiceNames := strings.Join(serviceNames, "|")
	queryComplet := fmt.Sprintf(`%s{namespace="%s",app=~"%s"}`, clientConfig.Query, clientConfig.Namespace, joinedServiceNames)
	fmt.Printf("Query: %s\n", queryComplet)
	return &Client{
		api:          v1.NewAPI(client),
		metricWindow: window,
		step:         step,
		query:        queryComplet,
		labels:       clientConfig.Labels,
	}, nil
}

func (c *Client) CollectMetrics(ctx context.Context) ([]domain.Metric, error) {
	result, err := c.queryRange(ctx, c.query)
	if err != nil {
		return nil, err
	}
	return c.parseMetricData(result, c.labels), nil
}

func (c *Client) queryRange(ctx context.Context, query string) (model.Matrix, error) {
	start, end := getTimeWindow(c.metricWindow)
	r := v1.Range{
		Start: start,
		End:   end,
		Step:  c.step,
	}
	result, _, err := c.api.QueryRange(ctx, query, r)
	if err != nil {
		return nil, err
	}
	return result.(model.Matrix), nil
}

func (c *Client) parseMetricData(matrix model.Matrix, selectedLabels []string) []domain.Metric {
	var metrics []domain.Metric
	for _, series := range matrix {
		labels := make(map[string]string)
		for _, labelName := range selectedLabels {
			if value, ok := series.Metric[model.LabelName(labelName)]; ok {
				labels[labelName] = string(value)
			}
		}

		for _, sample := range series.Values {
			metricData := domain.Metric{
				Timestamp: sample.Timestamp.Unix(),
				Value:     float64(sample.Value),
				Labels:    labels,
			}
			metrics = append(metrics, metricData)
		}
	}
	return metrics
}

func getTimeWindow(timeWindow time.Duration) (time.Time, time.Time) {
	end := time.Now()
	start := end.Add(-timeWindow)
	return start, end
}
