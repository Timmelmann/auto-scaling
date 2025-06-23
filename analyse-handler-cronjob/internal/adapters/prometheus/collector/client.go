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

type Client struct {
	api               v1.API
	metricWindow      time.Duration
	step              time.Duration
	useNameSpace      bool
	useServiceNames   bool
	query             string
	customQueryValues []domain.CustomQueryValues
	labels            []string
	namespace         string
	services          []domain.Service
}

func NewClient(url, metricWindow, stepDuration, query, namespace string, useNamespace, useServiceNames bool, labels []string, services []domain.Service) (*Client, error) {
	config := api.Config{
		Address: url,
	}

	client, err := api.NewClient(config)
	if err != nil {
		return nil, err
	}

	window, err := time.ParseDuration(metricWindow)
	if err != nil {
		return nil, err
	}

	step, err := time.ParseDuration(stepDuration)
	if err != nil {
		return nil, err
	}

	var serviceNames []string
	for _, service := range services {
		fmt.Println("Service Name", service.Name)
		serviceNames = append(serviceNames, service.Name)
	}
	joinedServiceNames := strings.Join(serviceNames, "|")
	fmt.Println("Joined Services", joinedServiceNames)
	queryComplet := fmt.Sprintf(`%s{namespace="%s",app=~"%s"}`, query, namespace, joinedServiceNames)

	return &Client{
		api:             v1.NewAPI(client),
		metricWindow:    window,
		step:            step,
		query:           queryComplet,
		useNameSpace:    useNamespace,
		useServiceNames: useServiceNames,
		labels:          labels,
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
