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
}

func NewClient(url, metricWindow, stepDuration, query string, useNamespace, useServiceNames bool, customQueryValues []domain.CustomQueryValues, labels []string) (*Client, error) {
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

	return &Client{
		api:               v1.NewAPI(client),
		metricWindow:      window,
		step:              step,
		query:             query,
		useNameSpace:      useNamespace,
		useServiceNames:   useServiceNames,
		customQueryValues: customQueryValues,
		labels:            labels,
	}, nil
}

func (c *Client) CollectMetrics(ctx context.Context, namespace string, services []domain.Service) ([]domain.Metric, error) {
	var serviceNames []string
	for _, service := range services {
		fmt.Println("Service Name", service.Name)
		serviceNames = append(serviceNames, service.Name)
	}
	joinedServiceNames := strings.Join(serviceNames, "|")
	fmt.Println("Joined Services", joinedServiceNames)
	queryComplet := fmt.Sprintf(`%s{namespace="%s",app=~"%s"}`, c.query, namespace, joinedServiceNames)
	return c.requestMetrics(ctx, queryComplet)
}

func (c *Client) CollectMetricsByCompleteQuery(ctx context.Context) ([]domain.Metric, error) {
	return c.requestMetrics(ctx, c.query)
}

func (c *Client) CollectMetricsWithBuildQuery(ctx context.Context, query, namespace string, services []domain.Service) ([]domain.Metric, error) {
	var queryKeys = ""
	if c.useServiceNames {
		queryKeys = fmt.Sprintf("namespace=%s", namespace)
	}
	if c.useServiceNames {
		var serviceNames []string
		for _, service := range services {
			fmt.Println("Service Name", service.Name)
			serviceNames = append(serviceNames, service.Name)
		}
		joinedServiceNames := strings.Join(serviceNames, "|")
		queryKeys = fmt.Sprintf("%s,app=%s", queryKeys, joinedServiceNames)
	}
	if queryKeys != "" {
		query = fmt.Sprintf("%s{%s}", query, queryKeys)
	}
	return c.requestMetrics(ctx, query)
}

func (c *Client) requestMetrics(ctx context.Context, query string) ([]domain.Metric, error) {
	result, err := c.queryRange(ctx, query)
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
