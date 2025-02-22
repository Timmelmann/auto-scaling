package pushgateway

import (
	"math"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/push"

	"timmelmann/analyse-handler/internal/core/domain"
)

type Client struct {
	pushgatewayURL string
	jobName        string
	registry       *prometheus.Registry
	requestGauge   *prometheus.GaugeVec
}

func NewClient(pushgatewayURL, metricsName, jobName string) *Client {
	registry := prometheus.NewRegistry()

	// Create a gauge vector with service label
	requestGauge := prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: metricsName,
			Help: "Total number of requests per service",
		},
		[]string{"service", "namespace"},
	)

	registry.MustRegister(requestGauge)

	return &Client{
		pushgatewayURL: pushgatewayURL,
		jobName:        jobName,
		registry:       registry,
		requestGauge:   requestGauge,
	}
}

func (c *Client) PublishScalingPrediction(httpMetrics domain.ScalingPrediction, namespace string) error {
	for _, service := range httpMetrics.Predictions {
		c.requestGauge.WithLabelValues(service.Service, namespace).Set(math.Round(service.Value))
	}
	pusher := push.New(c.pushgatewayURL, c.jobName)
	return pusher.Gatherer(c.registry).Push()
}
