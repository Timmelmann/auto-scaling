package main

import (
	"context"
	"fmt"
	"os"

	"timmelmann/analyse-handler/internal/adapters/predictor"
	prometheus "timmelmann/analyse-handler/internal/adapters/prometheus/collector"
	"timmelmann/analyse-handler/internal/adapters/prometheus/pushgateway"
	"timmelmann/analyse-handler/internal/core/app"
	"timmelmann/analyse-handler/internal/core/config"
	"timmelmann/analyse-handler/internal/core/domain"
)

func main() {
	cfg := config.LoadFromEnv()
	cfg.PrometheusURL = "http://localhost:60304"
	cfg.PushgatewayURL = "http://localhost:57115"
	cfg.PredictorUrl = "http://localhost:5001"
	cfg.MetricsWindow = "12m"
	var labels = []string{"source_app", "destination_app", "reporter"}
	// Initialize adapters
	prometheusAdapter, err := prometheus.NewClient(cfg.PrometheusURL, cfg.MetricsWindow, "5m", "istio_requests_total", true, true, nil, labels)
	if err != nil {
		fmt.Printf("Error creating Prometheus client: %v\n", err)
		os.Exit(1)
	}

	predictionAdapter := predictor.NewClient(cfg.PredictorUrl)
	pushgatewayAdapter := pushgateway.NewClient(cfg.PushgatewayURL, cfg.PredictedMetricName, cfg.JobName)

	services := []domain.Service{
		domain.Service{
			Name:       "adservice",
			Deployment: "adservice",
		},
		domain.Service{
			Name:       "recommendationservice",
			Deployment: "recommendationservice",
		},
		domain.Service{
			Name:       "currencyservice",
			Deployment: "currencyservice",
		},
		domain.Service{
			Name:       "productcatalogservice",
			Deployment: "productcatalogservice",
		},
		domain.Service{
			Name:       "paymentservice",
			Deployment: "paymentservice",
		},
		domain.Service{
			Name:       "emailservice",
			Deployment: "emailservice",
		},
		domain.Service{
			Name:       "checkoutservice",
			Deployment: "checkoutservice",
		},
		domain.Service{
			Name:       "shippingservice",
			Deployment: "shippingservice",
		},
		domain.Service{
			Name:       "cartservice",
			Deployment: "cartservice",
		},
	}

	// Initialize application service
	service := app.NewMetricsService(
		prometheusAdapter,
		predictionAdapter,
		pushgatewayAdapter,
		services,
		"webshop",
	)

	// Run the collection
	ctx := context.Background()

	if err := service.CollectAndAnalyze(ctx); err != nil {
		fmt.Printf("Error collecting metrics: %v\n", err)
		os.Exit(1)
	}
}
