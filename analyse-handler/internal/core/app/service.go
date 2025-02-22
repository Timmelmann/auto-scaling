package app

import (
	"context"
	"fmt"

	"timmelmann/analyse-handler/internal/core/domain"
	"timmelmann/analyse-handler/internal/core/ports"
)

type MetricsService struct {
	Collector ports.MetricCollector
	Predictor ports.MetricsPredictor
	Pusher    ports.MetricsPusher
	Services  []domain.Service
	Namespace string
}

func NewEmptyMetricsService() *MetricsService {
	return &MetricsService{}
}

func NewMetricsService(
	collector ports.MetricCollector,
	predictor ports.MetricsPredictor,
	pusher ports.MetricsPusher,
	services []domain.Service,
	namespace string,
) *MetricsService {
	return &MetricsService{
		Collector: collector,
		Predictor: predictor,
		Pusher:    pusher,
		Services:  services,
		Namespace: namespace,
	}
}

func (s *MetricsService) CollectAndAnalyze(ctx context.Context) error {

	_, err := s.Collector.CollectMetrics(ctx, s.Namespace, s.Services)
	if err != nil {
		return fmt.Errorf("error fetching metric: %w", err)
	}

	// httpPredictions, err := s.Predictor.PredictMetric(metricData)
	// if err != nil {
	// return fmt.Errorf("error analyzing metrics: %w", err)
	// }
	httpPredictions := domain.ScalingPrediction{Predictions: []domain.Prediction{domain.Prediction{
		Service: "adservice",
		Value:   1000,
	}, domain.Prediction{
		Service: "cartservice",
		Value:   1000,
	}, domain.Prediction{
		Service: "checkoutservice",
		Value:   1000,
	}, domain.Prediction{
		Service: "currencyservice",
		Value:   1000,
	}, domain.Prediction{
		Service: "emailservice",
		Value:   1000,
	}, domain.Prediction{
		Service: "frontend",
		Value:   1000,
	}, domain.Prediction{
		Service: "paymentservice",
		Value:   1000,
	}, domain.Prediction{
		Service: "productcatalogservice",
		Value:   1000,
	}, domain.Prediction{
		Service: "recommendationservice",
		Value:   1000,
	}, domain.Prediction{
		Service: "shippingservice",
		Value:   1000,
	},
	}}

	if err := s.Pusher.PublishScalingPrediction(httpPredictions, s.Namespace); err != nil {
		return fmt.Errorf("error pushing predictions: %w", err)
	}

	return nil
}
