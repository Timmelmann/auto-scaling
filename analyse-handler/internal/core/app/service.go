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

	metricData, err := s.Collector.CollectMetrics(ctx)
	if err != nil {
		return fmt.Errorf("error fetching metric: %w", err)
	}

	httpPredictions, err := s.Predictor.PredictMetric(metricData)
	if err != nil {
		return fmt.Errorf("error analyzing metrics: %w", err)
	}

	if err := s.Pusher.PublishScalingPrediction(*httpPredictions, s.Namespace); err != nil {
		return fmt.Errorf("error pushing predictions: %w", err)
	}

	return nil
}
