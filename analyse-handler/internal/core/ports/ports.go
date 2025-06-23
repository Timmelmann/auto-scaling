package ports

import (
	"context"

	"timmelmann/analyse-handler/internal/core/domain"
)

type MetricCollector interface {
	CollectMetrics(ctx context.Context) ([]domain.Metric, error)
}

type MetricsStorage interface {
	StoreHttpMetrics(metrics []domain.Metric) error
}

type MetricsPusher interface {
	PublishScalingPrediction(httpMetrics domain.ScalingPrediction, namespace string) error
}

type MetricsPredictor interface {
	PredictMetric(metrics []domain.Metric) (*domain.ScalingPrediction, error)
}
