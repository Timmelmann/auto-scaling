package config

import "os"

type Config struct {
	PrometheusURL       string
	PushgatewayURL      string
	PredictorUrl        string
	JobName             string
	MetricsWindow       string
	OutputDir           string
	PredictedMetricName string
}

func LoadFromEnv() *Config {
	return &Config{
		PrometheusURL:       getEnvOrDefault("PROMETHEUS_URL", "http://127.0.0.1:59420"),
		PushgatewayURL:      getEnvOrDefault("PUSHGATEWAY_URL", "http://127.0.0.1:59578"),
		PredictorUrl:        getEnvOrDefault("PREDICTOR_URL", "http://127.0.0.1:59579"),
		JobName:             getEnvOrDefault("JOB_NAME", "metrics_collector"),
		MetricsWindow:       getEnvOrDefault("METRICS_WINDOW", "60s"),
		OutputDir:           getEnvOrDefault("OUTPUT_DIR", "."),
		PredictedMetricName: getEnvOrDefault("PREDICTION_METRIC_NAME", "webshop_service_predicted_request_count"),
	}
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
