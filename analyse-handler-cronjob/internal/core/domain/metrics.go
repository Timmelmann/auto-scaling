package domain

type ScalingPrediction struct {
	Predictions []Prediction `json:"predictions"`
}

type Prediction struct {
	Service string  `json:"service"`
	Value   float64 `json:"value"`
}



type Metric struct {
	Timestamp int64             `json:"timestamp"`
	Value     float64           `json:"value"`
	Labels    map[string]string `json:"labels"`
}


