package domain

// type TargetType string

const (
	TargetTypeValue        string = "value"
	TargetTypeAverageValue string = "averagevalue"
)

type Service struct {
	Name        string     `json:"name"`
	Deployment  string     `json:"deployment"`
	MinReplicas int32      `json:"minReplicas"`
	MaxReplicas int32      `json:"maxReplicas"`
	TargetValue int32      `json:"targetValue"`
	TargetType  string `json:"targetType"`
}
