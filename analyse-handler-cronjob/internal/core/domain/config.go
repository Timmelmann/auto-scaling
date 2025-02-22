package domain

type CustomQueryValues struct {
	Name   string   `json:"name"`
	Values []string `json:"values"`
}

type Service struct {
	Name       string `json:"name"`
	Deployment string `json:"deployment"`
}