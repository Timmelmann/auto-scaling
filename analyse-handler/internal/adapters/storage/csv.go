package storage

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"timmelmann/analyse-handler/internal/core/domain"
)

type CSVStorage struct {
	outputDir string
}

func NewCSVStorage(outputDir string) *CSVStorage {
	return &CSVStorage{
		outputDir: outputDir,
	}
}

func (s *CSVStorage) StoreMetrics(metrics []domain.Metric) error {
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("http_%s.csv", timestamp)
	filepath := filepath.Join(s.outputDir, filename)

	file, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("error creating HTTP metrics file: %w", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	var header = make([]string, len(metrics[0].Labels)+2)
	header = append(header, "timestamp", "value")
	for _, label := range metrics[0].Labels {
		header = append(header, label)
	}

	if err := writer.Write(header); err != nil {
		return fmt.Errorf("error writing HTTP metrics header: %w", err)
	}
	// Write data
	for _, metric := range metrics {
		row := []string{strconv.FormatInt(metric.Timestamp, 64), strconv.FormatFloat(metric.Value, 'f', -1, 64)}
		for _, label := range metric.Labels {
			row = append(row, metric.Labels[label])
		}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("error writing HTTP metric row: %w", err)
		}
	}
	return nil
}
