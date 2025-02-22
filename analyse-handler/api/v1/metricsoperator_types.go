/*
Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// MetricsOperatorSpec defines the desired state of MetricsOperator.
type MetricsOperatorSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// Foo is an example field of MetricsOperator. Edit metricsoperator_types.go to remove/update
	Collector   Collector   `json:"collector"`
	Analyzer    Analyzer    `json:"analyzer"`
	Pusher      Pusher      `json:"pusher"`
	Interval    string      `json:"interval"`
	Application Application `json:"applications"`
}

type Application struct {
	// Name der Applikation
	Namespace string    `json:"namespace"`
	Services  []Service `json:"services"`
}

type Service struct {
	Name           string `json:"name"`
	DeploymentName string `json:"deploymentName"`
}

type Collector struct {
	EndpointUrl   string      `json:"endpointUrl"`
	MetricsWindow string      `json:"metricsWindow"`
	Step          string      `json:"step"`
	CustomQuery   CustomQuery `json:"customQuery,omitempty"`
}

type CustomQuery struct {
	Query             string              `json:"query"`
	UserServiceNames  bool                `json:"useServiceNames"`
	UseNameSpace      bool                `json:"useNamespace"`
	CustomQueryValues []CustomQueryValues `json:"customQueryValues"`
	Labels            []string            `json:"labels"`
}

type CustomQueryValues struct {
	Name   string   `json:"name"`
	Values []string `json:"values"`
}

type Analyzer struct {
	EndpointUrl string `json:"endpointUrl"`
}
type Pusher struct {
	EndpointUrl string `json:"endpointUrl"`
	MetricName  string `json:"metricName"`
	JobName     string `json:"jobName"`
}

// MetricsOperatorStatus defines the observed state of MetricsOperator.
type MetricsOperatorStatus struct {
	LastCollectionTime metav1.Time `json:"lastCollectionTime,omitempty"`

	// Aktueller Status des Collectors
	// +kubebuilder:validation:Enum=Starting;Running;Failed
	// +optional
	Phase string `json:"phase,omitempty"`

	// Fehlermeldung falls etwas schief ging
	// +optional
	Error string `json:"error,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// MetricsOperator is the Schema for the metricsoperators API.
type MetricsOperator struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   MetricsOperatorSpec   `json:"spec,omitempty"`
	Status MetricsOperatorStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// MetricsOperatorList contains a list of MetricsOperator.
type MetricsOperatorList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []MetricsOperator `json:"items"`
}

func init() {
	SchemeBuilder.Register(&MetricsOperator{}, &MetricsOperatorList{})
}
