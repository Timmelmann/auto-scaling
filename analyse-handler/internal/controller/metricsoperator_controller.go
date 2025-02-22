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

package controller

import (
	"context"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	metricsv1 "timmelmann/analyse-handler/api/v1"
	"timmelmann/analyse-handler/internal/adapters/predictor"
	metricCollector "timmelmann/analyse-handler/internal/adapters/prometheus/collector"
	"timmelmann/analyse-handler/internal/adapters/prometheus/pushgateway"
	"timmelmann/analyse-handler/internal/core/app"
	"timmelmann/analyse-handler/internal/core/domain"
)

// MetricsOperatorReconciler reconciles a MetricsOperator object
type MetricsOperatorReconciler struct {
	client.Client
	Scheme         *runtime.Scheme
	MetricsService *app.MetricsService
}

// +kubebuilder:rbac:groups=metrics.timmelmann.com,resources=metricsoperators,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=metrics.timmelmann.com,resources=metricsoperators/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=metrics.timmelmann.com,resources=metricsoperators/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the MetricsOperator object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.20.0/pkg/reconcile
func (r *MetricsOperatorReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = log.FromContext(ctx)

	// 1. Custom Resource laden
	var collector metricsv1.MetricsOperator
	if err := r.Get(ctx, req.NamespacedName, &collector); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if err := r.initIfNotPresent(collector); err != nil {
		return ctrl.Result{}, err
	}
	// 2. Business-Logik ausführen
	err := r.MetricsService.CollectAndAnalyze(ctx)
	if err != nil {
		return ctrl.Result{}, err
	}

	interval := collector.Spec.Interval
	if interval == "" {
		interval = "1m" // fallback default
	}

	repeateDuration, err := time.ParseDuration(interval)
	if err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{
		RequeueAfter: repeateDuration,
	}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *MetricsOperatorReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&metricsv1.MetricsOperator{}).
		Named("metricsoperator").
		Complete(r)
}

func (r *MetricsOperatorReconciler) initIfNotPresent(collector metricsv1.MetricsOperator) error {

	var customQueryValuesDomain []domain.CustomQueryValues
	for _, customQueryValue := range collector.Spec.Collector.CustomQuery.CustomQueryValues {
		customQueryValuesDomain = append(customQueryValuesDomain, domain.CustomQueryValues{
			Name:   customQueryValue.Name,
			Values: customQueryValue.Values,
		})
	}
	if r.MetricsService.Collector == nil {
		collectorAdapter, err := metricCollector.NewClient(
			collector.Spec.Collector.EndpointUrl,
			collector.Spec.Collector.MetricsWindow,
			collector.Spec.Collector.Step,
			collector.Spec.Collector.CustomQuery.Query,
			collector.Spec.Collector.CustomQuery.UseNameSpace,
			collector.Spec.Collector.CustomQuery.UserServiceNames,
			customQueryValuesDomain,
			collector.Spec.Collector.CustomQuery.Labels,
		)
		if err != nil {
			return err
		}
		r.MetricsService.Collector = collectorAdapter
	}

	if r.MetricsService.Predictor == nil {
		r.MetricsService.Predictor = predictor.NewClient(collector.Spec.Analyzer.EndpointUrl)
	}

	if r.MetricsService.Pusher == nil {
		r.MetricsService.Pusher = pushgateway.NewClient(collector.Spec.Pusher.EndpointUrl, collector.Spec.Pusher.MetricName, collector.Spec.Pusher.JobName)
	}

	if r.MetricsService.Services == nil {
		services := collector.Spec.Application.Services
		var servicesArray []domain.Service
		for _, service := range services {
			servicesArray = append(servicesArray, domain.Service{
				Name:       service.Name,
				Deployment: service.DeploymentName,
			})
		}
		r.MetricsService.Services = servicesArray
	}

	if r.MetricsService.Namespace == "" {
		r.MetricsService.Namespace = collector.Spec.Application.Namespace
	}

	return nil
}
