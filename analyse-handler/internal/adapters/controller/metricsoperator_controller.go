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
	"fmt"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	metricsv1 "timmelmann/analyse-handler/api/v1"
	"timmelmann/analyse-handler/internal/adapters/predictor"
	metricCollector "timmelmann/analyse-handler/internal/adapters/prometheus/collector"
	"timmelmann/analyse-handler/internal/adapters/prometheus/pushgateway"
	"timmelmann/analyse-handler/internal/core/app"
	"timmelmann/analyse-handler/internal/core/domain"
)

const finalizerName = "metrics.timmelmann.com/finalizer"

// MetricsOperatorReconciler reconciles a MetricsOperator object
type MetricsOperatorReconciler struct {
	client.Client
	Scheme         *runtime.Scheme
	MetricsService *app.MetricsService
}

// +kubebuilder:rbac:groups=metrics.timmelmann.com,resources=metricsoperators,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=metrics.timmelmann.com,resources=metricsoperators/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=metrics.timmelmann.com,resources=metricsoperators/finalizers,verbs=update
// +kubebuilder:rbac:groups=autoscaling,resources=horizontalpodautoscalers,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch
// +kubebuilder:rbac:groups=external.metrics.k8s.io,resources=*,verbs=get;list
// +kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch

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
	logger := log.FromContext(ctx)
	logger.Info("Starting Reconcile", "request", req.NamespacedName)

	var collector metricsv1.MetricsOperator
	if err := r.Get(ctx, req.NamespacedName, &collector); err != nil {
		if errors.IsNotFound(err) {
			// The resource is deleted, but we can still try to clean up HPAs based on labels
			logger.Info("MetricsOperator resource not found, may have been deleted")

			// Attempt to find any HPAs with our labels across all namespaces
			hpaList := &autoscalingv2.HorizontalPodAutoscalerList{}
			if listErr := r.List(ctx, hpaList,
				client.MatchingLabels{
					"app.kubernetes.io/managed-by": "metrics-operator",
					"created-by-namespace":         req.Namespace,
					"metrics-operator-name":        req.Name,
				}); listErr == nil {
				for _, hpa := range hpaList.Items {
					logger.Info("Found orphaned HPA to clean up",
						"name", hpa.Name,
						"namespace", hpa.Namespace)
					if delErr := r.Delete(ctx, &hpa); delErr != nil && !errors.IsNotFound(delErr) {
						logger.Error(delErr, "Failed to delete orphaned HPA",
							"name", hpa.Name,
							"namespace", hpa.Namespace)
					}
				}
			}

			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get MetricsOperator")
		return ctrl.Result{}, err
	}

	// Check if this is a deletion request and handle finalizers
	result, err := r.handleDeletion(ctx, &collector)
	if err != nil || result != (ctrl.Result{}) {
		return result, err
	}

	// Reset failure state if previously failed
	if collector.Status.RetryCount > 0 && collector.Status.FailureReason != "" {
		if err := r.resetFailureState(ctx, &collector); err != nil {
			logger.Error(err, "Failed to reset failure state")
			return ctrl.Result{Requeue: true}, err
		}
	}

	// Validate configuration
	if collector.Spec.Application.Namespace == "" {
		logger.Error(nil, "Target namespace is not specified")
		return r.handleFailure(ctx, &collector, "ConfigurationFailure",
			fmt.Errorf("target namespace not specified in metrics operator config"))
	}

	// Initialize services if not already done
	if err := r.initIfNotPresent(collector); err != nil {
		logger.Error(err, "Failed to initialize services")
		return r.handleFailure(ctx, &collector, "InitializationFailure", err)
	}

	if r.MetricsService.Services == nil || len(r.MetricsService.Services) == 0 {
		logger.Error(nil, "No services configured")
		return r.handleFailure(ctx, &collector, "ConfigurationFailure",
			fmt.Errorf("no services configured for metrics operator"))
	}

	// Collect and analyze metrics
	if err := r.MetricsService.CollectAndAnalyze(ctx); err != nil {
		logger.Error(err, "Failed to collect and analyze metrics")
		return r.handleFailure(ctx, &collector, "MetricsAnalysisFailure", err)
	}

	// Create/update HPAs for each service
	hpaCreationErrors := false
	for _, service := range r.MetricsService.Services {
		if err := r.createHPA(ctx, collector.Spec.Application.Namespace,
			collector.Spec.Pusher.MetricName, service, &collector); err != nil {
			logger.Error(err, "Failed to create/update HPA for service",
				"service", service.Name,
				"namespace", collector.Spec.Application.Namespace)
			hpaCreationErrors = true
			// Continue with other services instead of returning immediately
		} else {
			logger.Info("HPA created/updated successfully",
				"service", service.Name,
				"namespace", collector.Spec.Application.Namespace)
		}
	}

	// Update status
	if hpaCreationErrors {
		return r.handleFailure(ctx, &collector, "HPACreationFailure",
			fmt.Errorf("failed to create/update one or more HPAs"))
	}

	collector.Status.Phase = "Running"
	collector.Status.LastCollectionTime = metav1.Now()
	if err := r.Status().Update(ctx, &collector); err != nil {
		logger.Error(err, "Failed to update status")
		return ctrl.Result{Requeue: true}, err
	}

	// Schedule next reconciliation
	interval := collector.Spec.Interval
	if interval == "" {
		interval = "1m"
		logger.Info("No interval specified, using default", "interval", interval)
	}

	repeatDuration, err := time.ParseDuration(interval)
	if err != nil {
		logger.Error(err, "Failed to parse interval", "interval", interval)
		return ctrl.Result{}, err
	}

	return ctrl.Result{
		RequeueAfter: repeatDuration,
	}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *MetricsOperatorReconciler) SetupWithManager(mgr ctrl.Manager) error {
	// Create a logger
	setupLog := log.Log.WithName("setup")

	// Add watches for HPAs that we need to manage
	setupLog.Info("Setting up controller with HPA watching")

	return ctrl.NewControllerManagedBy(mgr).
		For(&metricsv1.MetricsOperator{}).
		// Watch HPAs with specific labels to catch external deletion events
		Watches(
			&autoscalingv2.HorizontalPodAutoscaler{},
			handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []reconcile.Request {
				hpa := obj.(*autoscalingv2.HorizontalPodAutoscaler)
				logger := log.FromContext(ctx)

				// Skip if this HPA is not managed by our operator
				if hpa.Labels["app.kubernetes.io/managed-by"] != "metrics-operator" {
					return nil
				}

				// Get the metrics operator name and namespace from labels
				operatorName := hpa.Labels["metrics-operator-name"]
				operatorNamespace := hpa.Labels["created-by-namespace"]

				// Check if this is a deletion event
				if hpa.DeletionTimestamp != nil {
					logger.Info("Detected HPA deletion event",
						"hpa", hpa.Name,
						"namespace", hpa.Namespace,
						"operator", operatorName,
						"operatorNamespace", operatorNamespace)

					// Only handle if we have enough information to find the parent
					if operatorName != "" && operatorNamespace != "" {
						return []reconcile.Request{
							{NamespacedName: types.NamespacedName{
								Name:      operatorName,
								Namespace: operatorNamespace,
							}},
						}
					}
				}

				return nil
			}),
		).
		Named("metricsoperator").
		Complete(r)
}

func (r *MetricsOperatorReconciler) initIfNotPresent(collector metricsv1.MetricsOperator) error {
	var services []domain.Service
	for _, service := range collector.Spec.Application.Services {
		services = append(services, domain.Service{
			Name:        service.Name,
			Deployment:  service.DeploymentName,
			MinReplicas: service.MinReplicas,
			MaxReplicas: service.MaxReplicas,
			TargetValue: service.TargetValue,
			TargetType:  service.TargetType,
		})
	}
	if r.MetricsService.Collector == nil {
		collectorAdapter, err := metricCollector.NewClient(metricCollector.ClientConfig{
			Url:          collector.Spec.Collector.EndpointUrl,
			MetricWindow: collector.Spec.Collector.MetricsWindow,
			StepDuration: collector.Spec.Collector.Step,
			Query:        collector.Spec.Collector.CustomQuery.Query,
			Namespace:    collector.Spec.Application.Namespace,
			Services:     services,
			Labels:       collector.Spec.Collector.CustomQuery.Labels,
		})
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
				Name:        service.Name,
				Deployment:  service.DeploymentName,
				MinReplicas: service.MinReplicas,
				MaxReplicas: service.MaxReplicas,
				TargetValue: service.TargetValue,
				TargetType:  service.TargetType,
			})
		}
		r.MetricsService.Services = servicesArray
	}

	if r.MetricsService.Namespace == "" {
		r.MetricsService.Namespace = collector.Spec.Application.Namespace
	}

	return nil
}

func (r *MetricsOperatorReconciler) resetFailureState(ctx context.Context, collector *metricsv1.MetricsOperator) error {
	if collector.Status.RetryCount > 0 {
		collector.Status.RetryCount = 0
		collector.Status.FailureReason = ""
		collector.Status.FailureMessage = ""
		collector.Status.LastFailureTime = ""

		return r.Status().Update(ctx, collector)
	}
	return nil
}

func (r *MetricsOperatorReconciler) handleFailure(ctx context.Context, controller *metricsv1.MetricsOperator, reason string, err error) (ctrl.Result, error) {

	controller.Status.RetryCount++
	controller.Status.LastFailureTime = time.Now().Format(time.RFC3339)
	controller.Status.FailureReason = reason

	if err != nil {
		controller.Status.FailureMessage = err.Error()
	}

	maxRetries := 5
	if controller.Spec.MaxRetries > 0 {
		maxRetries = controller.Spec.MaxRetries
	}

	if controller.Status.RetryCount >= maxRetries {
		controller.Status.FailureMessage = fmt.Sprintf("%s: max retries (%d) exceeded",
			controller.Status.FailureMessage, maxRetries)
		if updateErr := r.Status().Update(ctx, controller); updateErr != nil {
			return ctrl.Result{}, updateErr
		}
		return ctrl.Result{}, nil
	}

	controller.Status.Phase = "Failed"
	controller.Status.Error = err.Error()
	if updateErr := r.Status().Update(ctx, controller); updateErr != nil {
		return ctrl.Result{Requeue: true}, updateErr
	}

	delay := time.Duration(1<<uint(controller.Status.RetryCount)) * time.Second
	if delay > 5*time.Minute {
		delay = 5 * time.Minute
	}

	return ctrl.Result{RequeueAfter: delay}, nil
}

func (r *MetricsOperatorReconciler) createHPA(ctx context.Context, namespace, targetMetricName string, service domain.Service, controller *metricsv1.MetricsOperator) error {
	logger := log.FromContext(ctx)

	logger.Info("Attempting to create/update HPA in target namespace",
		"service", service.Name,
		"deployment", service.Deployment,
		"targetNamespace", namespace,
		"minReplicas", service.MinReplicas,
		"maxReplicas", service.MaxReplicas,
		"targetValue", service.TargetValue,
		"targetType", service.TargetType,
		"metricName", targetMetricName,
		"controllerNamespace", controller.Namespace)

	if namespace == "" {
		return fmt.Errorf("target namespace cannot be empty")
	}
	if service.Deployment == "" {
		return fmt.Errorf("deployment name cannot be empty for service %s", service.Name)
	}
	if targetMetricName == "" {
		return fmt.Errorf("target metric name cannot be empty")
	}

	deploymentKey := types.NamespacedName{
		Name:      service.Deployment,
		Namespace: namespace,
	}
	deployment := &appsv1.Deployment{}
	if err := r.Get(ctx, deploymentKey, deployment); err != nil {
		if errors.IsNotFound(err) {
			logger.Error(err, "Target deployment not found in namespace",
				"deployment", service.Deployment,
				"namespace", namespace)
			return fmt.Errorf("deployment %s not found in namespace %s", service.Deployment, namespace)
		}
		logger.Error(err, "Error checking deployment existence",
			"deployment", service.Deployment,
			"namespace", namespace)
		return fmt.Errorf("error checking deployment %s in namespace %s: %w", service.Deployment, namespace, err)
	}

	hpaName := service.Name + "-hpa"
	hpa := &autoscalingv2.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      hpaName,
			Namespace: namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "metrics-operator",
				"app.kubernetes.io/part-of":    controller.Name,
				"app.kubernetes.io/name":       hpaName,
				"created-by-namespace":         controller.Namespace,
				"metrics-operator-name":        controller.Name,        // Add additional identifying labels
				"metrics-operator-uuid":        string(controller.UID), // Add UID as a string for better tracking
			},
			// Don't add owner references when controller and resource are in different namespaces
			// Instead, we'll use the labels above and finalizers for cleanup
		},
		Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       service.Deployment,
			},
			MinReplicas: &service.MinReplicas,
			MaxReplicas: service.MaxReplicas,
			Metrics: []autoscalingv2.MetricSpec{
				{
					Type: autoscalingv2.ExternalMetricSourceType,
					External: &autoscalingv2.ExternalMetricSource{
						Metric: autoscalingv2.MetricIdentifier{
							Name: targetMetricName,
							Selector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"service":   service.Name,
									"namespace": namespace,
								},
							},
						},
					},
				},
			},
		},
	}

	// Set protection annotation to prevent inadvertent deletion
	if hpa.Annotations == nil {
		hpa.Annotations = make(map[string]string)
	}
	hpa.Annotations["metrics.timmelmann.com/protected"] = "true"

	switch strings.ToLower(service.TargetType) {
	case "value", "":
		hpa.Spec.Metrics[0].External.Target = autoscalingv2.MetricTarget{
			Type:  autoscalingv2.ValueMetricType,
			Value: resource.NewQuantity(int64(service.TargetValue), resource.DecimalSI),
		}
	case "averagevalue":
		hpa.Spec.Metrics[0].External.Target = autoscalingv2.MetricTarget{
			Type:         autoscalingv2.AverageValueMetricType,
			AverageValue: resource.NewQuantity(int64(service.TargetValue), resource.DecimalSI),
		}
	default:
		return fmt.Errorf("unsupported target type: %s", service.TargetType)
	}

	existing := &autoscalingv2.HorizontalPodAutoscaler{}
	err := r.Get(ctx, types.NamespacedName{Name: hpa.Name, Namespace: namespace}, existing)
	if err != nil {
		if errors.IsNotFound(err) {
			// Create new HPA if it doesn't exist
			logger.Info("Creating new HPA in target namespace",
				"name", hpa.Name,
				"namespace", namespace)
			if err = r.Create(ctx, hpa); err != nil {
				logger.Error(err, "Failed to create HPA in target namespace",
					"name", hpa.Name,
					"namespace", namespace,
					"error", err.Error())
				return fmt.Errorf("failed to create HPA in namespace %s: %w", namespace, err)
			}
			logger.Info("Successfully created HPA in target namespace",
				"name", hpa.Name,
				"namespace", namespace)
			return nil
		}
		logger.Error(err, "Failed to get existing HPA in target namespace",
			"name", hpa.Name,
			"namespace", namespace,
			"error", err.Error())
		return fmt.Errorf("failed to get existing HPA in namespace %s: %w", namespace, err)
	}

	logger.Info("Updating existing HPA in target namespace",
		"name", existing.Name,
		"namespace", existing.Namespace)

	// Check if HPA is managed by our operator before updating
	if val, exists := existing.Labels["app.kubernetes.io/managed-by"]; !exists || val != "metrics-operator" {
		logger.Info("HPA exists but is not managed by metrics-operator, skipping update",
			"name", existing.Name,
			"namespace", existing.Namespace)
		return nil
	}

	// Save current resource version to avoid conflicts
	resourceVersion := existing.ResourceVersion

	// Update HPA specification
	existing.Spec.MinReplicas = hpa.Spec.MinReplicas
	existing.Spec.MaxReplicas = hpa.Spec.MaxReplicas
	existing.Spec.Metrics = hpa.Spec.Metrics
	existing.ResourceVersion = resourceVersion

	// Update labels
	if existing.Labels == nil {
		existing.Labels = make(map[string]string)
	}
	for key, value := range hpa.Labels {
		existing.Labels[key] = value
	}

	// Update annotations
	if existing.Annotations == nil {
		existing.Annotations = make(map[string]string)
	}
	for key, value := range hpa.Annotations {
		existing.Annotations[key] = value
	}

	if err = r.Update(ctx, existing); err != nil {
		logger.Error(err, "Failed to update HPA in target namespace",
			"name", existing.Name,
			"namespace", existing.Namespace,
			"error", err.Error())
		return fmt.Errorf("failed to update HPA in namespace %s: %w", namespace, err)
	}

	logger.Info("Successfully updated HPA in target namespace",
		"name", existing.Name,
		"namespace", existing.Namespace)
	return nil
}

// func (r *MetricsOperatorReconciler) addFinalizer(ctx context.Context, m *metricsv1.MetricsOperator) error {
// 	logger := log.FromContext(ctx)

// 	if !containsString(m.GetFinalizers(), finalizerName) {
// 		logger.Info("Adding Finalizer to resource", "name", m.Name)
// 		m.SetFinalizers(append(m.GetFinalizers(), finalizerName))
// 		return r.Update(ctx, m)
// 	}
// 	return nil
// }

// handleDeletion manages the finalizer and cleanup when a MetricsOperator is being deleted
// handleDeletion manages the finalizer and cleanup when a MetricsOperator is being deleted
func (r *MetricsOperatorReconciler) handleDeletion(ctx context.Context, m *metricsv1.MetricsOperator) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Check if the MetricsOperator instance is being deleted
	if m.GetDeletionTimestamp() != nil {
		isForceDelete := m.GetDeletionGracePeriodSeconds() != nil && *m.GetDeletionGracePeriodSeconds() == 0
		logger.Info("MetricsOperator is being deleted",
			"name", m.Name,
			"namespace", m.Namespace,
			"force", isForceDelete)

		// Make a deep copy of the object for status updates to avoid conflicts with finalizer updates
		statusCopy := m.DeepCopy()
		if statusCopy.Status.Phase != "Deleting" {
			statusCopy.Status.Phase = "Deleting"
			// We try updating status but continue regardless
			if statusErr := r.Status().Update(ctx, statusCopy); statusErr != nil {
				logger.Error(statusErr, "Failed to update status to Deleting, continuing with cleanup anyway")
			} else {
				// If status update succeeded, update our working copy
				m.Status.Phase = "Deleting"
			}
		}

		// If our finalizer is present, we need to clean up resources before removing it
		if containsString(m.GetFinalizers(), finalizerName) {
			logger.Info("Performing cleanup before deletion", "name", m.Name)

			var cleanupErr error
			// Attempt to clean up all HPAs created by this controller with timeout
			cleanupCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
			defer cancel()

			cleanupErr = r.cleanupExternalResources(cleanupCtx, m)

			// Handle cleanup results
			if cleanupErr != nil {
				logger.Error(cleanupErr, "Failed to clean up external resources")

				// For force delete or if the context timed out, proceed with finalizer removal anyway
				if isForceDelete || isTimeoutError(cleanupErr) || cleanupCtx.Err() == context.DeadlineExceeded {
					logger.Info("Force delete or cleanup timeout detected, removing finalizer despite cleanup issues")
				} else {
					// For normal deletes with errors that are not timeouts, retry cleanup
					return ctrl.Result{RequeueAfter: time.Second * 5}, cleanupErr
				}
			} else {
				logger.Info("Cleanup completed successfully")
			}

			// Remove the finalizer once cleanup is done or in force delete case
			logger.Info("Removing finalizer", "name", m.Name)
			finalizerPatch := client.MergeFrom(m.DeepCopy())
			m.SetFinalizers(removeString(m.GetFinalizers(), finalizerName))

			if err := r.Patch(ctx, m, finalizerPatch); err != nil {
				logger.Error(err, "Failed to remove finalizer")

				// For force deletes, continue even if we can't remove the finalizer
				if isForceDelete {
					logger.Info("Force delete detected, continuing despite finalizer update error")
					return ctrl.Result{}, nil
				}

				// For normal deletes, retry finalizer removal
				return ctrl.Result{RequeueAfter: time.Second * 5}, err
			}

			logger.Info("Finalizer removed successfully", "name", m.Name)
		} else {
			logger.Info("No finalizer present, nothing to clean up", "name", m.Name)
		}

		// Resource is being deleted and either has no finalizer or we've removed it
		return ctrl.Result{}, nil
	}

	// Resource is not being deleted, ensure it has our finalizer
	if !containsString(m.GetFinalizers(), finalizerName) {
		logger.Info("Adding finalizer", "name", m.Name)

		finalizerPatch := client.MergeFrom(m.DeepCopy())
		m.SetFinalizers(append(m.GetFinalizers(), finalizerName))

		if err := r.Patch(ctx, m, finalizerPatch); err != nil {
			logger.Error(err, "Failed to add finalizer")
			return ctrl.Result{RequeueAfter: time.Second * 5}, err
		}

		logger.Info("Finalizer added successfully", "name", m.Name)
	}

	// Continue with normal reconciliation
	return ctrl.Result{}, nil
}

// cleanupExternalResources finds and deletes all HPAs created by this controller instance
func (r *MetricsOperatorReconciler) cleanupExternalResources(ctx context.Context, m *metricsv1.MetricsOperator) error {
	logger := log.FromContext(ctx)
	logger.Info("Cleaning up external resources for MetricsOperator",
		"name", m.Name,
		"namespace", m.Namespace)

	targetNamespace := m.Spec.Application.Namespace
	if targetNamespace == "" {
		// If target namespace is empty, try to find HPAs in all namespaces
		logger.Info("No target namespace specified, looking for HPAs in all namespaces")

		// To search across all namespaces, we'll use separate client calls
		namespaceList := &corev1.NamespaceList{}
		if err := r.List(ctx, namespaceList); err != nil {
			return fmt.Errorf("failed to list namespaces: %w", err)
		}

		totalDeleted := 0
		for _, ns := range namespaceList.Items {
			deleted, err := r.cleanupHPAsInNamespace(ctx, m, ns.Name)
			if err != nil {
				logger.Error(err, "Error cleaning HPAs in namespace", "namespace", ns.Name)
				// Continue with other namespaces
			}
			totalDeleted += deleted
		}

		logger.Info("Completed cleanup across all namespaces", "totalDeleted", totalDeleted)
		return nil
	}

	// If we have a specific target namespace, clean up HPAs there
	deleted, err := r.cleanupHPAsInNamespace(ctx, m, targetNamespace)
	if err != nil {
		return err
	}

	logger.Info("Completed cleanup in target namespace",
		"namespace", targetNamespace,
		"deletedCount", deleted)
	return nil
}

// cleanupHPAsInNamespace finds and deletes HPAs in a specific namespace
func (r *MetricsOperatorReconciler) cleanupHPAsInNamespace(ctx context.Context, m *metricsv1.MetricsOperator, namespace string) (int, error) {
	logger := log.FromContext(ctx)
	logger.Info("Looking for HPAs to clean up",
		"namespace", namespace,
		"controllerName", m.Name,
		"controllerNamespace", m.Namespace)

	// Find HPAs in this namespace that match our labels
	hpaList := &autoscalingv2.HorizontalPodAutoscalerList{}
	if err := r.List(ctx, hpaList,
		client.InNamespace(namespace),
		client.MatchingLabels{
			"app.kubernetes.io/managed-by": "metrics-operator",
			"metrics-operator-name":        m.Name,
			"created-by-namespace":         m.Namespace,
		}); err != nil {
		return 0, fmt.Errorf("failed to list HPAs in namespace %s: %w", namespace, err)
	}

	logger.Info("Found HPAs to delete",
		"namespace", namespace,
		"count", len(hpaList.Items))

	deletedCount := 0
	for _, hpa := range hpaList.Items {
		logger.Info("Deleting HPA",
			"name", hpa.Name,
			"namespace", hpa.Namespace,
			"labels", hpa.Labels)

		// Remove protection annotation if present
		if hpa.Annotations != nil && hpa.Annotations["metrics.timmelmann.com/protected"] == "true" {
			patch := client.MergeFrom(hpa.DeepCopy())
			delete(hpa.Annotations, "metrics.timmelmann.com/protected")
			if err := r.Patch(ctx, &hpa, patch); err != nil {
				if !errors.IsNotFound(err) {
					logger.Error(err, "Failed to remove protection annotation",
						"name", hpa.Name,
						"namespace", hpa.Namespace)
					// Continue with deletion attempt even if patch fails
				}
			}
		}

		// Delete the HPA with background propagation policy to ensure it's deleted even if it has finalizers
		deleteOptions := client.DeleteOptions{
			PropagationPolicy: func() *metav1.DeletionPropagation {
				policy := metav1.DeletePropagationBackground
				return &policy
			}(),
		}

		if err := r.Delete(ctx, &hpa, &deleteOptions); err != nil {
			if !errors.IsNotFound(err) {
				logger.Error(err, "Failed to delete HPA",
					"name", hpa.Name,
					"namespace", hpa.Namespace)
				// Continue with other HPAs
			} else {
				logger.Info("HPA already deleted",
					"name", hpa.Name,
					"namespace", hpa.Namespace)
			}
		} else {
			logger.Info("Successfully deleted HPA",
				"name", hpa.Name,
				"namespace", hpa.Namespace)
			deletedCount++
		}

		// Wait a moment to avoid overwhelming the API server if there are many HPAs
		time.Sleep(100 * time.Millisecond)
	}

	return deletedCount, nil
}

// Helper functions for finalizer handling
func containsString(slice []string, s string) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
	}
	return false
}

func removeString(slice []string, s string) []string {
	result := make([]string, 0, len(slice))
	for _, item := range slice {
		if item != s {
			result = append(result, item)
		}
	}
	return result
}

func isTimeoutError(err error) bool {
	if err == nil {
		return false
	}

	// Check if it's a context timeout
	if err == context.DeadlineExceeded {
		return true
	}

	// Check error string for common timeout indicators
	errStr := err.Error()
	return strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "timed out") ||
		strings.Contains(errStr, "deadline exceeded")
}
