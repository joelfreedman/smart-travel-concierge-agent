output "service_account_email" {
  description = "Least-privilege service account email bound to the agent runtime."
  value       = google_service_account.agent_runtime_sa.email
}

output "artifact_registry_repository" {
  description = "Docker image repository URI for container builds."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.agent_images.repository_id}"
}

output "staging_bucket_uri" {
  description = "GCS staging bucket URI for Vertex AI Agent Runtime deployments."
  value       = "gs://${google_storage_bucket.agent_runtime_staging.name}"
}

output "secret_manager_secret_id" {
  description = "Secret Manager resource ID for API key injection."
  value       = google_secret_manager_secret.gemini_api_key.secret_id
}

output "cloud_run_service_url" {
  description = "Deployed Cloud Run service URL."
  value       = google_cloud_run_v2_service.travel_concierge.uri
}
