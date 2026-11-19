locals {
  required_apis = toset([
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudtrace.googleapis.com",
    "dlp.googleapis.com",
    "iam.googleapis.com",
    "logging.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
  ])

  effective_image = (
    var.container_image != ""
    ? var.container_image
    : "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.agent_images.repository_id}/${var.service_name}:latest"
  )
}

# 1. Enable required Google Cloud APIs
resource "google_project_service" "required" {
  for_each           = local.required_apis
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# 2. Dedicated least-privilege runtime Service Account
resource "google_service_account" "agent_runtime_sa" {
  account_id   = "${var.service_name}-sa"
  display_name = "Smart Travel Concierge Agent Runtime Service Account"
  description  = "Least-privilege identity for Vertex AI, Secret Manager, Cloud DLP, and Cloud Trace."
  depends_on   = [google_project_service.required]
}

locals {
  agent_iam_roles = toset([
    "roles/aiplatform.user",
    "roles/cloudtrace.agent",
    "roles/dlp.user",
    "roles/logging.logWriter",
  ])
}

resource "google_project_iam_member" "agent_roles" {
  for_each = local.agent_iam_roles
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.agent_runtime_sa.email}"
}

# 3. Secret Manager container for optional API key (populated out-of-band, NEVER in HCL)
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "${var.service_name}-gemini-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret_iam_member" "agent_secret_accessor" {
  secret_id = google_secret_manager_secret.gemini_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent_runtime_sa.email}"
}

# 4. Artifact Registry repository for container builds
resource "google_artifact_registry_repository" "agent_images" {
  location      = var.region
  repository_id = "${var.service_name}-repo"
  description   = "Container images for the Smart Travel & Trip Budget Concierge Agent."
  format        = "DOCKER"

  depends_on = [google_project_service.required]
}

# 5. GCS staging bucket for Vertex AI Agent Runtime deployments (agents-cli / adk deploy)
resource "google_storage_bucket" "agent_runtime_staging" {
  name                        = "${var.project_id}-${var.service_name}-staging"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_storage_bucket_iam_member" "agent_staging_access" {
  bucket = google_storage_bucket.agent_runtime_staging.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.agent_runtime_sa.email}"
}

# 6. Cloud Run v2 Service hosting the FastAPI + ADK Travel Concierge wrapper
resource "google_cloud_run_v2_service" "travel_concierge" {
  name                = var.service_name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.agent_runtime_sa.email
    scaling {
      min_instance_count = 0
      max_instance_count = 5
    }

    containers {
      image = local.effective_image

      ports {
        container_port = 8080
      }

      env {
        name  = "APP_ENV"
        value = "production"
      }
      env {
        name  = "APP_HOST"
        value = "0.0.0.0"
      }
      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
      env {
        name  = "FAST_MODEL_NAME"
        value = var.fast_model_name
      }
      env {
        name  = "PLANNING_MODEL_NAME"
        value = var.planning_model_name
      }
      env {
        name  = "ENABLE_CLOUD_DLP"
        value = tostring(var.enable_cloud_dlp)
      }
      env {
        name  = "OTEL_EXPORT_CLOUD_TRACE"
        value = "true"
      }
      env {
        name  = "MAX_SINGLE_TRIP_BUDGET_USD"
        value = tostring(var.max_single_trip_budget_usd)
      }
      env {
        name  = "GEMINI_API_KEY_SECRET_NAME"
        value = google_secret_manager_secret.gemini_api_key.secret_id
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "1Gi"
        }
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
      }
    }
  }

  depends_on = [
    google_project_service.required,
    google_project_iam_member.agent_roles,
    google_secret_manager_secret_iam_member.agent_secret_accessor,
  ]
}
