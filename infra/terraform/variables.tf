variable "project_id" {
  description = "Google Cloud Project ID where the Travel Concierge Agent resources will be deployed."
  type        = string
}

variable "region" {
  description = "Primary Google Cloud region for Vertex AI Agent Runtime, Cloud Run, and Artifact Registry."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Base name for the deployed service, service account, and Artifact Registry repository."
  type        = string
  default     = "smart-travel-concierge"
}

variable "container_image" {
  description = "Container image URI pushed to Artifact Registry for Cloud Run deployment."
  type        = string
  default     = ""
}

variable "fast_model_name" {
  description = "Fast Gemini model for routing, intake, tool execution, and HITL booking."
  type        = string
  default     = "gemini-2.5-flash"
}

variable "planning_model_name" {
  description = "Reasoning Gemini model for multi-day itinerary synthesis and budget optimization."
  type        = string
  default     = "gemini-2.5-pro"
}

variable "enable_cloud_dlp" {
  description = "Enable Google Cloud Sensitive Data Protection (DLP) API in addition to local redaction."
  type        = bool
  default     = true
}

variable "max_single_trip_budget_usd" {
  description = "Hard enterprise single-trip policy budget cap in USD."
  type        = number
  default     = 25000
}
