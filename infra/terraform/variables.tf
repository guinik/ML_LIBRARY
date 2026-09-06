variable "region" {
  description = "AWS region to launch the training instance in."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "GPU spot instance type."
  type        = string
  default     = "g5.xlarge"
}

variable "spot_max_price" {
  description = "Maximum hourly bid (USD) for the spot instance. Leave null to default to the on-demand price cap."
  type        = string
  default     = null
}

variable "github_repo_url" {
  description = "Git URL of the ML_LIBRARY repo to clone on the instance."
  type        = string
  default     = "https://github.com/guinik/ML_LIBRARY.git"
}

variable "git_branch" {
  description = "Branch to check out on the instance."
  type        = string
  default     = "main"
}

variable "notification_email" {
  description = "Email address for AWS Budgets cost alerts. Required on purpose (no default) -- this spends real money."
  type        = string
}

variable "budget_amount_usd" {
  description = "Monthly budget threshold (USD) that triggers alert emails at 50/80/100% of this amount."
  type        = number
  default     = 90
}

variable "auto_stop_after_hours" {
  description = "Hard safety net: the instance self-issues `shutdown -h now` after this many cumulative *running* hours. Scheduled via `at`, which only counts down while the instance is powered on, so spot stop/resume cycles pause the clock instead of racing it."
  type        = number
  default     = 90
}

variable "key_name" {
  description = "Optional EC2 key pair name for SSH. Not required -- Session Manager (SSM) is the primary access path and needs no open inbound port."
  type        = string
  default     = null
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB (repo, build, dataset, and checkpoint all live here)."
  type        = number
  default     = 100
}
