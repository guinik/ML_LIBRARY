# NOTE: cost-allocation tags (here, the "Project" tag applied to every resource in this
# config) need a one-time manual activation in Billing -> Cost allocation tags before AWS
# will start honoring this filter, and newly-activated tags can take up to ~24h to appear
# in cost data. See infra/README.md.
resource "aws_budgets_budget" "training" {
  name         = "${local.project_tag}-budget"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_amount_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = [format("user:Project$%s", local.project_tag)]
  }

  dynamic "notification" {
    for_each = [50, 80, 100]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.notification_email]
    }
  }
}
