output "instance_id" {
  description = "EC2 instance ID."
  value       = aws_instance.training.id
}

output "ssm_connect_command" {
  description = "Open a shell on the instance -- no SSH key or open port needed."
  value       = "aws ssm start-session --target ${aws_instance.training.id} --region ${var.region}"
}

output "watch_training_log_command" {
  description = "Run after connecting via SSM to tail the live training log."
  value       = "sudo journalctl -u ml-train -f"
}

output "checkpoint_bucket" {
  description = "S3 bucket receiving periodic checkpoint/log syncs and the final model."
  value       = aws_s3_bucket.checkpoints.bucket
}

output "download_checkpoint_command" {
  description = "Pull the latest checkpoint down to your machine."
  value       = "aws s3 cp s3://${aws_s3_bucket.checkpoints.bucket}/checkpoints/tinystories.mlt ./tinystories.mlt"
}
