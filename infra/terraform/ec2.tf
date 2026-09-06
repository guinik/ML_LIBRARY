resource "aws_security_group" "training" {
  name        = "${local.project_tag}-sg"
  description = "Egress-only -- shell access goes through SSM Session Manager, no inbound ports needed."
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "allow all outbound (package installs, git clone, S3, SSM)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

locals {
  user_data = templatefile("${path.module}/templates/user_data.sh.tpl", {
    github_repo_url       = var.github_repo_url
    git_branch             = var.git_branch
    s3_bucket              = aws_s3_bucket.checkpoints.bucket
    auto_stop_after_hours  = var.auto_stop_after_hours
  })
}

# spot_instance_type="persistent" + interruption_behavior="stop": on reclaim the instance
# STOPS rather than terminating, so the EBS root volume (repo, build, checkpoint) survives
# and AWS auto-restarts it once spot capacity returns -- no S3 round-trip needed to resume,
# the ml-train systemd service (installed by user_data) just starts again on the next boot
# and EXECUTE resumes from ../tinystories.mlt on disk exactly as it does locally/on Colab.
resource "aws_instance" "training" {
  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.training.id]
  iam_instance_profile   = aws_iam_instance_profile.training.name
  key_name               = var.key_name

  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price                      = var.spot_max_price
      spot_instance_type             = "persistent"
      instance_interruption_behavior = "stop"
    }
  }

  user_data = local.user_data

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
  }

  tags = merge(local.common_tags, { Name = "${local.project_tag}-instance" })
}
