provider "aws" {
  region = var.region
}

locals {
  project_tag = "ml-library-training"
  common_tags = {
    Project = local.project_tag
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Deep Learning Base AMI (GPU, Ubuntu) ships with the NVIDIA driver + CUDA toolkit + nvcc
# preinstalled, so user_data doesn't need to install CUDA from scratch.
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}
