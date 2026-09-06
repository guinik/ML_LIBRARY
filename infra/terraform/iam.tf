data "aws_iam_policy_document" "assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "training" {
  name               = "${local.project_tag}-role"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
  tags               = local.common_tags
}

# SSM Session Manager access -- no open inbound port 22 needed
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.training.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "checkpoint_bucket_access" {
  statement {
    actions = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.checkpoints.arn,
      "${aws_s3_bucket.checkpoints.arn}/*",
    ]
  }
}

resource "aws_iam_role_policy" "checkpoint_bucket_access" {
  name   = "${local.project_tag}-s3-access"
  role   = aws_iam_role.training.id
  policy = data.aws_iam_policy_document.checkpoint_bucket_access.json
}

resource "aws_iam_instance_profile" "training" {
  name = "${local.project_tag}-profile"
  role = aws_iam_role.training.name
}
