#!/bin/bash
set -euxo pipefail

REPO_DIR=/home/ubuntu/ML_LIBRARY
S3_BUCKET=${s3_bucket}
GIT_REPO=${github_repo_url}
GIT_BRANCH=${git_branch}
AUTO_STOP_HOURS=${auto_stop_after_hours}

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git cmake build-essential python3-pip python3-venv unzip at

systemctl enable --now atd

# runs once on first boot only (standard EC2 user_data behavior) -- the ml-train systemd
# service installed below is what re-launches training on every *subsequent* boot, e.g.
# after a spot interruption stops and later restarts this instance
if [ ! -d "$REPO_DIR" ]; then
  sudo -u ubuntu git clone --branch "$GIT_BRANCH" "$GIT_REPO" "$REPO_DIR"
fi
cd "$REPO_DIR"

sudo -u ubuntu python3 -m pip install --user -r requirements.txt

# build_vocab.py --dataset combined downloads + tokenizes both TinyStories and DailyDialog
# itself (see scripts/build_vocab.py), no separate download_dataset.py call needed
if [ ! -f data/train_tinystories.bin ] || [ ! -f data/train_dailydialog.bin ]; then
  sudo -u ubuntu python3 scripts/build_vocab.py --dataset combined
fi

sudo -u ubuntu cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
sudo -u ubuntu cmake --build build -j"$(nproc)"

# EXECUTE already auto-resumes from ../tinystories.mlt if present (src/main.cpp), so simply
# re-enabling this service on every boot is enough to survive spot stop/resume cycles
cat > /etc/systemd/system/ml-train.service <<'UNIT'
[Unit]
Description=ML_LIBRARY transformer training
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ML_LIBRARY/build
ExecStart=/home/ubuntu/ML_LIBRARY/build/EXECUTE
Restart=no
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

# runs as root (not the ubuntu user) so it can read the system journal for ml-train's log
# without needing extra group permissions; S3 credentials come from the instance profile.
# relies on the AWS CLI preinstalled on the Deep Learning AMI being on root's PATH
cat > /etc/systemd/system/ml-checkpoint-sync.service <<UNIT
[Unit]
Description=Sync training checkpoint and log to S3

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'journalctl -u ml-train --no-pager -n 1000 > $REPO_DIR/training.log; \
  aws s3 cp $REPO_DIR/tinystories.mlt s3://$S3_BUCKET/checkpoints/tinystories.mlt || true; \
  aws s3 cp $REPO_DIR/training.log s3://$S3_BUCKET/checkpoints/training.log || true'
UNIT

cat > /etc/systemd/system/ml-checkpoint-sync.timer <<'TIMER'
[Unit]
Description=Run ml-checkpoint-sync every 15 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=15min

[Install]
WantedBy=timers.target
TIMER

systemctl daemon-reload
systemctl enable --now ml-train.service
systemctl enable --now ml-checkpoint-sync.timer

# hard budget guardrail: self-stop after AUTO_STOP_HOURS of cumulative *running* time.
# scheduled via `at`, which only counts down while the instance is powered on, so a spot
# stop/resume cycle naturally pauses this clock instead of racing it
echo "shutdown -h now" | at now + "$AUTO_STOP_HOURS" hours
