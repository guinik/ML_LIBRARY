# Training on AWS (Terraform)

Spins up a single g5.xlarge (A10G, 24GB) **spot** instance that clones this repo, builds it
with `-DUSE_CUDA=ON`, and trains the scaled-up model (see `src/main.cpp`) unattended, with
checkpoints synced to S3 every 15 minutes and a hard runtime cap as a budget guardrail.

This is real AWS infrastructure that spends real money. Nothing here runs itself — you run
`terraform apply` yourself, when you're ready.

## Prerequisites

- Terraform >= 1.5, AWS CLI v2, credentials for an account with GPU spot quota for the
  instance family below already approved (per your existing setup).
- `terraform/terraform.tfvars` filled in from `terraform.tfvars.example` — at minimum,
  `notification_email` (no default on purpose).

## Cost math

g5.xlarge spot pricing varies by region/time, roughly **$0.30-0.45/hr**. Against an
$80-100 budget that's **~180-260 spot-hours**. `auto_stop_after_hours` (default 90) self-stops
the instance well inside that range as a hard safety net — raise it once you've confirmed
actual ms/step on-instance and know how many hours you actually want.

Compare against on-demand (~$1.00/hr) if you ever need guaranteed non-interrupted time for a
final run — spot is the default here because checkpointing already exists and losing a spot
instance just means an automatic resume, not lost work (see "How interruption is handled"
below).

## One-time setup

1. **Activate cost-allocation tags.** `budget.tf`'s cost alert filters spend by the `Project`
   tag applied to every resource here, so it doesn't get mixed in with anything else on the
   account. AWS requires activating tag-based cost allocation manually once, outside
   Terraform: Billing Console → Cost allocation tags → activate `Project`. Newly activated
   tags can take up to ~24h to start appearing in cost data — the budget resource still gets
   created immediately, it just won't have real numbers to alert on until the tag is live.
2. `cd infra/terraform && cp terraform.tfvars.example terraform.tfvars` and fill it in.

## Apply

```bash
cd infra/terraform
terraform init
terraform plan   # review before applying anything
terraform apply
```

State is kept local (`terraform.tfstate`, gitignored) — fine for a single-user setup, no
remote backend configured.

On first boot the instance (see `templates/user_data.sh.tpl`):
1. Installs build deps, clones the repo at `git_branch`.
2. Runs `scripts/build_vocab.py --dataset combined` (downloads + tokenizes both TinyStories
   and DailyDialog, matching the two-phase pretrain/finetune pipeline `src/main.cpp` expects).
3. Builds with `-DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release`.
4. Installs and starts `ml-train.service` (runs `build/EXECUTE`) and
   `ml-checkpoint-sync.timer` (syncs `tinystories.mlt` + a recent training-log excerpt to S3
   every 15 minutes).
5. Schedules the `auto_stop_after_hours` self-stop.

This takes a while (dataset download/tokenization + CUDA build) before training visibly
starts — that's expected, not a hang.

## Before committing to the long run: tune batch size

`src/main.cpp`'s `BATCH_SIZE` is deliberately left conservative (16) rather than guessed,
because this engine keeps every intermediate activation — including the
`batch*heads*seq*seq` attention-score tensor per layer — as a full fp32 tensor with no
recomputation/activation-checkpointing, so real VRAM use at `seq_len=256` isn't reliably
predictable from parameter count alone. Once connected (see below):

```bash
cd ~/ML_LIBRARY/build
./BENCHMARK        # prints ms/step, tokens/sec, and will fail fast with a CUDA OOM if the config doesn't fit
```

If you want a bigger batch than the default, edit `BATCH_SIZE` in `src/main.cpp`, rebuild
(`cmake --build build`), and restart the service (`sudo systemctl restart ml-train`) *before*
letting it run for hours unattended.

## Watching progress

No SSH key or open port needed — access goes through SSM Session Manager (the instance role
has `AmazonSSMManagedInstanceCore`):

```bash
aws ssm start-session --target <instance_id> --region <region>   # from `terraform output`
sudo journalctl -u ml-train -f                                    # live training log
```

Or without connecting at all, read the periodic sync from S3:

```bash
aws s3 cp s3://<bucket>/checkpoints/training.log - | tail -100
```

(`terraform output` prints the exact bucket name and ready-to-run commands.)

## How interruption is handled

The instance is a **persistent** spot request with `instance_interruption_behavior = "stop"`:
on reclaim it *stops* rather than terminating, so the EBS root volume (repo, build,
checkpoint) survives untouched and AWS automatically restarts it once spot capacity returns.
`ml-train.service` is enabled (not just run once via user_data), so it starts again on that
next boot — and `EXECUTE` already auto-resumes from `../tinystories.mlt` on disk
(`src/main.cpp`) — so no manual intervention or S3 round-trip is needed to pick back up. The
periodic S3 sync exists for durability/inspection convenience, not as part of the resume path.

If `ml-train` ever crashes mid-run (rare, not the expected interruption path):
`sudo systemctl restart ml-train` — it'll resume from the last checkpoint on disk the same way.

## Getting your results

```bash
aws s3 cp s3://<bucket>/checkpoints/tinystories.mlt ./tinystories.mlt
```

(also printed by `terraform output download_checkpoint_command`). The checkpoint is a
self-contained `.mlt` file (see the root README's "Checkpoint format" section) — copy it back
into the repo root locally and run `CHAT` against it, or keep training further.

## Tearing down

```bash
terraform destroy
```

This terminates the instance and deletes the security group/IAM role/budget — **the S3
checkpoint bucket is not deleted automatically** (no `force_destroy` set), so your results
survive a `destroy`. Delete the bucket manually once you've pulled what you need from it.

## Known simplifications

- Local Terraform state, no remote backend — deliberate, appropriate for single-user use.
- `spot_max_price` defaults to unset (AWS treats that as the on-demand price cap); set it
  explicitly in `terraform.tfvars` if you want a tighter bid ceiling.
- The budget alert is account-billing-based (via cost-allocation tag filter) and can lag real
  spend by hours; it is a monitoring aid, not a real-time cutoff — `auto_stop_after_hours` is
  the actual hard stop.
