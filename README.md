# mlops-project-template (fork)

Fork of [`Azure/mlops-project-template`](https://github.com/Azure/mlops-project-template), maintained as the project scaffold for a small MLOps v2 factory — see [`rubyrayjuntos/azure-mlops`](https://github.com/rubyrayjuntos/azure-mlops) for the full architecture writeup and the first project generated from this factory.

## What's different from upstream

New projects generated via `sparse_checkout.sh` from this fork (instead of `Azure/mlops-project-template` directly) come pre-fixed:

- `id-token: write` permissions added to every job that calls an Azure-OIDC reusable workflow, in `tf-gha-deploy-infra.yml`, `deploy-model-training-pipeline-classical.yml`, `deploy-batch-endpoint-pipeline-classical.yml`, `deploy-online-endpoint-pipeline-classical.yml` — without these, GitHub silently caps OIDC token permissions and every one of these pipelines fails on its first real run.
- The deprecated `AZURE_CREDENTIALS`/`creds:` secret pattern replaced with the OIDC 3-secret pattern in the batch and online endpoint pipelines.
- All workflow references point at [`rubyrayjuntos/mlops-templates`](https://github.com/rubyrayjuntos/mlops-templates) instead of unpinned `Azure/mlops-templates@main`.
- `enable_private_endpoints` defaults to `false` (Terraform variable), matching the documented default — upstream currently has it hardcoded `true`.
- `config-infra-dev.yml`/`config-infra-prod.yml`: `terraform_workingdir` corrected to `infrastructure` (upstream points at `infrastructure/terraform`, a path that doesn't exist after `sparse_checkout.sh` flattens it).
- `train-conda.yml`: added `setuptools<70`, fixing an `mlflow` `ModuleNotFoundError` that only surfaces once the training pipeline actually runs.
- `batch-cluster`'s VM size/tier fixed to values actually available in a typical subscription (`STANDARD_D4S_V3`, `dedicated`).

**Note on file paths**: this repo carries two copies of most GitHub Actions workflows — a top-level `.github/workflows/*.yml` (this repo's own CI, never shipped to generated projects) and the real sparse-checkout sources under `infrastructure/terraform/github-actions/` and `classical/<mlops_version>/mlops/github-actions/`. All of the fixes above are in the real (nested) sources.

## Usage

Point `sparse_checkout.sh`'s `project_template_github_url` at `https://github.com/rubyrayjuntos/mlops-project-template` instead of the upstream Azure URL.

---

[Upstream accelerator README](https://github.com/Azure/mlops-v2/blob/main/README.md)
