# Sentinel — AWS setup log

Account: 440744255230 · Region: ap-south-1 · All resources tagged Project=sentinel.
Every step recorded here as it is executed (P4 teaching contract).

## Step 0-1 — 2026-07-10
- CLI verified: aws-cli/2.35.15, IAM user `syed` (AdministratorAccess). Default region set to ap-south-1.
- LEARNING: human CLI user = broad+temporary; Lambda execution role = narrow+permanent.

## Step 2 — S3 buckets — DONE 2026-07-10
- `sentinel-docs-440744255230` — prefixes: uploads/{user_id}/, keep/, index/. Public access blocked. Lifecycle: uploads/ expires after 90 days (infra/s3-lifecycle.json).
- `sentinel-frontend-440744255230` — static site, private, served only via CloudFront OAC (Step 8).

## Step 3 — DynamoDB — DONE 2026-07-11
- 3 tables created from infra/ddb-*.json (on-demand). TTL ENABLED on traces.ttl + quotas.ttl.
- Users seeded (admin + demo); plaintext keys in gitignored infra/API_KEYS.txt (rotate by re-running infra/seed_users.py).
- LEARNING: quota check is one conditional UpdateItem — the write IS the check, race-free by design.

## Step 4a — index upload + Lambda execution role — DONE 2026-07-11
- index/ artifacts uploaded to s3://sentinel-docs-440744255230/index/ (bm25.pkl, chunks.jsonl, faiss.index).
- Role `sentinel-lambda-role` (trust: lambda.amazonaws.com) + inline `sentinel-lambda-policy` (infra/iam/lambda-policy.json): S3 Get/Put on docs bucket objects, 5 DynamoDB actions on 3 tables + 2 GSIs, logs scoped to /aws/lambda/sentinel-api. No resource wildcards.
