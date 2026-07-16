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

## Step 4b — ECR + image — DONE 2026-07-15
- Repo sentinel-api. Image built from docker/Dockerfile.lambda (no torch; ONNX MiniLM). Compressed 297 MB (< 500 MB free tier).
- GOTCHA: Docker 29 pushes OCI index + provenance attestations; Lambda rejects them ("image manifest... not supported"). Fix: `docker buildx build --provenance=false --sbom=false ... --push`.

## Step 4c — Lambda + Function URL — DONE 2026-07-15
- Function sentinel-api: image package, 2048 MB, 30s, role sentinel-lambda-role, env GROQ_API_KEY + S3_BUCKET_DOCS (rest baked in image).
- Function URL (auth NONE): https://chlc5xtu67wbi2i3gfceyas5zy0etude.lambda-url.ap-south-1.on.aws/
- GOTCHA (Oct 2025 change): NONE-auth URLs need TWO resource policy statements — lambda:InvokeFunctionUrl (FunctionUrlAuthType=NONE) AND lambda:InvokeFunction (InvokedViaFunctionUrl=true). With only the first, every request gets AWS-level "Forbidden".
- Smoke test PASSED: /healthz 200; /v1/query returned cited table answer (India 2%), critic 1.0/1.0, trace_id issued; bad key -> 401 from app.

## Upload feature deploy — DONE 2026-07-16
- S3 CORS on docs bucket (POST, origins *) via infra/s3-cors.json — browser POSTs files direct to S3.
- Image rebuilt with pymupdf (PDF parsing in-Lambda); 324 MB compressed (< 500 MB tier).
- Lambda timeout 30s -> 120s (embedding headroom for uploads).
- LIVE end-to-end verified: presigned POST -> S3 204 -> index (2 chunks, 14.6s) -> query returns uploaded fact ($110 intl per diem) cited to new doc, critic 1.0/1.0.
- Design: presigned POST (not PUT) enforces 5MB via content-length-range; index endpoint takes filename as query param so no s3:ListBucket needed; merge reuses old vectors via faiss reconstruct_n, idempotent by doc_id; retriever.reset_cache() after merge so warm Lambda serves new doc.

## Step 4a — index upload + Lambda execution role — DONE 2026-07-11
- index/ artifacts uploaded to s3://sentinel-docs-440744255230/index/ (bm25.pkl, chunks.jsonl, faiss.index).
- Role `sentinel-lambda-role` (trust: lambda.amazonaws.com) + inline `sentinel-lambda-policy` (infra/iam/lambda-policy.json): S3 Get/Put on docs bucket objects, 5 DynamoDB actions on 3 tables + 2 GSIs, logs scoped to /aws/lambda/sentinel-api. No resource wildcards.
