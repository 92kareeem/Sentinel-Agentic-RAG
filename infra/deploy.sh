#!/usr/bin/env bash
# Sentinel deployment. Referenced by `make deploy`.
#
# This script was referenced by the Makefile but did not exist, so "deploy" was
# an undocumented sequence of manual console/CLI steps — not reproducible, and
# impossible to review. Everything below is idempotent: re-running it converges
# on the desired state rather than failing on already-created resources.
#
# Usage:
#   AWS_ACCOUNT_ID=... GROQ_API_KEY=... bash infra/deploy.sh [--skip-image]
#
# Prerequisites: aws cli v2 authenticated, docker with buildx, jq.

set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:?set AWS_ACCOUNT_ID}"
PROJECT=sentinel
FUNCTION="${PROJECT}-api"
ECR_REPO="${PROJECT}-api"
DOCS_BUCKET="${S3_BUCKET_DOCS:-${PROJECT}-docs-${ACCOUNT_ID}}"
ROLE_NAME="${PROJECT}-lambda-role"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:latest"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 1. storage
log "S3 buckets"
for bucket in "$DOCS_BUCKET" "${PROJECT}-frontend-${ACCOUNT_ID}"; do
  if ! aws s3api head-bucket --bucket "$bucket" 2>/dev/null; then
    aws s3api create-bucket --bucket "$bucket" --region "$REGION" \
      --create-bucket-configuration "LocationConstraint=$REGION"
    aws s3api put-public-access-block --bucket "$bucket" \
      --public-access-block-configuration \
      "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
  fi
done
aws s3api put-bucket-lifecycle-configuration --bucket "$DOCS_BUCKET" \
  --lifecycle-configuration "file://${HERE}/s3-lifecycle.json"
aws s3api put-bucket-cors --bucket "$DOCS_BUCKET" \
  --cors-configuration "file://${HERE}/s3-cors.json"

log "DynamoDB tables"
for table in users quotas traces documents; do
  spec="${HERE}/ddb-${table}.json"
  name="$(jq -r .TableName "$spec")"
  if ! aws dynamodb describe-table --table-name "$name" --region "$REGION" >/dev/null 2>&1; then
    aws dynamodb create-table --cli-input-json "file://${spec}" --region "$REGION" >/dev/null
    aws dynamodb wait table-exists --table-name "$name" --region "$REGION"
  fi
done
# TTL keeps traces/quotas from growing without bound (and off the free tier).
for table in traces quotas; do
  name="${PROJECT}-${table}"
  aws dynamodb update-time-to-live --table-name "$name" --region "$REGION" \
    --time-to-live-specification "Enabled=true,AttributeName=ttl" >/dev/null 2>&1 || true
done

# ---------------------------------------------------------------- 2. iam
log "IAM execution role"
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document "file://${HERE}/iam/lambda-trust.json" >/dev/null
fi
# Least privilege: scoped to this project's buckets/tables/log group, no wildcards.
aws iam put-role-policy --role-name "$ROLE_NAME" \
  --policy-name "${PROJECT}-lambda-policy" \
  --policy-document "file://${HERE}/iam/lambda-policy.json"

# ---------------------------------------------------------------- 3. image
if [[ "${1:-}" != "--skip-image" ]]; then
  log "ECR + container image"
  aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION" >/dev/null
  aws ecr put-lifecycle-policy --repository-name "$ECR_REPO" --region "$REGION" \
    --lifecycle-policy-text "file://${HERE}/ecr-lifecycle.json" >/dev/null

  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

  # --provenance/--sbom false: Docker >=29 otherwise pushes an OCI index with
  # attestations, which Lambda rejects ("image manifest ... not supported").
  docker buildx build \
    --platform linux/amd64 \
    --provenance=false --sbom=false \
    -f "${ROOT}/docker/Dockerfile.lambda" \
    -t "$IMAGE_URI" \
    --push "$ROOT"
fi

# ---------------------------------------------------------------- 4. lambda
log "Lambda function"
ENV_VARS="Variables={GROQ_API_KEY=${GROQ_API_KEY:?set GROQ_API_KEY},S3_BUCKET_DOCS=${DOCS_BUCKET},AWS_REGION_OVERRIDE=${REGION}}"

if aws lambda get-function --function-name "$FUNCTION" --region "$REGION" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$FUNCTION" --region "$REGION" \
    --image-uri "$IMAGE_URI" >/dev/null
  aws lambda wait function-updated --function-name "$FUNCTION" --region "$REGION"
  aws lambda update-function-configuration --function-name "$FUNCTION" --region "$REGION" \
    --timeout 120 --memory-size 2048 --ephemeral-storage "Size=1024" \
    --environment "$ENV_VARS" >/dev/null
else
  aws lambda create-function --function-name "$FUNCTION" --region "$REGION" \
    --package-type Image --code "ImageUri=${IMAGE_URI}" \
    --role "arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}" \
    --timeout 120 --memory-size 2048 --ephemeral-storage "Size=1024" \
    --environment "$ENV_VARS" >/dev/null
fi
aws lambda wait function-updated --function-name "$FUNCTION" --region "$REGION"

# ---------------------------------------------------------------- 5. url
log "Function URL"
aws lambda create-function-url-config --function-name "$FUNCTION" --region "$REGION" \
  --auth-type NONE >/dev/null 2>&1 || true
# A NONE-auth Function URL needs BOTH statements since the Oct 2025 change;
# with only InvokeFunctionUrl every request fails at the AWS layer with
# "Forbidden" before it ever reaches the app.
aws lambda add-permission --function-name "$FUNCTION" --region "$REGION" \
  --statement-id FunctionURLAllowPublicAccess --action lambda:InvokeFunctionUrl \
  --principal "*" --function-url-auth-type NONE >/dev/null 2>&1 || true
aws lambda add-permission --function-name "$FUNCTION" --region "$REGION" \
  --statement-id FunctionURLInvoke --action lambda:InvokeFunction \
  --principal "*" >/dev/null 2>&1 || true

URL="$(aws lambda get-function-url-config --function-name "$FUNCTION" --region "$REGION" \
  --query FunctionUrl --output text)"

# ---------------------------------------------------------------- 6. index
log "Publishing index artifacts"
if [[ -f "${ROOT}/index/current.json" ]]; then
  VERSION="$(jq -r .version "${ROOT}/index/current.json")"
  # Version directory first, pointer LAST: a Lambda cold-starting mid-sync must
  # read a complete older version rather than a torn newer one.
  aws s3 sync "${ROOT}/index/${VERSION}" "s3://${DOCS_BUCKET}/index/${VERSION}"
  aws s3 cp "${ROOT}/index/current.json" "s3://${DOCS_BUCKET}/index/current.json"
  echo "published index ${VERSION}"
else
  echo "WARNING: no local index (run 'make ingest' first) — /v1/query will have an empty corpus"
fi

# ---------------------------------------------------------------- 7. verify
log "Smoke test"
if curl -fsS "${URL}healthz" >/dev/null; then
  curl -sS "${URL}healthz"; echo
  echo "DEPLOYED: ${URL}"
else
  echo "healthz did not respond — check: aws logs tail /aws/lambda/${FUNCTION} --since 5m"
  exit 1
fi
