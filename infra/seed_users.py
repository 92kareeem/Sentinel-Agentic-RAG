"""Seed the admin and demo users into sentinel-users.

Run once after the tables exist:
    C:\\venvs\\sentinel\\Scripts\\python.exe infra\\seed_users.py

Generates two API keys, stores ONLY their sha256 hashes in DynamoDB, and
prints the plaintext keys exactly once — copy them somewhere safe (the demo
key goes in the frontend .env later; the admin key goes nowhere but your
password manager). Re-running rotates both keys.
"""

import hashlib
import secrets
from datetime import UTC, datetime

import boto3

REGION = "ap-south-1"
TABLE = "sentinel-users"

USERS = [
    {"user_id": "admin", "is_admin": True, "daily_query_limit": 0, "display_name": "Admin"},
    {"user_id": "demo", "is_admin": False, "daily_query_limit": 50, "display_name": "Demo"},
]


def main() -> None:
    table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE)
    print("Seeding users — plaintext keys are shown ONCE, only hashes are stored:\n")
    for user in USERS:
        key = f"snt_{secrets.token_urlsafe(32)}"
        item = {
            **user,
            "api_key_hash": hashlib.sha256(key.encode()).hexdigest(),
            "upload_doc_limit": 10,
            "upload_bytes_limit": 20_971_520,
            "created_at": datetime.now(UTC).isoformat(),
        }
        table.put_item(Item=item)
        print(f"  {user['user_id']:<6} API key: {key}")
    print("\nDone. Verify: aws dynamodb scan --table-name sentinel-users --select COUNT")


if __name__ == "__main__":
    main()
