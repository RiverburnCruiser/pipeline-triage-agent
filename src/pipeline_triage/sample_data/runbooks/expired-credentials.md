# Expired or revoked API credentials

## Symptoms
- An extract task fails with HTTP 401 Unauthorized or 403 Forbidden.
- Error bodies such as `invalid_token`, `invalid_client`, `token expired` or `client secret has expired`.
- Retries fail the same way; the task worked on the previous run.

## Likely causes
- An OAuth access token expired and the refresh failed.
- The client secret or API key itself expired or was rotated by the vendor.
- The service account was disabled or lost a permission scope.

## Diagnosis
1. Read the error body: `invalid_token` means the access token, `invalid_client` means the client credentials themselves.
2. Check the expiry date of the secret in the vendor portal and in the secrets manager.
3. Confirm no one rotated the secret without updating the Airflow connection.

## Fix
- Rotate the client secret in the vendor portal, store it in the secrets manager, update the Airflow connection, and rerun the failed runs.
- Never paste secrets into DAG code or logs.

## Prevention
- Track secret expiry dates and alert 14 days before expiry.
- Prefer credentials that rotate automatically.
