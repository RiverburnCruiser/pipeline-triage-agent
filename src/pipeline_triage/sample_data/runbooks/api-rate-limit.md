# Vendor API rate limiting

## Symptoms
- Extract tasks fail with HTTP 429 Too Many Requests or `rate limit exceeded`.
- Responses include a `Retry-After` header.
- Failures cluster when several DAGs call the same vendor at once.

## Likely causes
- Parallel tasks or backfills exceed the vendor's request quota.
- The client ignores `Retry-After` and retries immediately.

## Diagnosis
1. Count requests per minute from the logs and compare with the vendor quota.
2. Check whether a backfill or a new DAG started calling the same API.

## Fix
- Honour `Retry-After` with exponential backoff and jitter, then rerun.
- Limit concurrency with an Airflow pool sized to the vendor quota.

## Prevention
- Put every DAG that calls the vendor in one shared pool.
