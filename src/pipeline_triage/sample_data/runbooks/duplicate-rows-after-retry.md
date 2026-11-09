# Duplicate rows after a retried load

## Symptoms
- Row counts jump well above the baseline, often close to double.
- Uniqueness or primary key checks fail on the target table.
- The load task succeeded on its second or third try.

## Likely causes
- The load appends instead of replacing the partition, so a retry inserts the data again.
- A manual rerun was triggered while the original run was still loading.

## Diagnosis
1. Count duplicates on the business key for the affected partition.
2. Check the task try numbers and rerun history.

## Fix
- Delete and reload the partition, or deduplicate on the business key.

## Prevention
- Make loads idempotent: overwrite partitions or use MERGE on a key.
