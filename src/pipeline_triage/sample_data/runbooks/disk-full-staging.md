# Staging disk full

## Symptoms
- A task fails with `No space left on device` or `OSError: [Errno 28]`.
- Writes to a staging directory or temp folder fail part way.

## Likely causes
- Old staging files were never cleaned up.
- An unusually large partition filled the volume.

## Diagnosis
1. Check free space on the worker's staging volume.
2. Find the largest directories under the staging path.

## Fix
- Delete expired staging files, then rerun.
- Stage to object storage instead of local disk for large inputs.

## Prevention
- Add a cleanup task at the end of each DAG and alert at 80 percent disk usage.
