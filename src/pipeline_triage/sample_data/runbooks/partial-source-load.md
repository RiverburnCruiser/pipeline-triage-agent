# Partial source data loaded

## Symptoms
- A reconciliation or row count check fails because rows dropped sharply against the recent baseline.
- The extract logged fewer source files or objects than usual but still succeeded.
- The source manifest lists more objects than are present in the bucket.

## Likely causes
- The upstream export was still running or crashed part way when the extract started.
- A delivery job uploaded only some partitions or files.
- Files landed under a different prefix or date partition.

## Diagnosis
1. Compare the loaded row count with the trailing average and with the manifest.
2. Count objects in the source prefix now; late files may have arrived since.
3. Ask the upstream team whether their export completed.

## Fix
- Once the missing files arrive, rerun extract and load for the partition, replacing it rather than appending.
- If files will not arrive, mark the partition incomplete and tell downstream consumers.

## Prevention
- Make the minimum object check blocking instead of advisory, or wait for a manifest or `_SUCCESS` marker before extracting.
