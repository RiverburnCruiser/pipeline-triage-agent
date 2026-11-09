# Task killed for running out of memory

## Symptoms
- The task exits with `Negsignal.SIGKILL`, return code -9 or 137, or `MemoryError`.
- Kubernetes reports the pod as `OOMKilled` because it exceeded its memory limit.
- The log stops abruptly during a read, join, groupby or resample.

## Likely causes
- The whole input is loaded into pandas at once and has outgrown the memory limit.
- A wide groupby or resample creates large intermediate frames.
- Input volume grew (more devices, more events) while the limit stayed the same.

## Diagnosis
1. Compare the input row count and the in-memory size logged before the crash with the pod memory limit.
2. Check whether input volume has grown over recent runs.
3. Find the last log line before the kill to locate the expensive step.

## Fix
- Short term: raise the pod memory limit and rerun.
- Long term: process the input in chunks or partitions, read only needed columns, downcast dtypes, or push the aggregation into the warehouse or a columnar engine such as DuckDB or Polars.

## Prevention
- Alert on memory usage above 80 percent of the limit.
- Load test with projected volumes when fleet size grows.
