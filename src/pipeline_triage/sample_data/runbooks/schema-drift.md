# Schema drift from an upstream producer

## Symptoms
- A schema validation or contract check fails before the load step, for example `SchemaContractError`.
- The log lists unexpected columns, missing columns, or type mismatches (a timestamp arriving as a string).
- Downstream tasks are `upstream_failed` and dependent DAGs wait on this one.

## Likely causes
- An upstream service released a new version that renamed, added or retyped fields.
- A producer changed serialization (for example epoch integers to ISO strings).

## Diagnosis
1. Read the validation log and list every added, removed and retyped column.
2. Check the upstream release log for a deploy just before the first bad partition.
3. Confirm which partitions are affected by validating the previous and next day.

## Fix
- If the change is intended, publish a new contract version, map renamed columns (for example `speed_kmh` to `speed_kph`) and cast retyped fields in the transform, then rerun the failed partitions.
- If it is not intended, ask the producing team to roll back, then rerun.
- Clear downstream DAG runs that failed or timed out waiting on this load.

## Prevention
- Add a contract test to the producer's CI so breaking changes fail before release.
- Version schemas and announce changes in the data contracts channel.
