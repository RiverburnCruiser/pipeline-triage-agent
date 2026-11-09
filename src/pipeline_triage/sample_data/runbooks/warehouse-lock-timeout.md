# Warehouse lock wait timeout

## Symptoms
- A load or DDL statement fails with `lock wait timeout exceeded`, `deadlock detected` or `could not obtain lock`.
- The failure coincides with another job writing to the same table.

## Likely causes
- Two jobs write to or alter the same table at the same time.
- A long-running analytical query holds a lock during maintenance.

## Diagnosis
1. Find the blocking session in the warehouse lock views at the failure time.
2. Check the schedule for jobs touching the same table.

## Fix
- Rerun once the blocking job finishes.
- Serialise writers to the table with an Airflow pool of size one.

## Prevention
- Stagger schedules and keep maintenance windows away from loads.
