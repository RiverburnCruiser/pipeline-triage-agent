# Sensor timed out waiting for an upstream DAG

## Symptoms
- A sensor task such as `ExternalTaskSensor` fails with `AirflowSensorTimeout`.
- The log shows repeated "Poking for tasks" lines until the timeout.
- The upstream task is `failed` or `upstream_failed`, or has not run yet.

## Likely causes
- The upstream DAG failed, so the awaited task never succeeded. This is the most common cause; the sensor failure is a symptom.
- The upstream DAG is late because of a backlog or a long-running task.
- The sensor's execution date offset points at the wrong upstream run.

## Diagnosis
1. Find the upstream DAG and logical date the sensor was waiting for in its log.
2. Check that upstream run's state. If it failed, triage that run first.
3. If the upstream run succeeded, check the execution date mapping.

## Fix
- Fix and rerun the upstream run, then clear the sensor task so it picks up the success.
- Do not just raise the timeout when the upstream run has failed.

## Prevention
- Set `failed_states=["failed", "upstream_failed"]` on the sensor so it fails fast with the real reason instead of timing out.
- Use `mode="reschedule"` for long waits so sensors do not hold worker slots.
