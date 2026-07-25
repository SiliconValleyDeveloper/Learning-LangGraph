# Incident Playbook — P1 / P2

## Severity definitions

- **P1**: Customer-facing outage or data-loss risk. Acknowledge in **5 minutes**.
- **P2**: Major degradation with workaround. Acknowledge in **15 minutes**.
- **P3**: Localized bug. Acknowledge in **4 hours**.

## P1 response checklist

1. Page the primary on-call via PagerDuty service **contoso-ops-p1**.
2. Open a war-room Zoom and pin the incident ticket `INC-#####`.
3. Post a status update in `#ops-oncall` every **15 minutes**.
4. Freeze non-emergency deploys using the flag `DEPLOY_FREEZE=true`.
5. After mitigation, write a blameless postmortem within **48 hours**.

## Rollback command

```bash
kubectl rollout undo deployment/api-gateway -n production
```

Rollback owner: platform on-call. Approval code for emergency rollback: **RB-EMERGENCY-7**.

## Escalation

If not mitigated in 30 minutes, escalate to the incident commander listed
in the on-call calendar. Never share customer PII in public channels.
