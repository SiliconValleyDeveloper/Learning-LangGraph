# Access Control Policy

## SSO

All employees authenticate with Okta SSO. MFA is mandatory.
Service accounts use short-lived tokens from Vault path `secret/ops/svc`.

## Environment access

| Environment | Who | Ticket |
|-------------|-----|--------|
| `dev` | All engineers | none |
| `staging` | Engineers + QA | `ACC-STG` |
| `production` read | SRE + on-call | `ACC-PROD-READ` |
| `production` write | SRE lead approval | `ACC-PROD-WRITE` |

## Production write rules

1. Open a change ticket with risk notes.
2. Get approval from an SRE lead (not the requester).
3. Use break-glass role `prod-break-glass` only during P1 incidents.
4. Break-glass sessions expire after **60 minutes**.
5. Log the session ID in the incident ticket.

## Data classification

- **Public**: marketing site, public docs
- **Internal**: runbooks, FAQs, handbook
- **Confidential**: customer notes, salary bands, API keys

Confidential data must not be embedded into shared vector indexes without
row-level filters (`visibility=confidential`).
