# Onboarding FAQ

## Day-1 checklist

1. Accept Okta invite and enable MFA.
2. Join Slack workspaces and channels from the handbook.
3. Clone the internal monorepo and run `make bootstrap`.
4. Request staging access with ticket code **ACC-STG**.
5. Complete the security training quiz (quiz ID: **SEC-101**).

## Laptop setup

- OS: macOS or Ubuntu LTS
- Required tools: Docker, Ollama, Python 3.12+, Node 20+
- Install script: `./scripts/dev_setup.sh`
- Default LLM model for local demos: `qwen3:8b`
- Default embedding model: `nomic-embed-text`

## Who to ask

- Manager for goals and leave
- Buddy for repo navigation
- `#iam-help` for SSO / VPN
- `#langgraph-lab` for AI stack questions

## First project

New hires ship a small grounded Q&A bot against the Contoso Ops knowledge
base. Success means: citations on every answer, hybrid retrieval for ticket
IDs like `HR-LEAVE-24`, and a refuse path when evidence is missing.
