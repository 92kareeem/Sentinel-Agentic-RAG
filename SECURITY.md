# Security policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub's private vulnerability reporting](https://github.com/92kareeem/sentinel-rag/security/advisories/new)
for this repository. Do not open a public issue.

Include what you can: the component (`sentinel-proof` package, the API, the
UI, the AWS deployment scripts), steps to reproduce, and the impact you
believe it has. You will get an acknowledgement within 72 hours and a status
update within 14 days.

## Supported versions

| Component | Supported |
|---|---|
| `sentinel-proof` | the latest minor release on PyPI |
| Sentinel platform | the `main` branch |

## What counts

Anything that lets an answer bypass the grounding gate while reporting
`ok=True`, lets a caller read documents outside their tenant, role or
clearance, leaks a provider key or PII into logs or traces, or lets document
content act as instructions to the model. Prompt injection that is *blocked*
and shows up in the chaos demo is working as intended; one that gets through
is a report we want.

## Dependencies

Dependabot alerts on this repository are patched promptly. `sentinel-proof`
has no runtime dependencies by design, and CI enforces that.
