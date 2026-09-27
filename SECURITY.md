# Security

Do not post secrets or sensitive documents in public issues. For a vulnerability in your
published fork, use GitHub private vulnerability reporting if the repository owner enables it.
No public disclosure mailbox is fabricated in this template.

This version assumes a trusted operator and one shared corpus. It is not a multi-tenant
security boundary. See `docs/PRODUCTION.md` for public-deployment requirements.

Prompt boundaries and citation-ID validation are partial controls, not a guarantee against
prompt injection or hallucinations. Never use a generated answer as authorization to execute
code, modify data, or make a high-impact decision without appropriate verification.
