# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| main    | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in LOATO-Bench, please report it
responsibly:

1. **Do not** open a public GitHub issue
2. Email **ak58214n@pace.edu** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. You will receive an acknowledgment within 48 hours
4. We will work with you to understand and address the issue before any public
   disclosure

## Scope

This project is a research benchmark, not a production security tool. However,
we take the following seriously:

- **API key exposure** — ensure `.env` files are never committed
- **Data integrity** — tampering with datasets or splits that could affect
  published results
- **Dependency vulnerabilities** — outdated packages with known CVEs

## Out of Scope

- The prompt injection attacks in the dataset are intentional research artifacts
- Theoretical weaknesses in the classifiers under study (that is the point of
  the research)
