---
title: Information security policy
department: engineering
visibility: restricted
version: 1
status: active
effective: 2026-10-01
updated: 2026-11-01
---

# Information security policy

## External AI tools

Engineers must not upload company source code, credentials, or customer data to external AI tools.
An exception requires written approval from the engineering security lead before any upload occurs.
Approved internal AI tools may be used only with the data classifications listed in their approval record.

## Credentials and access

Production credentials must be stored in the company secrets manager.
Credentials must not be committed to a source code repository or shared in chat messages.
Multi-factor authentication is required for access to production systems.

## Security incident reporting

Engineers must report a suspected credential leak to the security team within fifteen minutes of discovery.
The security team must revoke exposed credentials before issuing replacements.
