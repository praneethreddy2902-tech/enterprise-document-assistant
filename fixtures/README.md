# Sample policy uploads

All policies in this directory are fictional demonstration data, not real company
rules. Upload the Markdown files through the app; this folder is not scanned.
The account picker is a demo identity selector, not authentication.

## Upload as Bob (HR)

- `policies/leave_policy.md`: internal; readable by all employees.
- `policies/compensation_policy.md`: restricted; readable only by HR employees.
- `policies/remote_work_policy.md`: internal, intentionally last updated January
  1, 2026 to demonstrate a freshness warning after 180 days.

## Upload as Alice (Engineering)

- `policies/security_policy.md`: restricted; readable only by engineering employees.

## Upload as Carol (Finance)

- `policies/refund_policy.md`: public; readable by guests and employees.
- `policies/travel_policy_v2.md`: superseded internal policy; historical searches.
- `policies/travel_policy_v3.md`: active internal policy; current searches.

Upload both travel policy files to compare versions. Version 2 is explicitly
superseded, so file order does not matter. Uploads call OpenAI to embed content.

## Questions to try

- Guest: "How long do customers have to request a refund?" Expect thirty days.
- Guest: "How much annual leave do employees receive?" Expect a refusal.
- Bob or Alice: "How much annual leave do employees receive?" Expect twenty days.
- Alice: "Can engineers upload company source code to external AI tools?"
  Expect a prohibition with an exception requiring prior written security approval.
- Bob: ask the same source-code question. Expect a refusal.
- Bob: "When is the annual compensation review?" Expect April.
- Alice: ask the compensation-review question. Expect a refusal.
- Any employee: "What is the current domestic hotel reimbursement limit?"
  Expect INR 4500 per night from version 3.
- Any employee: "What was the domestic hotel reimbursement limit in version 2?"
  Expect INR 3000 per night and a superseded-policy warning.
- Any employee: "How many days per week can I work remotely?"
  Expect two days with manager approval, plus a freshness warning.
- Any identity: "What is the company policy on bringing pets to the office?"
  Expect insufficient evidence; none of these files defines a pet policy.

These are expected behaviors, not guarantees of live model responses. Inspect
citations and warnings. Quotation checks do not prove semantic correctness.

## Duplicate check

Upload the same file again as its owning department. Expect `duplicate` and zero
new chunks. Renaming the file without changing its contents also counts as a
duplicate. Changing a title/version's text replaces that version's old chunks.
