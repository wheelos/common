---
name: module-development
description: Develop wheelos_common libraries while preserving capability boundaries and explicit Bzlmod dependencies.
---

# Common Module Development

## When and prerequisites

Use for common-library or dependency changes. Read `AGENTS.md`,
`.agents/knowledge/bzlmod-modules.md`, relevant source, nearest BUILD files,
and tests. Preserve existing work and public APIs.

## Steps

1. Identify the owning capability and its actual consumers.
2. Keep direct dependencies explicit in MODULE/BUILD declarations.
3. Make the smallest change and update adjacent tests when behavior changes.
4. Select existing focused build/test targets from their BUILD declarations.
   Validate through the consuming workspace's managed environment when testing
   local integration; keep local module overrides in the consumer.

## Acceptance and failures

Selected targets and tests pass; changed public behavior is covered.
Consumer override success is not proof of standalone release acceptance.
Stop at the first actionable failure; do not invent targets, install packages,
or change release versions to bypass errors without approval.
