# Agent Guide

## Rules

- Read the relevant source, configuration, and tests before making changes.
- Follow existing architecture, naming, and Bazel conventions.
- Keep changes scoped to the requested work and preserve unrelated changes.
- Read `.agents/knowledge/` when working on repository architecture or conventions.

## Knowledge

- Index and ownership: `.agents/knowledge/README.md`
- Bzlmod module design: `.agents/knowledge/bzlmod-modules.md`

## Agent layout

- `.github/` owns GitHub governance; `AGENTS.md` owns shared working rules.
- `.agents/skills/README.md` indexes task workflows.
- `.agents/knowledge/` owns durable repository knowledge.
- `.agents/notes/README.md` describes ignored temporary investigations.
