# Chris Skills - Architecture Guide

This document provides architecture guidance for Claude Code agents working with this repository.

## Repository Structure

```
chris-skills/
├── .claude-plugin/
│   └── marketplace.json    # Plugin registry & metadata
├── skills/
│   └── chris-stock-master/ # Financial report analyzer skill
│       ├── SKILL.md        # Skill definition & documentation
│       ├── scripts/        # Python implementations
│       └── references/     # Reference documentation
├── README.md               # User-facing documentation
├── CLAUDE.md               # This file - agent guidance
└── LICENSE
```

## Skill Organization Pattern

Each skill follows this structure:

```
skills/<skill-name>/
├── SKILL.md           # Required: YAML frontmatter + documentation
├── scripts/           # Executable scripts (Python/TypeScript)
├── references/        # Reference docs, templates, guides
└── [config files]     # Skill-specific configuration
```

### SKILL.md Format

```yaml
---
name: chris-<skill-name>
description: One-line description for marketplace
---

# Skill Title
[Documentation follows]
```

## Naming Conventions

- All skills use `chris-` prefix to avoid naming conflicts
- Directory names use kebab-case: `chris-stock-master`
- Script files use snake_case: `run_report.py`

## Adding New Skills

1. Create directory: `skills/chris-<name>/`
2. Add `SKILL.md` with YAML frontmatter
3. Add `scripts/` for executables
4. Add `references/` for documentation
5. Register in `.claude-plugin/marketplace.json`

## Current Skills

### chris-stock-master

Financial report analyzer supporting US/JP/CN/HK markets.

- **Language**: Python 3.11+
- **Dependencies**: yfinance, akshare, pandas, matplotlib
- **Entry point**: `scripts/run_report.py`
- **Output**: JSON data files, Markdown reports, PNG charts
