---
title: Workspace Structure
description: Canonical directory layout for organized workspaces
---

# Workspace Structure Convention

This workspace is organized into categories. Place new items in the correct category directory instead of the workspace root.

| Category    | Directory      | Contents                                  | Examples                                   |
| ----------- | -------------- | ----------------------------------------- | ------------------------------------------ |
| Platform    | `platform/`    | Infrastructure source code                | Core framework, memory system, desktop env |
| Skill Repos | `skill-repos/` | Source code for deployed skills           | TTS skill, LLM skill, memory skill         |
| Projects    | `projects/`    | Operator projects and applications        | Web apps, services, tools                  |
| Media       | `media/`       | Video frames, screenshots, media projects | Video renders, image sets                  |
| Outputs     | `outputs/`     | Skill and cron job output directories     | Monitor output, scraper output             |
| Archives    | `archives/`    | Completed or dormant artifacts            | Zip files, old releases                    |
| Docs        | `docs/`        | Planning documents and analysis files     | Plans, analyses, specs                     |

## Rules

- New git repos / projects → `projects/`
- New skill source code → `skill-repos/`
- Video/image generation output → `media/`
- Monitoring/scraping output → `outputs/`
- Bootstrap files (AGENTS.md, SOUL.md, TOOLS.md, etc.) stay at workspace root
