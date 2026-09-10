# Hermes Agent skill tree — snapshot

Snapshot of the Hermes Agent skill tree as of the llm-toolkit assembly date.
This is a reference copy, not the live tree. The live tree lives in
`~/AppData/Local/hermes/skills/` (Windows) or `~/.hermes/skills/` (Unix).

## What skills are

Skills are reusable procedure documents that Hermes loads into a session when
the task matches. Each skill is a `SKILL.md` file (optionally with reference
files, templates, and scripts). Skills encode specialized knowledge — API
endpoints, tool-specific commands, proven workflows — that outperforms a
generic approach.

A skill's first ~57 characters are its trigger: "Use when <trigger>."

## Snapshot layout

This snapshot mirrors the top-level skill categories and the hermes-agent hub
skill. Full skill bodies are in the original tree; this copy is for reference
and provenance.

### Categories (top-level)

- `animate-photo-gif` — photo GIFs where the subject moves
- `autonomous-ai-agents` — spawning/orchestrating agent subworkflows
  (claude-code, codex, computer-use, delegated-agent-verification, hermes-agent,
  memory-os-local, merge-reconciler, opencode, resuming-agent-coding-work)
- `blockchain` — DERO/DVM/crypto tooling (derohe, mycelium, obscura, relayos,
    cross-chain, dYdX, ZKP, mining, consensus)
- `business` — accounting practice rollups, fintech, bookkeeping automation,
    drone delivery, ecommerce margins, Ledger Sentinel, CPA practice tooling
- `community` — architecture, code review, docx, xlsx, pdf, pptx, markdown,
    scientific viz, ML (scikit-learn, PyMC, pymoo, polars, networkx, dask),
    writing, Go skills (20+ go-* skills), Rust skills (m01-m09), npm/browser tools
- `creative` — ASCII art, diagrams (drawio, excalidraw, architecture), video,
    ComfyUI, p5js, design docs, music
- `crypto` — donation wallets, crypto implementations (2020+)
- `design` — cold-climate earthship design
- `email` — inbox triage, Himalaya CLI
- `faceless-video-channel-production` — VidRush-style channels
- `games` — word-game solvers
- `github` — gh CLI workflow (auth, PRs, issues, repos, conventional commits)
- `homestead` — cold-climate earthships, market gardens
- `infrastructure` — Hetzner, DNS/email bringup, server economics, vehicle scan
- `marketing` — social content bots, YouTube VidRush portfolios
- `media` — video assembly, GIF search, songsee, YouTube transcripts
- `mlops` — HuggingFace hub, local GPU LLM, ComfyUI local setup
- `mlops-inference` — ComfyUI local setup
- `mlops/evaluation` — W&B experiment tracking
- `mlops/inference` — llama.cpp, GGUF inference
- `note-taking` — Obsidian vault
- `productivity` — Airtable, Box, docx, xlsx, formula workbooks, Google Workspace,
    maps, meeting action items, Notion, PDF, PowerPoint, price monitors,
    session librarian, OCR, Teams, weekly review, Windows audio
- `realestate` — land search
- `research` — arXiv, blogwatcher, competitor news, grounded citations,
    LLM wiki, PQ research, quantitative reporting
- `security` — Flipper Zero, novel harness, OSS-Fuzz, PQ PoW research,
    security commando, security testing stack, security toolkit ops
- `smart-home` — OpenHue (Philips Hue)
- `software-development` — Android build/ship, casino frontend, Chrome CDP,
    clean-room porting, codebase inspection, CUDA optimization, DERO tooling,
    differential porting, dogfood, elliptic curve arithmetic, empirical validation,
    fullstack model stack, git history rewrite, GitHub, Godot, Hermes skill
    authoring, desktop DOM inspection, Node debugger, PIR lookup,
    plan/test-driven/spike/systematic-debug, relay-os, rustls mTLS, secure
    messenger release, simplify-code, Windows game launch, word-game tools
- `troubleshooting` — context hygiene, Hermes cost observability, Hermes
    dashboard, Hermes runtime troubleshooting, model selection guardrail,
    Windows audio, Windows process hygiene, Windows user machine ops
- `web` — blocked page recovery

## Hub skill: hermes-agent

The `hermes-agent` skill is the hub. It covers:

- Identity: Hermes Agent is an open-source AI agent framework by Nous Research.
  Runs in terminal, native desktop app, messaging platforms, IDEs.
- Surfaces: CLI, Ink TUI, native Electron desktop app, web dashboard, ACP server.
- Provider-agnostic: OpenRouter, Anthropic, OpenAI, Google, DeepSeek, xAI,
  local models, 20+ others.
- Self-improving via skills; persistent memory; multi-platform gateway;
  profiles (isolated configs); extensible (plugins, MCP, cron, themes, pets).

Key paths (from the hub skill):

```
~/.hermes/config.yaml       Main configuration (settings — never secrets)
~/.hermes/.env              API keys and secrets ONLY
$HERMES_HOME/skills/        Installed skills
~/.hermes/skins/            Custom themes
~/.hermes/desktop-plugins/  Desktop app UI plugins
~/.hermes/tui-widgets/      TUI widget apps
~/.hermes/pets/             Installed pet mascots
~/.hermes/state.db          Canonical session store (SQLite + FTS5)
~/.hermes/sessions/         Gateway routing index, request dumps, transcripts
~/.hermes/logs/             Gateway and error logs
```

Docs: https://hermes-agent.nousresearch.com/docs/

## How the snapshot was taken

Directory listing of `~/AppData/Local/hermes/skills/` at assembly time.
Skill bodies are the original `SKILL.md` files; this snapshot preserves the
structure and the hub skill body for reference.
