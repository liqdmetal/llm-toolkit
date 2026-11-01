# FAQ — llm-toolkit

**What is this?**
A thin shared LLM client that wires free endpoints — local Ollama first,
Nous Portal and OpenRouter as free fallbacks — so scripts never burn paid
credits while keeping a local-first, private default.

**Is this a wrapper around Hermes Agent?**
No. It reads the Hermes credential store for portal auth (the same OAuth
token / API keys Hermes uses), but it is a standalone client: no Hermes
runtime, no agent process, plain functions on PYTHONPATH.

**Which models does it route to?**
Local Ollama (whatever you serve — verified with Qwen-class and
llama3.2 models), Nous Portal free models (`openai/gpt-5.6-luna`,
`solar-pro-4-mini`, `deepseek-ai/deepseek-v4-flash`), OpenRouter free
models (e.g. `nvidia/nemotron-3-*-a12b:free`).

**How does the router behave when everything is up?**
`prefer_local=True` tries local first; on local failure (or when the
caller opts into remote) it falls through to the chosen aux portal, and
if that fails it tries the other free portal automatically.

**Where do secrets live?**
Never in the repo. The client reads credentials at runtime from the
Hermes credential store or from environment variables. A checkout
contains nothing to leak.

**Is the fallback path verified?**
Yes — the routing ladder (local → OpenRouter, local → Nous) and the
gateway fallback are covered by the smoke tests in `reference/`.

**Do I need a GPU?**
No. Local is optional; without Ollama the client simply uses the free
portals.
