Agents that come with agent-env, each an agent folder a bundle builds and deploys like its own.

`agents/claude-code/` runs Anthropic's [Claude Code](https://code.claude.com/docs/en/headless) CLI behind an A2A
server: the prompt goes to `claude -p`, the task env's MCP servers to `--mcp-config`, and its stream-json output is the
turn's trajectory. It's the default agent, so a `deploy_agent` or an agent judge that names no agent deploys it. Name it
from another bundle as `@local/agentenv-framework/agents/claude-code`.

It needs Docker, and a model endpoint serving Anthropic's Messages API, `<base>/v1/messages`, which a LiteLLM gateway
does: the `[model]` base URL and key reach it as `ANTHROPIC_BASE_URL` and a bearer token, and `[model.headers]` are sent
on every model call. Building the image downloads Claude Code, which is Anthropic's software under its
[Commercial Terms](https://www.anthropic.com/legal/commercial-terms); the image is built where it runs and never
published. The CLI's version is the Dockerfile's `CLAUDE_CODE_VERSION`.

Run its smoke task with `agent-env run agents`.
