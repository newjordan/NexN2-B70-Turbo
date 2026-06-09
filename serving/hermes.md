# NX2 B70 Turbo + Hermes Agent

[Hermes Agent](https://github.com/nousresearch/hermes-agent) is an OpenAI-compatible autonomous agent. Point it at the local NX2 server.

## Install

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
source ~/.bashrc
```

## Point it at NX2 (custom provider)

```bash
hermes config set model.provider custom
hermes config set model.default nex-n2-mini
hermes config set model.base_url http://127.0.0.1:8090/v1
hermes config set model.context_length 131072
hermes config set OPENAI_API_KEY sk-local   # local server is no-auth; the OpenAI SDK just needs a non-empty key
```

This writes `~/.hermes/config.yaml` (and the key to `~/.hermes/.env`). Verify:

```bash
hermes config show | grep -i -A1 model
hermes -z "In one sentence, confirm you are online."   # one-shot smoke test
hermes                                                  # interactive
```

Notes:
- `provider: custom` also accepts the aliases `llamacpp` / `vllm` / `ollama`.
- NX2 is a reasoning model (emits a `<think>` trace); leave `max_tokens` unset to use the native ceiling.
- Make sure `serving/llama-server.sh` is running first (`:8090`).
