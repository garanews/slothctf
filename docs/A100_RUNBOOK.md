# A100 campaign runbook

The A100 is billed **per hour**, so the goal is one focused session: everything is
prepared offline (config written and validated, code committed, campaign tested on
mocks/Docker), then a single uninterrupted batch runs, and the instance is stopped the
moment it finishes. No idle hours, no repeated power cycles.

Prepare the config and run the offline tests (`python -m unittest tests.test_batch`)
**before** starting the instance.

## Steps

1. **Start the A100 instance** (provider console or CLI). Note its internal IP.

2. **Verify Ollama and the connection.**
   - On the instance: `ollama serve` is up and the model is pulled:
     `ollama list` shows `qwen2.5-coder:32b`.
   - Establish a **stable, authenticated tunnel** to it (see "Connection" below).
   - From your machine, confirm reachability and the digest:
     ```bash
     curl -s http://<ollama-endpoint>/api/tags | python -m json.tool | grep -A2 qwen2.5-coder
     ```
     The digest must match the one pinned in the campaign's `manifest.json` (or, for a new
     campaign, the one you intend to freeze).

3. **Start the campaign with one command:**
   ```bash
   PYTHONPATH=. python -m slothctf.batch run --config examples/campaigns/agentic_recover_elf_easy_15.json
   ```
   (Set the config's `ollama_url` to your tunnel endpoint first.) Run it under `nohup` /
   `tmux` so a dropped SSH session does not kill it.

4. **Check status and results** at any time:
   ```bash
   tail -f generated/<dest>/campaign.jsonl        # live per-run events
   PYTHONPATH=. python -m slothctf.batch report --dir generated/<dest>   # refresh report.md/csv
   ```

5. **Resume after any interruption** (tunnel drop, kill, Docker hiccup):
   ```bash
   PYTHONPATH=. python -m slothctf.batch resume --dir generated/<dest> --ollama-url http://<new-endpoint>
   ```
   Completed runs are not repeated; the interrupted attempt is preserved; the digest is
   re-verified. Pass `--ollama-url` only if the tunnel moved.

6. **Stop the A100 instance** as soon as the report shows the campaign done — do not leave
   it idling.

## Connection

- Prefer a **stable, authenticated channel** — e.g. SSH local port-forward:
  ```bash
  ssh -N -L 11434:127.0.0.1:11434 user@<a100-host>
  # then ollama_url = http://127.0.0.1:11434
  ```
  over an ephemeral Cloudflare quick tunnel (those rotate URLs and drop, which is what
  interrupted an earlier campaign).
- **Never expose Ollama publicly without protection.** Bind it to localhost on the
  instance and reach it only through the tunnel; if it must listen on a network interface,
  put authentication / an allow-list in front of it.

## Auto-stop (optional, not automated here)

The instance can be stopped programmatically via the provider's API/CLI at the end of the
campaign, to avoid idle billing. **This runbook does not perform any automatic shutdown.**
Wire an auto-stop only with explicit authorization, and only after confirming the report
is complete and the artifacts are saved locally — a premature stop would lose an in-flight
generation (recoverable via `resume`, but wasteful).
