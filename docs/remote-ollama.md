# Running SlothCTF against a remote Ollama

SlothCTF talks to Ollama over HTTP. The endpoint is configurable, so the model
can run anywhere you can reach over the network:

```bash
python main.py generate --name test \
  --prompt "Create an easy XOR reverse engineering challenge" \
  --ollama-url https://your-ollama-host \
  --model qwen2.5-coder:7b \
  --max-attempts 6
```

You can also set it once for the shell:

```bash
export SLOTHCTF_OLLAMA_URL=https://your-ollama-host   # bash
$env:SLOTHCTF_OLLAMA_URL="https://your-ollama-host"    # PowerShell
```

> The compiler (gcc) and the solver (python) still run **locally** — only the
> model inference is remote. So you still need MinGW/gcc on the machine that
> runs `main.py`.

---

## Option A — Google Colab (free tier) + cloudflared tunnel

Good for a quick test. Free Colab gives a T4 (16 GB), which comfortably runs a
7B model. Pick a **coder-tuned** model — it writes consistent C + Python solvers
far better than a small general model.

Set *Runtime → Change runtime type → T4 GPU* first, then run these cells:

```python
# 1. Install Ollama
!curl -fsSL https://ollama.com/install.sh | sh
```

```python
# 2. Start the server, listening on all interfaces and accepting any origin
import os, subprocess, time
os.environ["OLLAMA_HOST"] = "0.0.0.0:11434"
os.environ["OLLAMA_ORIGINS"] = "*"
subprocess.Popen(["ollama", "serve"],
                 env={**os.environ},
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)
```

```python
# 3. Pull a coder model (one-off download)
!ollama pull qwen2.5-coder:7b
```

```python
# 4. Expose it with a no-signup Cloudflare quick tunnel (robust URL capture)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
import subprocess, re, threading, time
p = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:11434"],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
# cloudflared logs several lines that mention trycloudflare.com; only the real
# tunnel address starts with https://, so match that and keep draining the log
# in a thread so the pipe buffer never fills and stalls cloudflared.
url = {"v": None}
def _drain():
    for line in p.stdout:
        m = re.search(r"https://[-\w]+\.trycloudflare\.com", line)
        if m and not url["v"]:
            url["v"] = m.group(0)
threading.Thread(target=_drain, daemon=True).start()
for _ in range(40):
    if url["v"]:
        break
    time.sleep(1)
print("TUNNEL URL:", url["v"] or "(not ready yet, re-run this cell in a few seconds)")
```

Copy the printed `https://<random>.trycloudflare.com` URL and use it on your
laptop as `--ollama-url`.

**Security:** that URL is public and unauthenticated — anyone who has it can use
your GPU. Keep it private and stop the Colab runtime when you're done. For a bit
of protection use ngrok with an authtoken and basic-auth instead of the open
quick tunnel.

**Limits:** Colab free disconnects after ~90 min idle / ~12 h max and the GPU
isn't guaranteed. Treat it as a test bench, not a stable service.

---

## Option B — a workstation on your LAN over WireGuard

Best for real use when the machine (e.g. a Z840) is on. No tunnel needed: you
reach it directly over the WireGuard tunnel.

On the workstation, make Ollama listen on the network instead of just localhost:

```bash
# Linux (systemd): override the service
sudo systemctl edit ollama
#   [Service]
#   Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
```

```powershell
# Windows: set a machine env var, then restart Ollama
setx OLLAMA_HOST "0.0.0.0:11434" /M
```

With 512 GB RAM the Z840 can hold very large models in RAM (CPU inference is
slower on Broadwell Xeons — only AVX2 — but the capacity is there). Try a larger
coder model, e.g. `qwen2.5-coder:32b`.

Then from the laptop, over WireGuard, point at the workstation's WG IP:

```bash
python main.py generate --name test \
  --ollama-url http://10.0.0.2:11434 \
  --model qwen2.5-coder:32b \
  --max-attempts 8
```

Binding to `0.0.0.0` exposes Ollama (no auth) on every interface the machine
has. Prefer binding to the WireGuard interface IP specifically, or firewall
11434 so it is only reachable from the WireGuard subnet.
