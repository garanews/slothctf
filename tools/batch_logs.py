"""Batch-generate a broad set of log-analysis challenges in one run.

Covers the common DFIR / CTF log families (web, proxy/WAF, DNS, firewall, Zeek,
mail, containers, k8s, cloud/IAM, SSO, endpoint/Sysmon, PowerShell, auditd, DB).
Each prompt asks the model for a realistic synthetic log plus several questions;
the self-verifying log pipeline checks every task (extractive + computed).

Usage:
    python tools/batch_logs.py --model qwen2.5-coder:32b --ollama-url https://your-ollama
    python tools/batch_logs.py --model qwen2.5-coder:32b --only dns_tunnel,cloudtrail_key
    python tools/batch_logs.py --model qwen2.5-coder:32b --max-attempts 5

Add  --input <logfile>  to run every prompt as ANALYSIS of that one provided log
instead of synthesis (useful to ask many questions about a real log you have).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slothctf.generator import GenerationError, generate_challenge  # noqa: E402

# Every prompt ends with the same ask so the model emits several good tasks.
_ASK = (
    " Produce 4-6 questions mixing EXTRACTIVE facts (values that appear verbatim, "
    "e.g. an IP, a domain, a username, a path) and COMPUTED aggregates (counts, "
    "number of distinct values, a max). Every question must have a single "
    "unambiguous answer (no ties)."
)

PROMPTS: list[tuple[str, str]] = [
    ("ssh_bruteforce",
     "A Linux auth.log showing an SSH brute-force: one IP tries many usernames and passwords, "
     "then fail2ban bans it."),
    ("nginx_sqli",
     "An Nginx access log where one IP performs SQL injection attempts (UNION SELECT, OR 1=1) "
     "and finally gets HTTP 200 on an uploaded webshell under /uploads/."),
    ("apache_lfi",
     "An Apache access log with Local File Inclusion / path traversal attempts "
     "(../../etc/passwd) from one attacker IP, mixed with normal traffic."),
    ("waf_modsecurity",
     "A ModSecurity (WAF) audit log: several blocked payloads with their triggered rule IDs "
     "from one source IP, and exactly one request that slipped through (bypass)."),
    ("dns_tunnel",
     "A DNS query log showing data exfiltration via DNS tunneling: many long base32-looking "
     "subdomain TXT/A queries to a single C2 domain, amid normal queries."),
    ("dns_dga",
     "A DNS resolver log where malware uses a Domain Generation Algorithm: many random-looking "
     "domains resolving to one C2 IP, among normal lookups."),
    ("firewall_portscan",
     "An iptables/pfSense firewall log showing a SYN port scan from one source IP hitting many "
     "destination ports in a short window, among normal allowed traffic."),
    ("zeek_conn",
     "A Zeek conn.log in TSV format (ts, uid, orig_h, orig_p, resp_h, resp_p, proto, service, "
     "duration, orig_bytes, resp_bytes, ...) where one internal host beacons to an external IP "
     "and exfiltrates a large amount of bytes."),
    ("postfix_phishing",
     "A Postfix mail log for a phishing campaign: one sending IP delivers spoofed-sender mail to "
     "many recipients, with some rejected and some accepted."),
    ("dovecot_spray",
     "A Dovecot IMAP auth log showing password spraying: one source IP tries the same password "
     "against many different mailboxes, succeeding on exactly one."),
    ("docker_escape",
     "A Docker daemon log showing a privileged container being started and a container-escape / "
     "host mount attempt, among normal container lifecycle events."),
    ("k8s_audit",
     "A Kubernetes audit log in JSON lines: a suspicious actor creates a privileged pod and runs "
     "kubectl exec into it (cryptominer), among routine API calls. Each line is one JSON event."),
    ("cloudtrail_key",
     "An AWS CloudTrail log in JSON lines: a stolen IAM access key suddenly calls RunInstances "
     "from a new source IP, among normal Describe/List calls. Each line is one JSON event."),
    ("gcp_audit",
     "A GCP Cloud Audit log in JSON lines: an identity performs IAM privilege escalation "
     "(setIamPolicy granting owner) from an unusual IP, among normal activity."),
    ("okta_impossible_travel",
     "An Okta/Entra SSO sign-in log in JSON lines. INCLUDE a country field per event. One user "
     "signs in from two distant countries minutes apart (impossible travel); that user is the "
     "answer. Make the geo data explicit in the log so it is derivable."),
    ("sysmon_lolbas",
     "A Windows Sysmon log exported as text/XML-like lines (Event ID 1 process creation) showing a "
     "LOLBAS abuse: a trusted binary (e.g. certutil or mshta) spawned by an office app downloads a "
     "payload, among normal process events."),
    ("powershell_obf",
     "A Windows PowerShell operational log containing an encoded/obfuscated command "
     "(-EncodedCommand base64) that decodes to a download cradle, among benign PowerShell use."),
    ("auditd_suid",
     "A Linux auditd log showing privilege escalation: abuse of a SUID binary and a write to "
     "/etc/shadow by a non-root user, among normal syscalls."),
    ("mysql_exfil",
     "A MySQL general query log showing mass data exfiltration: one user runs large SELECT * "
     "queries against a sensitive table far more than anyone else, among normal queries."),
    ("mongodb_dump",
     "A MongoDB log showing an unauthorized mass dump: one client reads an entire sensitive "
     "collection and changes a user's privileges, among normal operations."),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-generate log challenges")
    ap.add_argument("--model", default="qwen2.5-coder:32b")
    ap.add_argument("--ollama-url", default=None, help="Ollama base URL (default: env / localhost)")
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--only", default=None, help="Comma-separated subset of names to run")
    ap.add_argument("--input", default=None, help="Run every prompt as analysis of this one provided log")
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    prompts = [(n, p) for n, p in PROMPTS if not only or n in only]
    url = args.ollama_url
    kwargs = {"model": args.model, "max_attempts": args.max_attempts, "genre": "log", "return_report": True}
    if url:
        kwargs["ollama_url"] = url
    if args.input:
        kwargs["input_path"] = args.input

    results, t0 = [], time.time()
    for i, (name, prompt) in enumerate(prompts, 1):
        start = time.time()
        try:
            rep = generate_challenge(name, prompt=prompt + _ASK, difficulty="medium", **kwargs)
            last = rep.attempts[-1].stage if rep.attempts else "?"
            ok = rep.success
            line = (f"[{i:2}/{len(prompts)}] {name:24} {'VALID' if ok else 'FAIL '} "
                    f"attempts={rep.attempts_used} last={last} {int(time.time()-start)}s")
        except GenerationError as exc:
            ok = False
            line = f"[{i:2}/{len(prompts)}] {name:24} ERROR {str(exc)[:70]} {int(time.time()-start)}s"
        results.append((name, ok))
        print(line, flush=True)

    valid = sum(1 for _, ok in results if ok)
    print(f"\n==== SUMMARY ====  VALID {valid}/{len(results)}  total {int(time.time()-t0)}s", flush=True)
    for name, ok in results:
        print(("  OK  " if ok else "  --  ") + name, flush=True)
    return 0 if valid == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
