from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slothctf.generator import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    GenerationError,
    GenerationReport,
    generate_challenge,
)
from slothctf.validator import install_toolchain_windows, validate_challenge


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SlothCTF challenge generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate a challenge")
    gen.add_argument("--name", required=True, help="Challenge folder name")
    gen.add_argument(
        "--prompt",
        default="Create an easy reverse engineering challenge with XOR encryption",
        help="Prompt for challenge generation",
    )
    gen.add_argument(
        "--genre",
        choices=["reverse", "log"],
        default="reverse",
        help="Challenge genre: 'reverse' (C binary) or 'log' (synthetic log analysis)",
    )
    gen.add_argument(
        "--category",
        default=None,
        help="Optional free-form category/flavour hint passed to the model (e.g. 'cryptography', 'reverse engineering')",
    )
    gen.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy", help="Difficulty level")
    gen.add_argument("--surprise", action="store_true", help="Let the model invent the scenario (ignores --prompt)")
    gen.add_argument("--input", type=str, default=None, help="Optional evidence file for file-based generation (reserved)")
    gen.add_argument("--context", type=str, default="", help="Extra context prepended to the prompt")
    gen.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    gen.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Base URL of the Ollama server, e.g. http://10.0.0.2:11434 for a remote "
        "workstation over WireGuard (env: SLOTHCTF_OLLAMA_URL)",
    )
    gen.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="How many generate->compile->solve->fix rounds to allow before giving up "
        "(raise it on a bigger machine / larger model)",
    )
    gen.add_argument(
        "--roundtrip",
        action="store_true",
        help="(reverse genre) also re-check decode(encode(x))==x on a few random inputs, not "
        "just the chosen flag — a property check over samples for added confidence, not a proof",
    )
    gen.add_argument(
        "--adversary",
        action="store_true",
        help="(reverse 'check' challenges) run a budgeted reference attacker (dictionary + "
        "short brute force); a medium/hard challenge it cracks is rejected as too easy",
    )
    gen.add_argument(
        "--adversary-budget",
        default=None,
        metavar="ATTEMPTS,SECONDS",
        help="override the attacker budget (e.g. '2000,20'); default scales with difficulty "
        "(medium=800/8s, hard=2000/20s)",
    )

    validate = subparsers.add_parser("validate", help="Validate a generated challenge")
    validate.add_argument("--dir", required=True, help="Challenge directory to validate")
    validate.add_argument("--install-toolchain", action="store_true", help="Attempt to install the required C compiler if it is missing")

    install = subparsers.add_parser("install-toolchain", help="Install the required C compiler on Windows")
    install.add_argument("--dry-run", action="store_true", help="Print the installation steps without executing them")

    gui = subparsers.add_parser("gui", help="Launch the desktop GUI")
    gui.add_argument("--no-splash", action="store_true", help="Launch without the startup message")

    lst = subparsers.add_parser("list", help="List generated challenges from the catalog")
    lst.add_argument("--category", default=None, help="Filter by category")
    lst.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None, help="Filter by difficulty")
    lst.add_argument("--kind", choices=["decode", "check"], default=None, help="Filter by challenge kind")
    lst.add_argument("--search", default=None, help="Free-text search over name/title/category")
    lst.add_argument("--json", action="store_true", help="Output JSON instead of a table")
    lst.add_argument("--no-scan", action="store_true", help="Do not rescan generated/ before listing")

    st = subparsers.add_parser("stats", help="Show offline metrics over the generated challenges")
    st.add_argument("--json", action="store_true", help="Output JSON instead of a report")

    web = subparsers.add_parser("web", help="Launch the web GUI to browse the catalog")
    web.add_argument("--host", default="127.0.0.1", help="Host to bind")
    web.add_argument("--port", type=int, default=8000, help="Port to bind")
    web.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        input_path = Path(args.input) if args.input else None
        adv_budget = None
        if args.adversary_budget:
            import math as _math
            try:
                a_str, s_str = args.adversary_budget.split(",")
                a, s = int(a_str), float(s_str)
            except ValueError:
                print("--adversary-budget must be 'ATTEMPTS,SECONDS' (e.g. 2000,20)", file=sys.stderr)
                raise SystemExit(2)
            if a < 1 or not _math.isfinite(s) or s <= 0:
                print("--adversary-budget needs attempts>=1 and seconds>0 (finite)", file=sys.stderr)
                raise SystemExit(2)
            adv_budget = (a, s)
        try:
            report = generate_challenge(
                args.name,
                prompt=args.prompt,
                challenge_type=args.category,
                difficulty=args.difficulty,
                input_path=input_path,
                context=args.context,
                model=args.model,
                surprise=args.surprise,
                max_attempts=args.max_attempts,
                ollama_url=args.ollama_url,
                genre=args.genre,
                roundtrip=args.roundtrip,
                adversary=args.adversary,
                adversary_budget=adv_budget,
                return_report=True,
            )
        except GenerationError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc

        assert isinstance(report, GenerationReport)
        print(report.as_text())
        if not report.success:
            raise SystemExit(1)
        return

    if args.command == "validate":
        if args.install_toolchain:
            result = install_toolchain_windows(dry_run=False)
            print(result)
        result = validate_challenge(Path(args.dir))
        print(result)
        if not result.success:
            raise SystemExit(1)
        return

    if args.command == "install-toolchain":
        result = install_toolchain_windows(dry_run=args.dry_run)
        print(result)
        return

    if args.command == "gui":
        try:
            from slothctf.gui import launch_gui
        except ModuleNotFoundError as exc:
            print("GUI mode requires tkinter, which is not available in this environment.", file=sys.stderr)
            raise SystemExit(1) from exc
        launch_gui(show_splash=not args.no_splash)
        return

    if args.command == "list":
        from slothctf import catalog
        if not args.no_scan:
            catalog.index_generated()
        rows = catalog.query(
            category=args.category, difficulty=args.difficulty,
            kind=args.kind, search=args.search,
        )
        if args.json:
            import json as _json
            print(_json.dumps(rows, indent=2))
            return
        if not rows:
            print("No challenges found. Generate some with 'python main.py generate ...'.")
            return
        print(f"{'NAME':22} {'KIND':7} {'DIFF':7} {'CATEGORY':22} FLAG")
        print("-" * 86)
        for r in rows:
            print(f"{r['name'][:22]:22} {r['kind']:7} {r['difficulty'][:7]:7} "
                  f"{(r['category'] or '')[:22]:22} {r['flag']}")
        print(f"\n{len(rows)} challenge(s).")
        return

    if args.command == "stats":
        from slothctf import stats
        data = stats.collect()
        if args.json:
            import json as _json
            print(_json.dumps(data, indent=2))
        else:
            print(stats.render(data))
        return

    if args.command == "web":
        from slothctf.web import launch_web
        launch_web(host=args.host, port=args.port, open_browser=not args.no_browser)
        return


if __name__ == "__main__":
    main()
