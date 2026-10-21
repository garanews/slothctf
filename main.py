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

    validate = subparsers.add_parser("validate", help="Validate a generated challenge")
    validate.add_argument("--dir", required=True, help="Challenge directory to validate")
    validate.add_argument("--install-toolchain", action="store_true", help="Attempt to install the required C compiler if it is missing")

    install = subparsers.add_parser("install-toolchain", help="Install the required C compiler on Windows")
    install.add_argument("--dry-run", action="store_true", help="Print the installation steps without executing them")

    gui = subparsers.add_parser("gui", help="Launch the desktop GUI")
    gui.add_argument("--no-splash", action="store_true", help="Launch without the startup message")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        input_path = Path(args.input) if args.input else None
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


if __name__ == "__main__":
    main()
