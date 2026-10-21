from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from slothctf.generator import generate_challenge
from slothctf.validator import validate_challenge


def launch_gui(show_splash: bool = True) -> None:
    app = SlothCTFGUI(show_splash=show_splash)
    app.run()


class SlothCTFGUI:
    def __init__(self, show_splash: bool = True) -> None:
        self.root = tk.Tk()
        self.root.title("SlothCTF Challenge Factory")
        self.root.geometry("820x620")
        self.root.minsize(760, 560)
        self.show_splash = show_splash
        self.input_path: Path | None = None

        self._build_widgets()

    def _build_widgets(self) -> None:
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="SlothCTF", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w", pady=(0, 10))

        subtitle = ttk.Label(main, text="Local AI challenge factory for reverse, forensics, OSINT, and stego")
        subtitle.pack(anchor="w", pady=(0, 10))

        form = ttk.LabelFrame(main, text="Challenge details", padding=12)
        form.pack(fill="x", pady=(0, 12))

        ttk.Label(form, text="Name").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=6)
        self.name_var = tk.StringVar(value="demo_challenge")
        ttk.Entry(form, textvariable=self.name_var, width=40).grid(row=0, column=1, sticky="ew", pady=6)

        ttk.Label(form, text="Type").grid(row=1, column=0, sticky="w", padx=(0, 12), pady=6)
        self.type_var = tk.StringVar(value="reverse")
        ttk.Combobox(
            form,
            textvariable=self.type_var,
            values=["reverse", "forensics", "stego", "osint", "surprise"],
            state="readonly",
            width=30,
        ).grid(row=1, column=1, sticky="ew", pady=6)

        ttk.Label(form, text="Difficulty").grid(row=2, column=0, sticky="w", padx=(0, 12), pady=6)
        self.difficulty_var = tk.StringVar(value="easy")
        ttk.Combobox(
            form,
            textvariable=self.difficulty_var,
            values=["easy", "medium", "hard"],
            state="readonly",
            width=30,
        ).grid(row=2, column=1, sticky="ew", pady=6)

        ttk.Label(form, text="Prompt").grid(row=3, column=0, sticky="nw", padx=(0, 12), pady=6)
        self.prompt_var = tk.Text(form, height=5, wrap="word")
        self.prompt_var.insert("1.0", "Create an easy reverse engineering challenge with XOR encryption")
        self.prompt_var.grid(row=3, column=1, sticky="ew", pady=6)

        ttk.Label(form, text="Input file").grid(row=4, column=0, sticky="w", padx=(0, 12), pady=6)
        self.input_label = tk.StringVar(value="No file selected")
        input_row = ttk.Frame(form)
        input_row.grid(row=4, column=1, sticky="ew", pady=6)
        ttk.Entry(input_row, textvariable=self.input_label, state="readonly", width=35).pack(side="left", fill="x", expand=True)
        ttk.Button(input_row, text="Browse", command=self._choose_input_file).pack(side="left", padx=(8, 0))

        ttk.Label(form, text="Context").grid(row=5, column=0, sticky="nw", padx=(0, 12), pady=6)
        self.context_var = tk.StringVar(value="")
        ttk.Entry(form, textvariable=self.context_var, width=40).grid(row=5, column=1, sticky="ew", pady=6)

        form.columnconfigure(1, weight=1)

        actions = ttk.Frame(main)
        actions.pack(fill="x", pady=(0, 12))
        ttk.Button(actions, text="Generate challenge", command=self._generate).pack(side="left", padx=(0, 10))
        ttk.Button(actions, text="Validate challenge", command=self._validate).pack(side="left")

        self.output = tk.Text(main, height=12, wrap="word")
        self.output.insert("1.0", "Ready. Generate a challenge or validate a generated project.\n")
        self.output.configure(state="disabled")
        self.output.pack(fill="both", expand=True)

        if self.show_splash:
            self._show_splash()

    def _show_splash(self) -> None:
        messagebox.showinfo(
            "SlothCTF",
            "This is the local challenge factory.\nGenerate a challenge, then validate it with the built-in checker.",
        )

    def _choose_input_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Select evidence file",
            filetypes=[("All files", "*.*"), ("Text files", "*.txt *.log *.csv *.json")],
        )
        if not chosen:
            return
        self.input_path = Path(chosen)
        self.input_label.set(str(self.input_path))

    def _generate(self) -> None:
        challenge_name = self.name_var.get().strip() or "demo_challenge"
        prompt = self.prompt_var.get("1.0", "end").strip() or "Create an easy reverse engineering challenge with XOR encryption"
        challenge_type = self.type_var.get()
        difficulty = self.difficulty_var.get()
        context = self.context_var.get().strip()
        try:
            result = generate_challenge(
                challenge_name,
                prompt=prompt,
                challenge_type=challenge_type,
                difficulty=difficulty,
                input_path=self.input_path,
                context=context,
                surprise=(challenge_type == "surprise"),
            )
            self._set_output(f"Generation successful.\nOutput directory: {result}\n")
        except Exception as exc:  # pragma: no cover - GUI error surface
            self._set_output(f"Generation failed.\n{exc}\n")
            messagebox.showerror("Generation issue", str(exc))

    def _validate(self) -> None:
        challenge_name = self.name_var.get().strip() or "demo_challenge"
        target_dir = Path("generated") / challenge_name
        if not target_dir.exists():
            self._set_output(f"Validation skipped: no generated project found at {target_dir}.\n")
            messagebox.showwarning("Missing project", f"No challenge folder was found at {target_dir}.")
            return

        try:
            result = validate_challenge(target_dir)
            self._set_output(f"Validation result:\n{result}\n")
        except Exception as exc:  # pragma: no cover - GUI error surface
            self._set_output(f"Validation failed.\n{exc}\n")
            messagebox.showerror("Validation issue", str(exc))

    def _set_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def run(self) -> None:
        self.root.mainloop()
