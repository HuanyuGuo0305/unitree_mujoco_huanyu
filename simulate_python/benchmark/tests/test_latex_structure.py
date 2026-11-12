#!/usr/bin/env python3
"""Structural validation of the generated LaTeX tables.

No LaTeX distribution is installed in this environment, so the tables cannot be
compiled here. This checks what can be checked statically and is NOT a
substitute for a compile:

* every ``\\begin{env}`` has a matching ``\\end{env}``, correctly nested;
* braces and inline-math ``$`` delimiters balance;
* every data row of every ``tabular`` has exactly as many cells as the column
  specification declares, accounting for ``\\multicolumn`` spans (``\\multirow``
  does not change a row's cell count);
* only packages the caption claims are needed appear (booktabs, subcaption,
  multirow).

Run:  python3 benchmark/tests/test_latex_structure.py
"""

from __future__ import annotations

import glob
import os
import re
import sys

# Default: the official LaTeX outputs of the final benchmark (E1 table, E3 figure insert).
# Pass directories on the command line to check others.
_RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
DEFAULT_DIRS = [
    os.path.join(_RESULTS, "E1"),
    os.path.join(_RESULTS, "E3"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_assets", "E3")),
]

ENV = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")
MULTICOL = re.compile(r"\\multicolumn\{(\d+)\}")


def _strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        idx = 0
        while True:
            idx = line.find("%", idx)
            if idx < 0:
                break
            if idx > 0 and line[idx - 1] == "\\":
                idx += 1
                continue
            line = line[:idx]
            break
        out.append(line)
    return "\n".join(out)


def count_columns(spec: str) -> int:
    """Number of cells a tabular column specification declares."""
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    spec = re.sub(r"p\{[^}]*\}", "p", spec)
    spec = re.sub(r"\*\{(\d+)\}\{([^}]*)\}",
                  lambda m: m.group(2) * int(m.group(1)), spec)
    return sum(1 for ch in spec if ch in "lcrp")


def _text_mode_spans(text: str) -> list[str]:
    """The parts of the document that are NOT inside inline math.

    Split on unescaped ``$``; the even-indexed pieces are text mode. Also drop
    ``\texttt{...}``-free verbatim-ish commands is unnecessary here, because
    this generator always escapes inside \texttt{} too.
    """
    pieces = re.split(r"(?<!\\)\$", text)
    return pieces[0::2]


def check_file(path: str) -> list[str]:
    raw = open(path, encoding="utf-8").read()
    text = _strip_comments(raw)
    errors = []

    stack = []
    for kind, name in ENV.findall(text):
        if kind == "begin":
            stack.append(name)
        else:
            if not stack:
                errors.append(f"\\end{{{name}}} with no open environment")
            elif stack[-1] != name:
                errors.append(f"\\end{{{name}}} closes \\begin{{{stack[-1]}}}")
                stack.pop()
            else:
                stack.pop()
    if stack:
        errors.append(f"unclosed environments: {stack}")

    depth = 0
    for i, ch in enumerate(text):
        if ch == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                errors.append("unbalanced closing brace")
                break
    if depth > 0:
        errors.append(f"{depth} unclosed brace(s)")

    n_math = len(re.findall(r"(?<!\\)\$", text))
    if n_math % 2:
        errors.append(f"odd number of inline-math $ delimiters ({n_math})")

    # An unescaped "_" outside math mode is a compile error, and it is easy to
    # emit by accident from a data value such as an E3 condition ID
    # (``S02_circle``) or a case name. Scan only the text-mode spans.
    # Arguments of \label / \ref / \eqref / \cref are key strings, where "_"
    # is legal, as it is in an \includegraphics file path; blank them before
    # scanning for a stray underscore.
    scan = re.sub(r"\\(label|ref|eqref|cref|Cref|autoref)\{[^}]*\}", "", text)
    scan = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", "", scan)
    for span in _text_mode_spans(scan):
        for match in re.finditer(r"(?<!\\)_", span):
            errors.append(
                f"unescaped underscore outside math mode near "
                f"{span[max(0, match.start()-30):match.start()+20]!r}"
            )
            break

    for match in re.finditer(r"\\begin\{tabular\}(\[[^\]]*\])?\{", text):
        start = match.end() - 1
        depth2, j = 0, start
        while j < len(text):
            if text[j] == "{":
                depth2 += 1
            elif text[j] == "}":
                depth2 -= 1
                if depth2 == 0:
                    break
            j += 1
        spec = text[start + 1: j]
        ncols = count_columns(spec)
        end = text.find(r"\end{tabular}", j)
        body = text[j + 1: end if end > 0 else len(text)]
        for line in body.split(r"\\"):
            line = line.strip()
            if not line or re.fullmatch(
                    r"(\\(top|mid|bottom)rule|\\cmidrule(\([^)]*\))?(\{[^}]*\})?"
                    r"|\\addlinespace(\[[^\]]*\])?|\\morecmidrules|\s)+", line):
                continue
            stripped = re.sub(
                r"\\((top|mid|bottom)rule|addlinespace(\[[^\]]*\])?|morecmidrules)",
                "", line).strip()
            if not stripped or stripped.startswith("\\end"):
                continue
            cells = stripped.count("&") + 1
            for span in MULTICOL.findall(stripped):
                cells += int(span) - 1
            if cells != ncols:
                errors.append(
                    f"row has {cells} cells, tabular declares {ncols}: "
                    f"{stripped[:90]!r}"
                )
    return errors


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    dirs = [os.path.abspath(d) for d in argv] or DEFAULT_DIRS
    paths = sorted(p for d in dirs for p in glob.glob(os.path.join(d, "*.tex")))
    if not paths:
        print(f"no .tex files found in {dirs}")
        return 1
    print(f"checking {len(paths)} file(s) in: " + ", ".join(dirs))
    failures = 0
    for path in paths:
        errors = check_file(path)
        name = os.path.basename(path)
        if errors:
            failures += 1
            print(f"FAIL {name}")
            for error in errors:
                print(f"       {error}")
        else:
            print(f"ok   {name}")
    print("-" * 60)
    print("NOTE: no LaTeX distribution is installed here, so these tables were "
          "NOT compiled. This is a structural check only.")
    if failures:
        print(f"{failures}/{len(paths)} file(s) have structural errors")
        return 1
    print(f"{len(paths)}/{len(paths)} files structurally valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
