"""Paper tables at their native type size, with literal numeric cells.

Wide six-dataset comparisons use two stacked three-dataset panels. Nothing
scales the typeset table: the manuscript's ``\\small`` is the font floor.
"""

from __future__ import annotations

import re


def _panels(tabular: str) -> str:
    lines = tabular.splitlines()
    rows = [line for line in lines if "&" in line]
    if not rows:
        return tabular
    groups = re.findall(r"\\multicolumn\{(\d+)\}\{c\}\{\\textbf\{[^}]*\}\}", rows[0])
    if len(groups) != 6 or len(set(groups)) != 1:
        return tabular
    n = int(groups[0])
    if len(rows[0].split("&")) != 7:
        return tabular
    panels = []
    for start in (0, 3):
        selected = [0] + list(range(1 + start * n, 1 + (start + 3) * n))
        panel = [r"\begin{tabular}{@{}l" + "c" * (3 * n) + r"@{}}"]
        for line in lines[1:-1]:
            if line.startswith(r"\cmidrule"):
                panel.append(" ".join(rf"\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}" for i in range(3)))
            elif "&" in line:
                cells = [s.strip() for s in line.removesuffix(r"\\").split("&")]
                indices = [0] + list(range(start + 1, start + 4)) if line == rows[0] else selected
                panel.append(" & ".join(cells[i] for i in indices) + r" \\")
            else:
                panel.append(line)
        panel.append(r"\end{tabular}")
        panels.append("\n".join(panel))
    return "\n\\par\\medskip\n".join(panels)


def readable_table(source: str) -> str:
    """Reflow existing literal cells; retain all values and formatting marks."""
    source = re.sub(r"^.*\\begin\{adjustbox\}.*\n?", "", source, flags=re.M)
    source = re.sub(r"^.*\\end\{adjustbox\}.*\n?", "", source, flags=re.M)
    source = re.sub(r"\\(?:footnotesize|scriptsize|tiny)\b", r"\\small", source)
    source = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", lambda _: r"\setlength{\tabcolsep}{3pt}", source)

    def layout(match):
        block = match.group(0)
        # Two-column floats are necessary for these comparisons at 9 pt.
        single = "tab:search-space" in block
        main_summary = r"\label{tab:result}" in block or r"\label{tab:calibration}" in block
        block = re.sub(
            r"\\begin\{table\*?\}\[[^]]*\]", lambda _: r"\begin{table}[t]" if single else r"\begin{table*}[t]" if main_summary else r"\begin{table*}[tp]", block
        )
        block = re.sub(r"\\end\{table\*?\}", lambda _: r"\end{table}" if single else r"\end{table*}", block)
        if not re.search(r"^\\small$", block, flags=re.M):
            block = block.replace(r"\centering", "\\centering\n\\small", 1)
        if r"\setlength{\tabcolsep}" not in block:
            block = block.replace(r"\centering", "\\centering\n\\setlength{\\tabcolsep}{3pt}", 1)
        block = re.sub(r"\\begin\{tabular\}[^\n]*\n.*?\\end\{tabular\}", lambda m: _panels(m.group(0)), block, flags=re.S)
        if "tab:models" in block:
            block = re.sub(r"\\begin\{tabular\}[^\n]*", lambda _: r"\begin{tabular}{@{}p{0.39\textwidth}p{0.19\textwidth}p{0.38\textwidth}@{}}", block)
        if "tab:efficiency}" in block:
            block = re.sub(
                r"\\begin\{tabular\}[^\n]*", lambda _: r"\begin{tabular}{@{}p{0.23\textwidth}p{0.27\textwidth}p{0.19\textwidth}p{0.25\textwidth}@{}}", block
            )
        if "tab:significance}" in block:
            table = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", block, flags=re.S)
            if table and any(len(line.split("&")) == 6 for line in table.group(0).splitlines()):
                lines = table.group(0).splitlines()
                panels = []
                for setting, indices in ((r"\m vs.\ best baseline", (0, 1, 2, 3)), (r"\mA vs.\ best judge-free", (0, 1, 4, 5))):
                    panel = [
                        r"\begin{tabular}{@{}llcl@{}}",
                        r"\toprule",
                        rf"\multicolumn{{4}}{{c}}{{\textbf{{{setting}}}}} \\",
                        r"\textbf{Backbone} & \textbf{Dataset} & $\Delta$AUROC [95\% CI] & vs. \\",
                        r"\midrule",
                    ]
                    for line in lines[6:-1]:
                        if "&" in line:
                            cells = [s.strip() for s in line.removesuffix(r"\\").split("&")]
                            panel.append(" & ".join(cells[i] for i in indices) + r" \\")
                        else:
                            panel.append(line)
                    panel.append(r"\end{tabular}")
                    panels.append("\n".join(panel))
                block = block[: table.start()] + "\n\\par\\medskip\n".join(panels) + block[table.end() :]
        if "Risk@90" in block and "Risk@90: correctness-classification error" not in block:
            # Captions have nested braces, so insert before the caption's balanced close.
            start = block.index(r"\caption{") + len(r"\caption{")
            depth, end = 1, start
            while depth:
                if block[end] == "{" and block[end - 1] != "\\":
                    depth += 1
                elif block[end] == "}" and block[end - 1] != "\\":
                    depth -= 1
                end += 1
            block = block[: end - 1] + r" Risk@90: correctness-classification error at 90\% coverage." + block[end - 1 :]
        return block

    return re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", layout, source, flags=re.S)
