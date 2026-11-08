"""Fail-fast audit for weak annotations in the local pipeline."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

BANNED = ("Any", "Optional", "None")


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    scope: str
    annotation: str


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return f"{_name(node.value)}[{_name(node.slice)}]"
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.BinOp):
        return f"{_name(node.left)} | {_name(node.right)}"
    if isinstance(node, ast.Tuple):
        return ", ".join(_name(item) for item in node.elts)
    return type(node).__name__


def _is_weak(annotation: str) -> bool:
    return any(token in annotation for token in BANNED)


def _annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.AST, ...]:
    arguments = (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        node.args.vararg,
        node.args.kwarg,
    )
    present = tuple(
        argument.annotation
        for argument in arguments
        if isinstance(argument, ast.arg) and argument.annotation
    )
    if node.returns:
        return present + (node.returns,)
    return present


def audit(root: Path) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for annotation in _annotations(node):
                    text = _name(annotation)
                    if _is_weak(text):
                        findings.append(
                            Finding(
                                path=str(path),
                                line=annotation.lineno,
                                scope=node.name,
                                annotation=text,
                            )
                        )
            if isinstance(node, ast.AnnAssign):
                text = _name(node.annotation)
                if _is_weak(text):
                    scope = node.target.id if isinstance(node.target, ast.Name) else "<assignment>"
                    findings.append(
                        Finding(
                            path=str(path),
                            line=node.lineno,
                            scope=scope,
                            annotation=text,
                        )
                    )
    return tuple(findings)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit annotations for Any/Optional/None.")
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()
    roots = tuple(Path(item) for item in args.root) or (Path("preprocess"),)
    findings = tuple(finding for root in roots for finding in audit(root))
    for finding in findings:
        print(f"{finding.path}:{finding.line}  {finding.scope}  {finding.annotation}")
    print(f"weak annotations: {len(findings)}")
    return 1 if args.fail and findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
