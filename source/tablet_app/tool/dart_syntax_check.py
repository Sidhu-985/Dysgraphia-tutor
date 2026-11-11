"""Syntax-only check of every .dart file using the tree-sitter Dart grammar.

This is NOT `flutter analyze`. It finds parse errors (unbalanced brackets,
malformed declarations) but does no type checking, import resolution or
lint checks. It exists because the Flutter SDK could not be installed in
the environment where this code was written.

Usage: python tool/dart_syntax_check.py [root]   (needs: pip install tree-sitter tree-sitter-language-pack)
"""
import pathlib
import sys

from tree_sitter_language_pack import get_parser

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
parser = get_parser("dart")
bad = 0
files = sorted(root.rglob("*.dart"))
for f in files:
    tree = parser.parse(f.read_bytes())
    errors = []

    def walk(n):
        if n.type == "ERROR" or n.is_missing:
            errors.append((n.start_point[0] + 1, n.start_point[1] + 1, n.type, n.text[:60]))
        for c in n.children:
            walk(c)
    walk(tree.root_node)
    status = "OK" if not errors else f"{len(errors)} syntax error(s)"
    print(f"{status:<22} {f}")
    for e in errors[:5]:
        print("    line %d col %d %s %r" % e)
    bad += bool(errors)
print(f"\n{len(files)} files parsed, {bad} with syntax errors")
sys.exit(1 if bad else 0)
