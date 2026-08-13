"""AST-based dependency analysis for rcb_runner split."""
import ast

SRC = "/workspace/agent/huginn/cli/rcb_runner.py"
with open(SRC, encoding="utf-8") as f:
    src = f.read()
tree = ast.parse(src)

module_defs = set()
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        module_defs.add(node.name)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name):
                module_defs.add(t.id)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        module_defs.add(node.target.id)
    elif isinstance(node, ast.Import):
        for a in node.names:
            module_defs.add((a.asname or a.name).split(".")[0])
    elif isinstance(node, ast.ImportFrom):
        for a in node.names:
            module_defs.add(a.asname or a.name)

targets = [
    "_step2_execute", "_step3_adversarial", "_run_mcmc_mode", "run", "main",
]

# List all top-level defs/classes with line ranges
print("=== ALL TOP-LEVEL DEFS/CLASSES ===")
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        print(f"{node.lineno:5}-{node.end_lineno:5}  {node.name}")
print()

def body_global_names(node):
    local_names = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n is node:
                continue
            local_names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    local_names.add(t.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            local_names.add(n.target.id)
        elif isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name):
            local_names.add(n.target.id)
        elif isinstance(n, ast.For):
            if isinstance(n.target, ast.Name):
                local_names.add(n.target.id)
        elif isinstance(n, ast.With):
            for item in n.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    local_names.add(item.optional_vars.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            local_names.add(n.name)
        elif isinstance(n, ast.arguments):
            for a in n.args + n.kwonlyargs + n.posonlyargs:
                local_names.add(a.arg)
            if n.vararg: local_names.add(n.vararg.arg)
            if n.kwarg: local_names.add(n.kwarg.arg)
        elif isinstance(n, ast.Lambda):
            for a in n.args.args + n.args.kwonlyargs + n.args.posonlyargs:
                local_names.add(a.arg)
            if n.args.vararg: local_names.add(n.args.vararg.arg)
            if n.args.kwarg: local_names.add(n.args.kwarg.arg)
        elif isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for g in n.generators:
                if isinstance(g.target, ast.Name):
                    local_names.add(g.target.id)

    refs = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id not in local_names and n.id in module_defs:
                refs.add(n.id)
    return refs

for target in targets:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == target:
            refs = sorted(body_global_names(node))
            print(f"=== {target} (L{node.lineno}-{node.end_lineno}) ===")
            print("module-level names referenced:", refs)
            print()