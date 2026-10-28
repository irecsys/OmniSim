"""
Every chat / embedding call of the dialogue run goes through utils.llm_calls,
which counts calls and errors per call site; no except clause around such a call swallows the error without
llm_calls.fallback (or a raise). Static (ast) checks, in the spirit of the test that tools use
is_tmp_dialogue: they keep later code from bypassing the wrapper.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# the modules of the dialogue run (run.py → quick_start → simulator / utils / profile builder / retrieval)
GENERATION = ['run.py', 'utils/quick_start.py', 'utils/simulator.py', 'utils/utils.py',
              'utils/user_profile_builder.py', 'utils/dataset.py'] + \
             sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'utils' / 'retrieval').glob('*.py'))

# direct client calls allowed outside utils/llm_calls.py: (file, enclosing function, reason)
ALLOWED_DIRECT = {
    ('utils/retrieval/local_index.py', 'embed'): '--build-index (_api_embed_fn): not the dialogue run; errors stop the build',
}

# a call site whose name is a parameter: (file, function) -> the parameter
VARIABLE_SITE = {('utils/user_profile_builder.py', '_llm'): 'site'}


def _tree(rel):
    return ast.parse((ROOT / rel).read_text(encoding='utf-8'), filename=rel)


def _functions(tree):
    """node -> name of the innermost enclosing function (module level: '')."""
    owner = {}

    def visit(node, fn):
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            owner[child] = name
            visit(child, name)
    visit(tree, '')
    return owner


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return '.'.join(reversed(parts))


def _is_direct_client_call(call):
    name = _dotted(call.func)
    return name.endswith('.completions.create') or name.endswith('.embeddings.create')


def _is_wrapper_call(call):
    return _dotted(call.func) in ('llm_calls.chat', 'llm_calls.embed')


def test_scanned_modules_exist():
    assert all((ROOT / rel).exists() for rel in GENERATION)
    assert len(GENERATION) >= 10


def test_no_direct_client_calls():
    hits = []
    for rel in GENERATION:
        tree = _tree(rel)
        owner = _functions(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_direct_client_call(node):
                if (rel, owner.get(node, '')) not in ALLOWED_DIRECT:
                    hits.append(f'{rel}:{node.lineno} {_dotted(node.func)}')
    assert hits == [], 'call these through llm_calls.chat / llm_calls.embed:\n' + '\n'.join(hits)


def test_call_sites_are_registered():
    from utils.llm_calls import SITES
    bad, used = [], set()
    for rel in GENERATION:
        tree = _tree(rel)
        owner = _functions(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_wrapper_call(node):
                site = node.args[1] if len(node.args) > 1 else None
                if isinstance(site, ast.Constant) and isinstance(site.value, str):
                    if site.value not in SITES:
                        bad.append(f'{rel}:{node.lineno} unregistered site {site.value!r}')
                    used.add(site.value)
                elif isinstance(site, ast.Name) and VARIABLE_SITE.get((rel, owner.get(node, ''))) == site.id:
                    pass
                else:
                    bad.append(f'{rel}:{node.lineno} the site must be a string constant')
    assert bad == []
    assert used, 'no llm_calls call found'


def test_no_except_swallows_llm_errors():
    """A try whose body calls llm_calls.chat / embed: each handler that catches Exception, BaseException or
    everything must call llm_calls.fallback or raise."""
    bad = []
    for rel in GENERATION:
        tree = _tree(rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            if not any(isinstance(n, ast.Call) and _is_wrapper_call(n) for b in node.body for n in ast.walk(b)):
                continue
            for h in node.handlers:
                broad = h.type is None or _dotted(h.type) in ('Exception', 'BaseException')
                if not broad:
                    continue
                inner = [n for b in h.body for n in ast.walk(b)]
                ok = any(isinstance(n, ast.Raise) for n in inner) or \
                    any(isinstance(n, ast.Call) and _dotted(n.func) in ('llm_calls.fallback', 'llm_calls.startup_failed')
                        for n in inner)
                if not ok:
                    bad.append(f'{rel}:{h.lineno}')
    assert bad == [], 'except clauses that swallow LLM errors without llm_calls.fallback:\n' + '\n'.join(bad)


# ── every dialogue entry point runs inside the per-conversation scope ─────────────────────────────────────────
def _decorated_with_conversation(fn):
    return any(_dotted(d) == 'llm_calls.conversation' for d in fn.decorator_list)


def test_every_simulate_function_has_the_conversation_scope():
    """Each simulate_* in utils/simulator.py is decorated with @llm_calls.conversation, or only returns a call
    of one that is (simulate_static → simulate_adaptive): a single call statement."""
    tree = _tree('utils/simulator.py')
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('simulate_')}
    assert {'simulate_free', 'simulate_static', 'simulate_adaptive'} <= set(funcs)
    decorated = {name for name, fn in funcs.items() if _decorated_with_conversation(fn)}
    bad = []
    for name, fn in funcs.items():
        if name in decorated:
            continue
        body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        ok = (len(body) == 1 and isinstance(body[0], (ast.Return, ast.Expr)) and isinstance(body[0].value, ast.Call)
              and _dotted(body[0].value.func) in decorated)
        if not ok:
            bad.append(name)
    assert bad == [], f'simulate_* functions outside the conversation scope: {bad}'


def test_run_simulation_dispatches_only_through_simulation_funcs():
    """quick_start runs conversations only as SIMULATION_FUNCS[mode] through _run_parallel (all four strategies),
    and never calls a simulate_* function directly."""
    tree = _tree('utils/quick_start.py')
    table = next(n for n in tree.body if isinstance(n, ast.Assign) and _dotted(n.targets[0]) == 'SIMULATION_FUNCS')
    assert sorted(_dotted(v) for v in table.value.values) == ['simulate_adaptive', 'simulate_free', 'simulate_static']
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    direct = [n.lineno for n in calls if _dotted(n.func).startswith('simulate_')]
    assert direct == [], f'direct simulate_* calls in quick_start.py: {direct}'
    runs = [n for n in calls if _dotted(n.func) == '_run_parallel']
    assert len(runs) == 4 and all(_dotted(n.args[0]) == 'sim_func' for n in runs)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(_dotted(t) == 'sim_func' for t in n.targets)]
    assert assigns and all(isinstance(n.value, ast.Subscript) and _dotted(n.value.value) == 'SIMULATION_FUNCS'
                           for n in assigns)


def test_simulation_funcs_are_scoped_at_runtime():
    import utils.quick_start as qs
    for mode, fn in qs.SIMULATION_FUNCS.items():
        if mode == 'static':
            continue                                  # a wrapper of simulate_adaptive (checked above)
        assert getattr(fn, 'omnisim_conversation_scope', False), mode
