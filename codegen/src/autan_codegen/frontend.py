"""Parse a pointwise body (restricted Python) into the scalar SSA IR.

Vectors and matrices are scalarised here; `for i in range(N)` is unrolled; `if` statements become
selects, so every name is assigned once per path and a predicate always sees the value it was
computed from. Calls to other `@pointwise` bodies are inlined.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import textwrap
import typing
from dataclasses import dataclass

from autan_codegen import dsl
from autan_codegen.errors import Code, CodegenError
from autan_codegen.ir import Function, Instr, optimise

Number = int | float


@dataclass(frozen=True)
class Bool:
    v: int


@dataclass(frozen=True)
class Arr:
    """A scalarised vector (shape (n,)) or matrix (shape (n, m)), row-major."""

    ids: tuple[int, ...]
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Real:
    """An IR real scalar; plain Python ints and floats are compile-time constants."""

    v: int


type Val = Real | Bool | Arr | int | float | tuple[Val, ...]

UNARY_MATH = {"sqrt", "exp", "log", "sin", "cos"}
INTRINSIC_ARITY = {
    **dict.fromkeys(UNARY_MATH, 1),
    "abs": 1,
    "norm": 1,
    "transpose": 1,
    "inverse": 1,
    "identity": 1,
    "dot": 2,
    "cross": 2,
    "outer": 2,
    "rel": 2,
    "min": 2,
    "max": 2,
    "Vec3": 3,
    "Vec5": 5,
}
INTRINSICS = UNARY_MATH | {
    "dot",
    "cross",
    "norm",
    "rel",
    "Vec3",
    "Vec5",
    "transpose",
    "inverse",
    "identity",
    "outer",
    "abs",
    "min",
    "max",
    "range",
}


@dataclass
class Signature:
    names: list[str]
    types: list[dsl.Type]
    returns: list[dsl.Type] | None


def signature(body: dsl.Body) -> Signature:
    fn = body.fn
    sig = inspect.signature(fn)
    names, types = [], []
    for p in sig.parameters.values():
        if p.annotation is inspect.Parameter.empty:
            raise CodegenError(
                Code.NOT_A_DSL_TYPE, f"argument {p.name} of {body.name} has no DSL type"
            )
        names.append(p.name)
        types.append(dsl.as_type(_annotation(p.annotation, fn)))
    ret = sig.return_annotation
    returns: list[dsl.Type] | None = None
    if ret is not inspect.Signature.empty:
        ret = _annotation(ret, fn)
        returns = [dsl.as_type(r) for r in ret] if isinstance(ret, tuple) else [dsl.as_type(ret)]
    return Signature(names, types, returns)


def _annotation(a: object, fn: object) -> object:
    if isinstance(a, str):
        a = eval(a, fn.__globals__)  # type: ignore[attr-defined]
    origin = typing.get_origin(a)
    if origin is tuple:
        return tuple(typing.get_args(a))
    return a


class _Builder:
    def __init__(self, fn: Function, body: dsl.Body, line0: int) -> None:
        self.fn = fn
        self.body = body
        self.globals = body.fn.__globals__
        self.line0 = line0
        self.line = 0
        self.geom_inputs: set[int] = set()

    # -- error helper
    def err(self, code: Code, msg: str) -> CodegenError:
        return CodegenError(code, msg, f"{self.body.source_file}:{self.line0 + self.line}")

    # -- IR emission
    def emit(self, op: str, *args: int, attr: object = None) -> int:
        return self.fn.add(Instr(op, args, attr, self.line0 + self.line))

    def real(self, x: Val) -> int:
        if isinstance(x, Real):
            return x.v
        if isinstance(x, bool):
            raise self.err(Code.TYPE_MISMATCH, "booleans are not numbers")
        if isinstance(x, int | float):
            return self.emit("const", attr=float(x))
        raise self.err(Code.TYPE_MISMATCH, f"expected a scalar, got {_kind(x)}")

    def boolean(self, x: Val) -> int:
        if isinstance(x, Bool):
            return x.v
        raise self.err(Code.TYPE_MISMATCH, f"expected a condition, got {_kind(x)}")

    # -- arithmetic on values
    def binop(self, op: str, a: Val, b: Val) -> Val:
        if isinstance(a, int | float) and isinstance(b, int | float):
            return _fold(op, a, b, self)
        if isinstance(a, Arr) or isinstance(b, Arr):
            return self._arr_binop(op, a, b)
        return Real(self.emit(op, self.real(a), self.real(b)))

    def _arr_binop(self, op: str, a: Val, b: Val) -> Arr:
        if isinstance(a, Arr) and isinstance(b, Arr):
            if op not in ("add", "sub") or a.shape != b.shape:
                raise self.err(
                    Code.TYPE_MISMATCH, f"cannot {op} {a.shape} and {b.shape}; use dot() or @"
                )
            return Arr(
                tuple(self.emit(op, x, y) for x, y in zip(a.ids, b.ids, strict=True)), a.shape
            )
        if isinstance(a, Arr):
            if op not in ("mul", "div"):
                raise self.err(Code.TYPE_MISMATCH, f"cannot {op} an array and a scalar")
            s = self.real(b)
            return Arr(tuple(self.emit(op, x, s) for x in a.ids), a.shape)
        assert isinstance(b, Arr)
        if op != "mul":
            raise self.err(Code.TYPE_MISMATCH, f"cannot {op} a scalar and an array")
        s = self.real(a)
        return Arr(tuple(self.emit("mul", s, x) for x in b.ids), b.shape)

    def neg(self, a: Val) -> Val:
        if isinstance(a, int | float):
            return -a
        if isinstance(a, Arr):
            return Arr(tuple(self.emit("neg", x) for x in a.ids), a.shape)
        return Real(self.emit("neg", self.real(a)))

    def select(self, c: int, a: Val, b: Val) -> Val:
        if isinstance(a, tuple) and isinstance(b, tuple):
            if len(a) != len(b):
                raise self.err(
                    Code.INVALID_CONTROL_FLOW, "branches return different numbers of values"
                )
            return tuple(self.select(c, x, y) for x, y in zip(a, b, strict=True))
        if isinstance(a, Arr) or isinstance(b, Arr):
            if not (isinstance(a, Arr) and isinstance(b, Arr) and a.shape == b.shape):
                raise self.err(
                    Code.INVALID_CONTROL_FLOW, "branches give values of different shapes"
                )
            return Arr(
                tuple(self.emit("select", c, x, y) for x, y in zip(a.ids, b.ids, strict=True)),
                a.shape,
            )
        if isinstance(a, Bool) or isinstance(b, Bool):
            raise self.err(
                Code.INVALID_CONTROL_FLOW, "a condition cannot depend on another condition's branch"
            )
        if (
            isinstance(a, int)
            and isinstance(b, int)
            and not isinstance(a, bool)
            and a == b
            and type(a) is type(b)
        ):
            return a
        return Real(self.emit("select", c, self.real(a), self.real(b)))

    # -- expressions
    def expr(self, node: ast.expr, env: dict[str, Val]) -> Val:
        self.line = getattr(node, "lineno", self.line)
        match node:
            case ast.Constant(value=v) if isinstance(v, int | float) and not isinstance(v, bool):
                return v
            case ast.Name(id=name):
                return self.name(name, env)
            case ast.Tuple(elts=elts):
                return tuple(self.expr(e, env) for e in elts)
            case ast.UnaryOp(op=ast.USub(), operand=x):
                return self.neg(self.expr(x, env))
            case ast.UnaryOp(op=ast.UAdd(), operand=x):
                return self.expr(x, env)
            case ast.UnaryOp(op=ast.Not(), operand=x):
                return Bool(self.emit("not", self.boolean(self.expr(x, env))))
            case ast.BinOp(left=lhs, op=op, right=rhs):
                return self.binary(op, self.expr(lhs, env), self.expr(rhs, env))
            case ast.Compare(left=lhs, ops=[op], comparators=[rhs]):
                ops = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}
                if type(op) not in ops:
                    raise self.err(Code.UNSUPPORTED_CONSTRUCT, "only <, <=, >, >= compare reals")
                a, b = self.expr(lhs, env), self.expr(rhs, env)
                return Bool(self.emit(ops[type(op)], self.real(a), self.real(b)))
            case ast.BoolOp(op=op, values=values):
                name = "and" if isinstance(op, ast.And) else "or"
                acc = self.boolean(self.expr(values[0], env))
                for v in values[1:]:
                    acc = self.emit(name, acc, self.boolean(self.expr(v, env)))
                return Bool(acc)
            case ast.IfExp(test=t, body=a, orelse=b):
                c = self.boolean(self.expr(t, env))
                return self.select(c, self.expr(a, env), self.expr(b, env))
            case ast.Subscript(value=v, slice=s):
                return self.subscript(self.expr(v, env), s, env)
            case ast.Attribute(value=v, attr=a) if a in ("x", "y", "z"):
                return self.index(self.expr(v, env), "xyz".index(a))
            case ast.Call(func=ast.Name(id=fname), args=args, keywords=[]):
                return self.call(fname, [self.expr(a, env) for a in args])
        raise self.err(Code.UNSUPPORTED_CONSTRUCT, f"unsupported expression: {ast.unparse(node)}")

    def binary(self, op: ast.operator, a: Val, b: Val) -> Val:
        ops = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
        if type(op) in ops:
            return self.binop(ops[type(op)], a, b)
        if isinstance(op, ast.MatMult):
            return self.matmul(a, b)
        if isinstance(op, ast.Pow):
            if not isinstance(b, int) or isinstance(b, bool) or not 0 <= b <= 8:
                raise self.err(
                    Code.UNSUPPORTED_CONSTRUCT,
                    "exponents are integer constants in [0, 8]; use sqrt/exp/log",
                )
            if isinstance(a, int | float):
                return a**b
            result: Val = 1.0
            for _ in range(b):
                result = self.binop("mul", result, a) if result != 1.0 else a
            return result
        raise self.err(Code.UNSUPPORTED_CONSTRUCT, f"unsupported operator {type(op).__name__}")

    def matmul(self, a: Val, b: Val) -> Val:
        if not (isinstance(a, Arr) and isinstance(b, Arr) and len(a.shape) == 2):
            raise self.err(Code.TYPE_MISMATCH, "@ multiplies a matrix by a vector or matrix")
        n, m = a.shape
        if b.shape[0] != m:
            raise self.err(Code.TYPE_MISMATCH, f"@ shapes {a.shape} and {b.shape} do not match")
        cols = b.shape[1] if len(b.shape) == 2 else 1
        ids = []
        for i in range(n):
            for j in range(cols):
                acc: Val = 0.0
                for k in range(m):
                    bk = b.ids[k * cols + j]
                    acc = self.binop("add", acc, Real(self.emit("mul", a.ids[i * m + k], bk)))
                ids.append(self.real(acc))
        shape = (n, cols) if len(b.shape) == 2 else (n,)
        return Arr(tuple(ids), shape)

    def name(self, name: str, env: dict[str, Val]) -> Val:
        if name in env:
            return env[name]
        g = self.globals.get(name, getattr(builtins, name, None))
        if isinstance(g, int | float) and not isinstance(g, bool):
            return g
        raise self.err(
            Code.UNSUPPORTED_CONSTRUCT,
            f"name {name!r} is not an argument, a local or a numeric constant",
        )

    def subscript(self, v: Val, s: ast.expr, env: dict[str, Val]) -> Val:
        if isinstance(s, ast.Tuple):
            i, j = (self.expr(e, env) for e in s.elts)
            if not (
                isinstance(v, Arr)
                and len(v.shape) == 2
                and isinstance(i, int)
                and isinstance(j, int)
            ):
                raise self.err(Code.TYPE_MISMATCH, "m[i, j] needs a matrix and constant indices")
            return Real(v.ids[i * v.shape[1] + j])
        i = self.expr(s, env)
        if not isinstance(i, int) or isinstance(i, bool):
            raise self.err(Code.TYPE_MISMATCH, "indices are compile-time integer constants")
        return self.index(v, i)

    def index(self, v: Val, i: int) -> Val:
        if isinstance(v, tuple):
            return v[i]
        if not isinstance(v, Arr):
            raise self.err(Code.TYPE_MISMATCH, f"cannot index {_kind(v)}")
        if not 0 <= i < v.shape[0]:
            raise self.err(Code.TYPE_MISMATCH, f"index {i} out of range for shape {v.shape}")
        if len(v.shape) == 1:
            return Real(v.ids[i])
        m = v.shape[1]
        return Arr(v.ids[i * m : (i + 1) * m], (m,))

    def call(self, fname: str, args: list[Val]) -> Val:
        g = self.globals.get(fname, getattr(builtins, fname, None))
        if isinstance(g, dsl.Body):
            if g.compute != self.fn.compute:
                raise self.err(
                    Code.TYPE_MISMATCH,
                    f"{fname} computes in {g.compute!r}, its caller in {self.fn.compute!r}",
                )
            return inline(g, args, self)
        if fname not in INTRINSICS or not (
            g is dsl._intrinsic or g is getattr(builtins, fname, None)
        ):
            raise self.err(
                Code.UNSUPPORTED_CONSTRUCT,
                f"call to {fname!r}: not a DSL intrinsic or @pointwise body",
            )
        arity = INTRINSIC_ARITY.get(fname)
        if arity is not None and len(args) != arity:
            raise self.err(
                Code.TYPE_MISMATCH, f"{fname} takes {arity} argument(s), got {len(args)}"
            )
        if fname in UNARY_MATH:
            (x,) = args
            if isinstance(x, Arr):
                raise self.err(Code.TYPE_MISMATCH, f"{fname} takes a scalar")
            return Real(self.emit(fname, self.real(x)))
        if fname == "abs":
            (x,) = args
            return self._abs(self.real(x))
        if fname in ("min", "max"):
            if len(args) != 2:
                raise self.err(Code.TYPE_MISMATCH, f"{fname} takes two arguments")
            a, b = args
            if isinstance(a, Arr) or isinstance(b, Arr):
                if not (isinstance(a, Arr) and isinstance(b, Arr) and a.shape == b.shape):
                    raise self.err(Code.TYPE_MISMATCH, f"{fname} of arrays needs equal shapes")
                return Arr(
                    tuple(self._minmax(fname, x, y) for x, y in zip(a.ids, b.ids, strict=True)),
                    a.shape,
                )
            return Real(self._minmax(fname, self.real(a), self.real(b)))
        if fname == "dot":
            a, b = args
            if not (
                isinstance(a, Arr)
                and isinstance(b, Arr)
                and a.shape == b.shape
                and len(a.shape) == 1
            ):
                raise self.err(Code.TYPE_MISMATCH, "dot takes two vectors of the same size")
            acc: Val = 0.0
            for x, y in zip(a.ids, b.ids, strict=True):
                acc = self.binop("add", acc, Real(self.emit("mul", x, y)))
            return acc
        if fname == "norm":
            (a,) = args
            return Real(self.emit("sqrt", self.real(self.call("dot", [a, a]))))
        if fname == "cross":
            a, b = args
            if not (isinstance(a, Arr) and isinstance(b, Arr) and a.shape == b.shape == (3,)):
                raise self.err(Code.TYPE_MISMATCH, "cross takes two 3-vectors")
            xa, xb = a.ids, b.ids
            ids = tuple(
                self.emit("sub", self.emit("mul", xa[i], xb[j]), self.emit("mul", xa[j], xb[i]))
                for i, j in ((1, 2), (2, 0), (0, 1))
            )
            return Arr(ids, (3,))
        if fname in ("Vec3", "Vec5"):
            n = int(fname[3])
            if len(args) != n:
                raise self.err(Code.TYPE_MISMATCH, f"{fname} takes {n} scalars")
            return Arr(tuple(self.real(a) for a in args), (n,))
        if fname == "outer":
            a, b = args
            if not (
                isinstance(a, Arr) and isinstance(b, Arr) and len(a.shape) == len(b.shape) == 1
            ):
                raise self.err(Code.TYPE_MISMATCH, "outer takes two vectors")
            ids = tuple(self.emit("mul", x, y) for x in a.ids for y in b.ids)
            return Arr(ids, (a.shape[0], b.shape[0]))
        if fname == "transpose":
            (a,) = args
            if not (isinstance(a, Arr) and len(a.shape) == 2):
                raise self.err(Code.TYPE_MISMATCH, "transpose takes a matrix")
            n, m = a.shape
            return Arr(tuple(a.ids[i * m + j] for j in range(m) for i in range(n)), (m, n))
        if fname == "identity":
            (n,) = args
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                raise self.err(Code.TYPE_MISMATCH, "identity takes a constant size")
            one, zero = self.real(1.0), self.real(0.0)
            return Arr(tuple(one if i == j else zero for i in range(n) for j in range(n)), (n, n))
        if fname == "inverse":
            (a,) = args
            if not (isinstance(a, Arr) and len(a.shape) == 2 and a.shape[0] == a.shape[1]):
                raise self.err(Code.TYPE_MISMATCH, "inverse takes a square matrix")
            return self.inverse(a)
        if fname == "rel":
            a, b = args
            return self.rel(a, b)
        raise self.err(Code.UNSUPPORTED_CONSTRUCT, f"{fname} is not usable here")

    def _abs(self, x: int) -> Real:
        # abs(x) = x if x >= 0 else -x: slope +1 at 0 (selected-branch convention).
        zero = self.emit("const", attr=0.0)
        return Real(self.emit("select", self.emit("ge", x, zero), x, self.emit("neg", x)))

    def inverse(self, a: Arr) -> Arr:
        # Gauss-Jordan on [A | I] with partial pivoting, in the operation order of invert5
        # (physics/src/block_matrix.cpp); row swaps become selects. A zero or non-finite pivot
        # makes the scale factor NaN, so every entry of the result is NaN.
        n = a.shape[0]
        m: list[list[Val]] = [
            [Real(a.ids[i * n + j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)]
            for i in range(n)
        ]
        zero = self.emit("const", attr=0.0)
        nan = self.emit("div", zero, zero)
        for k in range(n):
            mag = [self._abs(self.real(m[i][k])) for i in range(n)]
            best, picks = mag[k], []
            for i in range(k + 1, n):
                c = self.emit("gt", self.real(mag[i]), self.real(best))
                best = self.select(c, mag[i], best)
                picks.append((i, c))
            # Row i is the pivot row if it beat every earlier candidate and no later one beat it.
            is_pivot: dict[int, int] = {}
            later: int | None = None
            for i, c in reversed(picks):
                is_pivot[i] = c if later is None else self.emit("and", c, self.emit("not", later))
                later = c if later is None else self.emit("or", c, later)
            row_k = m[k]
            for i, _ in picks:
                m[k] = [self.select(is_pivot[i], x, y) for x, y in zip(m[i], m[k], strict=True)]
                m[i] = [self.select(is_pivot[i], x, y) for x, y in zip(row_k, m[i], strict=True)]
            pivot = self.real(m[k][k])
            size = self.real(best)
            finite = self.emit("ge", self.emit("sub", size, size), zero)
            regular = self.emit("and", self.emit("gt", size, zero), finite)
            d = Real(self.emit("select", regular, self.emit("div", self.real(1.0), pivot), nan))
            m[k] = [self.binop("mul", x, d) for x in m[k]]
            for i in range(n):
                if i != k:
                    f = m[i][k]
                    m[i] = [
                        self.binop("sub", x, self.binop("mul", f, y))
                        for x, y in zip(m[i], m[k], strict=True)
                    ]
        return Arr(tuple(self.real(m[i][n + j]) for i in range(n) for j in range(n)), (n, n))

    def _minmax(self, fname: str, a: int, b: int) -> int:
        # min(a, b) = a if a <= b else b; ties select the first argument.
        cmp = self.emit("le" if fname == "min" else "ge", a, b)
        return self.emit("select", cmp, a, b)

    def rel(self, a: Val, b: Val) -> Val:
        def comps(v: Val) -> tuple[int, ...]:
            ids = v.ids if isinstance(v, Arr) else (v.v,) if isinstance(v, Real) else None
            if ids is None or any(i not in self.geom_inputs for i in ids):
                raise self.err(
                    Code.REL_OF_COMPUTED_VALUE,
                    "rel() takes Geom arguments as loaded, before any arithmetic",
                )
            return ids

        ia, ib = comps(a), comps(b)
        if len(ia) != len(ib):
            raise self.err(Code.TYPE_MISMATCH, "rel() needs two values of the same shape")
        ids = tuple(self.emit("rel", x, y) for x, y in zip(ia, ib, strict=True))
        return Arr(ids, a.shape) if isinstance(a, Arr) else Real(ids[0])

    # -- statements
    def block(self, stmts: list[ast.stmt], env: dict[str, Val]) -> Val | None:
        for k, st in enumerate(stmts):
            self.line = st.lineno
            ret = self.stmt(st, env)
            if ret is not None:
                if k != len(stmts) - 1:
                    raise self.err(Code.INVALID_CONTROL_FLOW, "statements after return")
                return ret
        return None

    def stmt(self, st: ast.stmt, env: dict[str, Val]) -> Val | None:
        match st:
            case ast.Expr(value=ast.Constant(value=str())) | ast.Pass():
                return None
            case ast.Return(value=v) if v is not None:
                return self.expr(v, env)
            case ast.Assign(targets=[target], value=v):
                self.assign(target, self.expr(v, env), env)
                return None
            case ast.AnnAssign(target=target, value=v) if v is not None:
                self.assign(target, self.expr(v, env), env)
                return None
            case ast.AugAssign(target=ast.Name(id=name), op=op, value=v):
                env[name] = self.binary(op, self.name(name, env), self.expr(v, env))
                return None
            case ast.If(test=t, body=then, orelse=other):
                c = self.boolean(self.expr(t, env))
                env_t, env_f = dict(env), dict(env)
                rt = self.block(then, env_t)
                rf = self.block(other, env_f) if other else None
                if (rt is None) != (rf is None):
                    raise self.err(
                        Code.INVALID_CONTROL_FLOW,
                        "return inside if needs a return in both branches",
                    )
                if rt is not None:
                    return self.select(c, rt, rf)  # type: ignore[arg-type]
                for name in set(env_t) | set(env_f):
                    if name not in env_t or name not in env_f:
                        raise self.err(
                            Code.INVALID_CONTROL_FLOW, f"{name!r} is assigned in only one branch"
                        )
                    a, b = env_t[name], env_f[name]
                    env[name] = a if a is b else self.select(c, a, b)
                return None
            case ast.For(
                target=ast.Name(id=var),
                iter=ast.Call(func=ast.Name(id="range"), args=rargs),
                body=body,
                orelse=[],
            ):
                bounds = [self.expr(a, env) for a in rargs]
                if not all(isinstance(b, int) and not isinstance(b, bool) for b in bounds):
                    raise self.err(
                        Code.UNSUPPORTED_CONSTRUCT, "range() bounds are compile-time integers"
                    )
                for i in range(*bounds):  # type: ignore[arg-type]
                    env[var] = i
                    if self.block(body, env) is not None:
                        raise self.err(Code.INVALID_CONTROL_FLOW, "return inside a loop")
                return None
        raise self.err(
            Code.UNSUPPORTED_CONSTRUCT, f"unsupported statement: {ast.unparse(st).splitlines()[0]}"
        )

    def assign(self, target: ast.expr, v: Val, env: dict[str, Val]) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = v
        elif isinstance(target, ast.Tuple) and isinstance(v, tuple) and len(v) == len(target.elts):
            for t, x in zip(target.elts, v, strict=True):
                self.assign(t, x, env)
        else:
            raise self.err(
                Code.UNSUPPORTED_CONSTRUCT, "assign to a name or unpack a tuple of matching length"
            )


def _fold(op: str, a: Number, b: Number, b_: _Builder) -> Number:
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if b == 0:
        raise b_.err(Code.TYPE_MISMATCH, "division by zero in a constant expression")
    return a / b


def _kind(v: object) -> str:
    return {Real: "a scalar", Bool: "a condition", Arr: "an array", tuple: "a tuple"}.get(
        type(v), type(v).__name__
    )


def _parse_def(body: dsl.Body) -> tuple[ast.FunctionDef, int]:
    lines, line0 = inspect.getsourcelines(body.fn)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    fdef = tree.body[0]
    assert isinstance(fdef, ast.FunctionDef)
    return fdef, line0 - 1


def _shape_value(ids: list[int], t: dsl.Type) -> Val:
    if t.shape == ():
        return Real(ids[0])
    return Arr(tuple(ids), t.shape)


def inline(body: dsl.Body, args: list[Val], caller: _Builder) -> Val:
    sig = signature(body)
    if len(args) != len(sig.names):
        raise caller.err(Code.TYPE_MISMATCH, f"{body.name} takes {len(sig.names)} arguments")
    fdef, line0 = _parse_def(body)
    env: dict[str, Val] = dict(zip(sig.names, args, strict=True))
    saved = (caller.globals, caller.line0, caller.body, caller.line)
    caller.globals, caller.line0, caller.body = body.fn.__globals__, line0, body
    try:
        ret = caller.block(fdef.body, env)
    finally:
        caller.globals, caller.line0, caller.body, caller.line = saved
    if ret is None:
        raise caller.err(Code.INVALID_CONTROL_FLOW, f"{body.name} returns nothing")
    return ret


def flatten(b: _Builder, v: Val) -> list[int]:
    if isinstance(v, tuple):
        return [i for x in v for i in flatten(b, x)]
    if isinstance(v, Arr):
        return list(v.ids)
    return [b.real(v)]


def parse(body: dsl.Body) -> tuple[Function, Signature]:
    """Parse `body` into an optimised IR function; inputs follow the argument order."""
    sig = signature(body)
    fdef, line0 = _parse_def(body)
    fn = Function(
        body.name,
        sig.names,
        [t.size for t in sig.types],
        source=body.source_file,
        compute=body.compute,
    )
    b = _Builder(fn, body, line0)
    env: dict[str, Val] = {}
    for k, (name, t) in enumerate(zip(sig.names, sig.types, strict=True)):
        b.line = fdef.lineno
        ids = [b.emit("input", attr=(k, c)) for c in range(t.size)]
        if t.role is dsl.Role.GEOM:
            b.geom_inputs.update(ids)
        env[name] = _shape_value(ids, t)
    ret = b.block(fdef.body, env)
    if ret is None:
        raise b.err(Code.INVALID_CONTROL_FLOW, f"{body.name} returns nothing")
    values = list(ret) if isinstance(ret, tuple) else [ret]
    if sig.returns is not None:
        if len(sig.returns) != len(values):
            raise b.err(
                Code.TYPE_MISMATCH,
                f"{body.name} returns {len(values)} values, annotation says {len(sig.returns)}",
            )
        for t, v in zip(sig.returns, values, strict=True):
            shape = v.shape if isinstance(v, Arr) else ()
            if shape != t.shape:
                raise b.err(
                    Code.TYPE_MISMATCH,
                    f"{body.name} returns shape {shape}, annotation says {t.shape}",
                )
    fn.outputs = flatten(b, ret)
    return optimise(fn), sig
