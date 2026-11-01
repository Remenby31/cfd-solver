# `autan prepare`

`autan prepare` writes the document the solver runtime `autan-run` reads: the case planned against its mesh, every zone resolved and every value in SI ([solver-input.md](../contracts/solver-input.md)). It refuses a case whose plan has errors, with the plan's diagnostics and exit status 1, and reports the plan's warnings on stderr.

```text
autan prepare CASE --mesh-info MESHINFO [--device-kind cpu|cuda|hip] [--device-count N] [-o FILE]
```

Without `-o`, it prints the document to stdout. [`autan-run`](run.md#solver-backend) rejects with `E0706` a combination of physics, precision and device it does not implement.
