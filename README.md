# spark-eagle3-forge

Fine-tuning the EAGLE-3 draft head for Spark-X2.6-4B, to raise the draft acceptance rate in
Chinese + thinking registers so speculative decoding speeds up single-sequence decode.

- [`AGENTS.md`](AGENTS.md) — purpose, hard constraints, design, plan, verified commands
- [`docs/facts.md`](docs/facts.md) — core facts: model geometry, cost model, vocabulary ceiling
- [`docs/notes.md`](docs/notes.md) — measurements, derivations, open questions

Everything up to stage 1 is reproducible with the released v8 draft head alone — no training
needed to re-verify the feature pipeline.

> This repository is public. Every tracked file, including notes and code comments, must stay free
> of local paths, hardware identifiers and personal details; see `AGENTS.md` §12.
