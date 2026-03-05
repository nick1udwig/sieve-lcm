# sieve-lcm

A [lossless-claw](https://github.com/Martian-Engineering/lossless-claw) clone -> the memory backend for [sieve](https://github.com/nick1udwig/sieve).

Based on [LCM](https://papers.voltropy.com/LCM).

## CLI

This repo now ships `sieve-lcm-cli` for tool-driven memory access.

- `ingest`: append a message to a lane database.
- `query`: retrieve trusted excerpts plus untrusted opaque refs.
- `expand`: resolve an untrusted opaque ref (for quarantined/qLLM flows).
