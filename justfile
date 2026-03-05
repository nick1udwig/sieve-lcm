set shell := ["bash", "-euo", "pipefail", "-c"]

# Build the release CLI binary.
build:
    cargo build --release --bin sieve-lcm-cli

# Install the CLI binary to ~/.local/bin.
install: build
    mkdir -p "$HOME/.local/bin"
    install -m 0755 target/release/sieve-lcm-cli "$HOME/.local/bin/sieve-lcm-cli"
