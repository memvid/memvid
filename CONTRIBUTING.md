# Contributing to Memvid

Thank you for your interest in contributing to Memvid! We welcome contributions from everyone.

## Getting Started

### Prerequisites

- **Rust 1.88.0+** — Install from [rustup.rs](https://rustup.rs)
- **Git** — For version control

### Setup

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/memvid.git
   cd memvid
   ```

3. **Build the project**:
   ```bash
   cargo build
   ```

4. **Run tests**:
   ```bash
   cargo test
   ```

## Development Workflow

### Creating a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### Making Changes

1. Write your code following the [code style guidelines](#code-style)
2. Add tests for new functionality
3. Ensure all tests pass: `cargo test`
4. Run clippy: `cargo clippy`
5. Format code: `cargo fmt`

### Committing

Write clear, concise commit messages:

```bash
git commit -m "feat: add support for XYZ"
git commit -m "fix: resolve issue with ABC"
git commit -m "docs: update README examples"
```

### Submitting a Pull Request

1. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request on GitHub

3. Fill out the PR template completely

4. Wait for review and address feedback

## Code Style

### Rust Guidelines

- Follow standard Rust idioms and conventions
- Use `rustfmt` for formatting (`cargo fmt`)
- Use `clippy` for linting (`cargo clippy`). We maintain a **zero-warning policy**.
- Prefer explicit types for public APIs
- Use `thiserror` for error definitions

### Linting & Safety

We enforce strict linting to ensure safety and portability:

1.  **Zero Warnings**: CI will fail on any warning. Run `cargo clippy --workspace --all-targets -- -D warnings` locally.
2.  **No Panics**: `unwrap()` and `expect()` are **denied** in library code. Use `Result` propagation (`?`) or graceful error handling. They are allowed in `tests/`.
3.  **No Truncation**: `cast_possible_truncation` is denied. Use `try_from` when converting `u64` to `usize`/`u32`.
4.  **Exceptions**: We allow pragmatic lints (e.g., `cast_precision_loss` for ML math) in `src/lib.rs`. Do not add global `#![allow]` without discussion.

### Documentation

- Add doc comments (`///`) to all public functions, structs, and modules
- Include examples in doc comments where helpful
- Keep comments concise and up-to-date

### Testing

- Write unit tests for new functionality
- Place tests in the same file using `#[cfg(test)]` module
- Integration tests go in the `tests/` directory
- Aim for high coverage of edge cases

## Project Structure

```
memvid/
├── src/              # Source code
│   ├── lib.rs        # Public API
│   ├── memvid/       # Core implementation
│   ├── io/           # File I/O
│   └── types/        # Type definitions
├── tests/            # Integration tests
├── examples/         # Example code
├── benchmarks/       # Benchmarks
└── data/             # Required data files
```

## Feature Flags

When adding new functionality, consider if it should be behind a feature flag:

```toml
[features]
my_feature = ["dep:some-dependency"]
```

This keeps the default build lean and fast.

## Reporting Issues

When reporting bugs, please include:

- Rust version (`rustc --version`)
- Operating system
- Memvid version
- Minimal code to reproduce
- Expected vs actual behavior

## Translations

Interested in translating Memvid's documentation? See [Contributing Translations](docs/i18n/CONTRIBUTING_TRANSLATIONS.md) for guidelines on translating the README and other documentation.

## Getting Help

- Open a [Discussion](https://github.com/memvid/memvid/discussions) for questions
- Check existing [Issues](https://github.com/memvid/memvid/issues) for similar problems
- Email: contact@memvid.com

## Recognition

Contributors are:
- Listed in release notes
- Part of the Memvid community

---

**Thank you for making Memvid better!**
