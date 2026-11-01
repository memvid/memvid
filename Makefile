.PHONY: help build build-release test test-verbose clean fmt fmt-check clippy clippy-fix doc doc-open check install run-example docker-build docker-test docker-push pre-commit tree outdated bloat test-doc test-release package publish-dry-run coverage watch

# Default target
.DEFAULT_GOAL := help

# Variables
CARGO := cargo
RUST_VERSION := 1.88.0
FEATURES := lex,pdf_extract

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(CYAN)Memvid Makefile Commands:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

install: ## Install Rust toolchain and dependencies
	@echo "$(CYAN)Installing Rust toolchain...$(NC)"
	@rustup toolchain install $(RUST_VERSION) || true
	@rustup default $(RUST_VERSION) || true
	@echo "$(CYAN)Installing cargo dependencies...$(NC)"
	@$(CARGO) fetch

download-models: ## Download required models and dictionaries
	@echo "$(CYAN)Downloading SymSpell dictionaries...$(NC)"
	@mkdir -p data
	@curl -L https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell/frequency_dictionary_en_82_765.txt -o data/frequency_dictionary_en_82_765.txt
	@curl -L https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell/frequency_bigramdictionary_en_243_342.txt -o data/frequency_bigramdictionary_en_243_342.txt


check: ## Check code without building
	@echo "$(CYAN)Checking code...$(NC)"
	@$(CARGO) check --features $(FEATURES)

build: ## Build in debug mode
	@echo "$(CYAN)Building in debug mode...$(NC)"
	@$(CARGO) build --features $(FEATURES)

build-release: ## Build in release mode (optimized)
	@echo "$(CYAN)Building in release mode...$(NC)"
	@$(CARGO) build --release --features $(FEATURES)

build-verbose: ## Build with verbose output
	@echo "$(CYAN)Building with verbose output...$(NC)"
	@$(CARGO) build --verbose --features $(FEATURES)

build-release-verbose: ## Build release with verbose output
	@echo "$(CYAN)Building release with verbose output...$(NC)"
	@$(CARGO) build --release --verbose --features $(FEATURES)

build-all-features: ## Build with all features enabled
	@echo "$(CYAN)Building with all features...$(NC)"
	@$(CARGO) build --release --all-features

test: ## Run tests
	@echo "$(CYAN)Running tests...$(NC)"
	@$(CARGO) test --features $(FEATURES)

test-verbose: ## Run tests with output
	@echo "$(CYAN)Running tests with output...$(NC)"
	@$(CARGO) test --features $(FEATURES) -- --nocapture

test-release: ## Run tests in release mode
	@echo "$(CYAN)Running tests in release mode...$(NC)"
	@$(CARGO) test --release --features $(FEATURES)

test-doc: ## Run documentation tests only
	@echo "$(CYAN)Running documentation tests...$(NC)"
	@$(CARGO) test --doc --features $(FEATURES)

test-all-targets: ## Run all test targets (lib, bins, tests, examples)
	@echo "$(CYAN)Running all test targets...$(NC)"
	@$(CARGO) test --all-targets --features $(FEATURES)

test-no-fail-fast: ## Run all tests even if one fails
	@echo "$(CYAN)Running all tests (no fail fast)...$(NC)"
	@$(CARGO) test --features $(FEATURES) -- --no-fail-fast

test-integration: ## Run integration tests only
	@echo "$(CYAN)Running integration tests...$(NC)"
	@$(CARGO) test --test lifecycle --test search --test mutation --test crash_recovery --test doctor_recovery --test encryption_capsule --test replay_integrity --test single_file --features $(FEATURES)

test-unit: ## Run unit tests only
	@echo "$(CYAN)Running unit tests...$(NC)"
	@$(CARGO) test --lib --features $(FEATURES)

fmt: ## Format code
	@echo "$(CYAN)Formatting code...$(NC)"
	@$(CARGO) fmt --all

fmt-check: ## Check code formatting
	@echo "$(CYAN)Checking code formatting...$(NC)"
	@$(CARGO) fmt --all -- --check

clippy: ## Run clippy linter
	@echo "$(CYAN)Running clippy...$(NC)"
	@$(CARGO) clippy --all-targets --features $(FEATURES) -- -D warnings

clippy-fix: ## Run clippy and auto-fix issues
	@echo "$(CYAN)Running clippy with auto-fix...$(NC)"
	@$(CARGO) clippy --fix --all-targets --features $(FEATURES) -- -D warnings

doc: ## Generate documentation
	@echo "$(CYAN)Generating documentation...$(NC)"
	@$(CARGO) doc --features $(FEATURES) --no-deps

doc-open: ## Generate and open documentation
	@echo "$(CYAN)Generating and opening documentation...$(NC)"
	@$(CARGO) doc --features $(FEATURES) --no-deps --open

clean: ## Clean build artifacts
	@echo "$(CYAN)Cleaning build artifacts...$(NC)"
	@$(CARGO) clean

clean-all: clean ## Clean everything including target directory
	@echo "$(CYAN)Cleaning all artifacts...$(NC)"
	@rm -rf target/

run-example-basic: ## Run basic_usage example
	@echo "$(CYAN)Running basic_usage example...$(NC)"
	@$(CARGO) run --example basic_usage --features $(FEATURES)

run-example-pdf: ## Run pdf_ingestion example (usage: make run-example-pdf PDF_PATH=/path/to/pdf)
	@if [ -z "$(PDF_PATH)" ]; then \
		echo "$(YELLOW)Usage: make run-example-pdf PDF_PATH=/path/to/pdf$(NC)"; \
		echo "$(YELLOW)Example: make run-example-pdf PDF_PATH=examples/1706.03762v7.pdf$(NC)"; \
		exit 1; \
	fi
	@echo "$(CYAN)Running pdf_ingestion example with $(PDF_PATH)...$(NC)"
	@$(CARGO) run --example pdf_ingestion --features $(FEATURES) -- $(PDF_PATH)

run-example-clip: ## Run clip_visual_search example (usage: make run-example-clip PDF_PATH=/path/to/pdf)
	@if [ -z "$(PDF_PATH)" ]; then \
		echo "$(YELLOW)Usage: make run-example-clip PDF_PATH=/path/to/pdf$(NC)"; \
		echo "$(YELLOW)Example: make run-example-clip PDF_PATH=examples/document.pdf$(NC)"; \
		exit 1; \
	fi
	@echo "$(CYAN)Running clip_visual_search example with $(PDF_PATH)...$(NC)"
	@$(CARGO) run --example clip_visual_search --features $(FEATURES),clip -- $(PDF_PATH)

run-example-whisper: ## Run test_whisper example (usage: make run-example-whisper AUDIO_PATH=/path/to/audio)
	@if [ -z "$(AUDIO_PATH)" ]; then \
		echo "$(YELLOW)Usage: make run-example-whisper AUDIO_PATH=/path/to/audio$(NC)"; \
		echo "$(YELLOW)Example: make run-example-whisper AUDIO_PATH=examples/call_sale.mp3$(NC)"; \
		exit 1; \
	fi
	@echo "$(CYAN)Running test_whisper example with $(AUDIO_PATH)...$(NC)"
	@$(CARGO) run --example test_whisper --features $(FEATURES),whisper -- $(AUDIO_PATH)

lint: fmt-check clippy ## Run all linting checks

verify: check lint test ## Run all verification checks (check, lint, test)

ci: verify build-release ## Run CI pipeline (verify + release build)

docker-build: ## Build Docker image
	@echo "$(CYAN)Building Docker image...$(NC)"
	@cd docker/cli && docker build -t memvid/cli:latest .

docker-test: ## Test Docker image
	@echo "$(CYAN)Testing Docker image...$(NC)"
	@cd docker/cli && ./test.sh

docker-push: ## Push Docker image to registry (requires login)
	@echo "$(CYAN)Pushing Docker image...$(NC)"
	@docker push memvid/cli:latest

docker-tag: ## Tag Docker image with version
	@echo "$(CYAN)Tagging Docker image...$(NC)"
	@VERSION=$$(grep "^version" Cargo.toml | cut -d'"' -f2); \
	docker tag memvid/cli:latest memvid/cli:$$VERSION; \
	echo "Tagged as memvid/cli:$$VERSION"

bench: ## Run benchmarks (if available)
	@echo "$(CYAN)Running benchmarks...$(NC)"
	@$(CARGO) bench --features $(FEATURES) || echo "$(YELLOW)No benchmarks found$(NC)"

update: ## Update dependencies
	@echo "$(CYAN)Updating dependencies...$(NC)"
	@$(CARGO) update

audit: ## Audit dependencies for security vulnerabilities
	@echo "$(CYAN)Auditing dependencies...$(NC)"
	@$(CARGO) audit || echo "$(YELLOW)cargo-audit not installed. Install with: cargo install cargo-audit$(NC)"

version: ## Show version information
	@echo "$(CYAN)Version Information:$(NC)"
	@$(CARGO) --version
	@rustc --version
	@echo ""
	@echo "$(CYAN)Project version:$(NC)"
	@grep "^version" Cargo.toml

tree: ## Show dependency tree
	@echo "$(CYAN)Dependency tree:$(NC)"
	@$(CARGO) tree --features $(FEATURES)

tree-duplicates: ## Show duplicate dependencies
	@echo "$(CYAN)Duplicate dependencies:$(NC)"
	@$(CARGO) tree --duplicates --features $(FEATURES)

outdated: ## Check for outdated dependencies
	@echo "$(CYAN)Checking for outdated dependencies...$(NC)"
	@$(CARGO) outdated || echo "$(YELLOW)cargo-outdated not installed. Install with: cargo install cargo-outdated$(NC)"

bloat: ## Analyze binary size bloat
	@echo "$(CYAN)Analyzing binary size...$(NC)"
	@$(CARGO) bloat --release --features $(FEATURES) || echo "$(YELLOW)cargo-bloat not installed. Install with: cargo install cargo-bloat$(NC)"

package: ## Create a package for publishing
	@echo "$(CYAN)Creating package...$(NC)"
	@$(CARGO) package

publish-dry-run: ## Dry run of publish (check if ready)
	@echo "$(CYAN)Running publish dry-run...$(NC)"
	@$(CARGO) publish --dry-run

coverage: ## Generate test coverage report
	@echo "$(CYAN)Generating test coverage...$(NC)"
	@$(CARGO) tarpaulin --features $(FEATURES) --out Html || echo "$(YELLOW)cargo-tarpaulin not installed. Install with: cargo install cargo-tarpaulin$(NC)"

watch: ## Watch for changes and run tests
	@echo "$(CYAN)Watching for changes...$(NC)"
	@$(CARGO) watch -x "test --features $(FEATURES)" || echo "$(YELLOW)cargo-watch not installed. Install with: cargo install cargo-watch$(NC)"

watch-build: ## Watch for changes and rebuild
	@echo "$(CYAN)Watching for changes and rebuilding...$(NC)"
	@$(CARGO) watch -x "build --features $(FEATURES)" || echo "$(YELLOW)cargo-watch not installed. Install with: cargo install cargo-watch$(NC)"

pre-commit: fmt-check clippy test ## Run pre-commit checks (fmt, clippy, test)
