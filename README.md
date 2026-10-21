# MLOps DSL

A domain-specific language (DSL), implemented in [Rascal](https://www.rascal-mpl.org/), for describing ML pipelines (loading, splitting, feature selection, transformation, model training, evaluation, deployment, monitoring) — including a VS Code extension that makes the language usable without a local Rascal installation.

## Repository structure

This repository is a monorepo with two tightly coupled sub-projects:

```
.
├── mlopsdsl/            The actual language implementation (Rascal, Maven)
└── mlopsdsl-extension/  VS Code extension that packages mlopsdsl as a standalone JAR
```

Both folders belong together and are intentionally versioned together: `mlopsdsl-extension` contains no logic of its own — only packaging and activation code for exactly the JAR built from `mlopsdsl`. Module/function names (`Plugin::pipelineLanguageServices`) and the convention for the Python environment path (`.mlopsenv/bin/python3` resp. `.mlopsenv/Scripts/python.exe`) are hard-coded across both projects and must be kept in sync.

## `mlopsdsl` — Language implementation

Contains:

- **Grammar** (`Syntax.rsc`) — parser for `.mlops` files
- **Type checking** (`TypeSystem.rsc`, `Checker.rsc`) — including schema inference from CSV files via an external Python script
- **Python bridge** (`PythonBridge.rsc`) — starts/communicates with Python helper scripts (`src/main/python/`) at runtime
- **Language services** (`Plugin.rsc`) — registers parser, type checking, code lenses, and execution with the Rascal LSP

### Requirements

- Java 11+ and Maven
- Rascal VS Code extension 0.13.5 (for the `main()` test mode during development)

### Building

```bash
cd mlopsdsl
mvn clean package
```

### Registering manually for testing

In the Rascal terminal inside VS Code:

```rascal
import Plugin;
main();
```

This loads the language and its Python scripts directly from the local source folders (`src/main/rascal`, `src/main/python`) — including a virtual Python environment created locally under `src/main/python/.mlopsenv` (see `src/main/python/requirements.txt`).

## `mlopsdsl-extension` — VS Code Extension

Packages the JAR built from `mlopsdsl` together with the Python scripts (excluding `.mlopsenv`) into an installable `.vsix` file, so users can use the language without their own Rascal installation.

### Requirements

- Node.js + npm
- An already-built `mlopsdsl` JAR (created automatically by the `prepackage` script)

### Building

```bash
cd mlopsdsl-extension
npm install
npm run prepackage   # rebuilds mlopsdsl and copies the JAR + Python scripts here
npm run compile
npx vsce package     # produces the .vsix

### Runtime behavior

- **Java**: Automatically detected by the underlying library (`@usethesource/rascal-vscode-dsl-lsp-server`); if missing, VS Code automatically offers a download/installation dialog.
- **Python**: Managed by the extension itself. On first opening a `.mlops` file, it checks whether a virtual environment already exists at VS Code's own persistent storage location (`globalStorage`); if not, it offers to set one up (`python -m venv` + `pip install -r requirements.txt`). If Python itself is missing, this is detected and a link to the official download page is offered. This can be re-triggered at any time via the **"MLOps DSL: (Re-)setup Python environment"** command (command palette).

## Language example

```
pipeline example {
  load(path = "data/raw.csv", target = "label")
  split(train_size = 0.8, random_state = 42)
  select(features = ["age", "income", "score"])
  transformation(
    fillna("age", mean),
    encode("income", onehot),
    scale("score", std)
  )
  model LinReg(dir = "models/linreg", learning_rate = 0.01)
  evaluation(accuracy, f1)
  deployment(port = 8080)
  monitoring(
    driftKS(feature = "age", window = 1000) <= 0.05,
    latency <= 200
  )
}
```

## License

_TODO: add license._
