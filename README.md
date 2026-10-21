# MLOps DSL

A domain-specific language (DSL), implemented in [Rascal](https://www.rascal-mpl.org/), for describing ML pipelines (loading, splitting, feature selection, transformation, model training, evaluation, deployment, and monitoring) in a declarative way.

## Repository structure

This repository is a monorepo with two tightly coupled sub-projects:

```
.
├── mlopsdsl/            The actual language implementation (Rascal, Maven)
└── mlopsdsl-extension/  VS Code extension that packages mlopsdsl as a standalone JAR
```

Both folders belong together and are intentionally versioned together: `mlopsdsl-extension` contains no logic of its own — only packaging and activation code for exactly the JAR built from `mlopsdsl`.

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

This loads the language and its Python scripts directly from the local source folders (`src/main/rascal`, `src/main/python`) — including a virtual Python environment created locally under `src/main/.mlopsenv`.

## `mlopsdsl-extension` — VS Code Extension

Packages the JAR built from `mlopsdsl` together with the Python scripts (excluding `.mlopsenv`) into an installable `.vsix` file, so users can use the language without their own Rascal installation.

### Requirements

- Node.js + npm

### Building

```bash
cd mlopsdsl-extension
npm install
npx vsce package     # produces the .vsix after running npm run prepackage (producing JAR) and npm run compile (compiling TypeScript) automatically
```

### Runtime behavior

- **Java**: Automatically detected by the underlying library (`@usethesource/rascal-vscode-dsl-lsp-server`); if missing, VS Code automatically offers a download/installation dialog.
- **Python**: Managed by the extension itself. On first opening a `.mlops` file, it checks whether a virtual environment already exists at VS Code's own persistent storage location (`globalStorage`); if not, it creates one automatically.

## Language example

```text
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
