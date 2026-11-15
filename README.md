# MLOps DSL

A domain-specific language (DSL), implemented in [Rascal](https://www.rascal-mpl.org/), for describing ML pipelines (loading, splitting, feature selection, transformation, model training, evaluation, deployment, and monitoring) in a declarative way.

## Repository structure

This repository is a monorepo with two tightly coupled sub-projects:

```
.
├── mlopsdsl/            The actual language implementation (Rascal, Python, Maven)
└── mlopsdsl-extension/  TypeScript implementation that packages mlopsdsl as a `.vsix`
```

Both folders belong together and are intentionally versioned together: `mlopsdsl-extension` contains no logic of its own — only packaging and activation code for exactly the JAR built from `mlopsdsl`.

## `mlopsdsl` — Language implementation

Contains:

- **Grammar** (`Syntax.rsc`, `AST.rsc`) — concrete and abstract syntax for parsing `.mlops` files
- **Type checking** (`TypeSystem.rsc`) — including schema inference from CSV files via an external Python script
- **Interpreter** (`Interpreter.rsc`) — execution of pipeline to generate model and deployment artifacts via external Python script
- **Code Generation** (`CodeGen.rsc`) — called by `Interpreter.rsc` to generate deployment artifacts
- **Checker** (`Checker.rsc`) — uses parts of type `TypeSystem.rsc` and `Interpreter.rsc` to enable IDE highlighting
- **Outline** (`Outline.rsc`) — generates tree structure for outline in the IDE 
- **Helper modules** (`PythonBridge.rsc`, `Store.rsc`, `InlineFixes.rsc`) — starts/communicates with Python helper scripts (`src/main/python/`) at runtime, encapsulates stores and related methods for typing and execution, implements quick fixes in IDE
- **Language services** (`Plugin.rsc`) — registers parser, type checking, code lenses, and execution with the Rascal LSP

### Requirements

- Java 11+ and Maven
- Rascal VS Code extension 0.13.5 (for the `main()` test mode during development)
- Python 3.11+
- (Optional) Docker

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

### Language examples

Examples of .mlops files can be found in [src/main/mlops](mlopsdsl/src/main/mlops), the datasets they use are located in [data/](mlopsdsl/data) and the generated artifacts live in [models/](mlopsdsl/models). For the full [demo](mlopsdsl/src/main/mlops/demo.mlops) to work properly Docker is required to first run the [PostgreSQL database](mlopsdsl/data/advanced/docker-compose.db.yml). Afterwards run `docker compose up` [here](mlopsdsl/models/demo) to start an API that will be reachable under `localhost:8000/predict` responding to requests with the JSON format e.g. `{"Pclass": 1, "Sex": "female", "Age": 58.0, "SibSp": 0, "Parch": 0, "Fare": 26.55}`. Not generated automatically after running but still included for minimal example running is the Python script in [minimal_ex](mlopsdsl/models/minimal_ex) which loads the generated model artifact and calls an example prediction.

## `mlopsdsl-extension` — VS Code Extension

Packages the JAR built from `mlopsdsl` together with the Python scripts (excluding `.mlopsenv`) into an installable `.vsix` file, so users can use the language without their own Rascal installation.

### Additional Requirements

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
