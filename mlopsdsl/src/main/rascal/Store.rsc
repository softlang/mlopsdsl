module Store

import util::IDEServices;
import util::Maybe;
import AST;

data PipelineState
    = uninitialized()
    | dataLoaded()
    | dataSplit()
    | featuresSelected()
    | transformed()
    | modelTrained()
    | modelEvaluated()
    | deployed(int port);

data MLOpsStore = iStore(str name, PipelineState state, str targetVariable, bool dbConnection, str trainedModelFilePath, bool monitored, Maybe[int] latency)
                | cStore(str targetVariable, bool dbConnection, Task task, rel[loc, Message] messages)
                | empty();

data ColumnType
    = tInteger()
    | tFloat()
    | tCategorical()
    | tBoolean()
    | tUnknown()
    ;

ColumnType parseColumnType("integer") = tInteger();

ColumnType parseColumnType("float") = tFloat();

ColumnType parseColumnType("categorical") = tCategorical();

ColumnType parseColumnType("boolean") = tBoolean();

default ColumnType parseColumnType(str _) = tUnknown();

data ColumnInfo = colInfo(ColumnType ctype) | colInfoEnc(ColumnType ctype, EncodingMethod method, ColumnType oType);

alias Schema = map[str featName, ColumnInfo info];

data TypeStore = tStore(str target, Schema schema, Task task, rel[loc, Message] messages = {});

data Task = classification() | regression() | null();

data ParamType
    = ptInt()
    | ptFloat()
    | ptBool()
    | ptEnum(set[str] allowed)
    ;

alias ParamSignature = map[str name, ParamType ptype];

data ColumnInfoJson = columnInfoJson(str \type);
data SchemaJson = schemaJson(str status, map[str, ColumnInfoJson] columns = (), str message = "");

map[Algorithm, ParamSignature] ALGO_SIGNATURES = (
    AST::algoLR(): (
        "fit_intercept": ptBool(),
        "positive": ptBool()
    ),
    AST::algoRF(): (
        "n_estimators": ptInt(),
        "max_depth": ptInt(),
        "criterion": ptEnum({"gini", "entropy", "log_loss"}),
        "bootstrap": ptBool(),
        "random_state": ptInt()
    ),
    AST::algoLogReg(): (
        "max_iter": ptInt(),
        "C": ptFloat(),
        "penalty": ptEnum({"l1", "l2", "elasticnet", "none"}),
        "random_state": ptInt()
    )
);

ParamSignature signatureOf(algoLR()) = (
    "fit_intercept": ptBool(),
    "positive": ptBool()
);

ParamSignature signatureOf(algoRF()) = (
    "n_estimators": ptInt(),
    "max_depth": ptInt(),
    "criterion": ptEnum({"gini", "entropy", "log_loss"}),
    "bootstrap": ptBool(),
    "random_state": ptInt()
);

ParamSignature signatureOf(algoLogReg()) = (
    "max_iter": ptInt(),
    "C": ptFloat(),
    "penalty": ptEnum({"l1", "l2", "elasticnet", "none"}),
    "random_state": ptInt()
);

bool paramTypeMatches(intLit(_), ptInt()) = true;

bool paramTypeMatches(intLit(_), ptFloat()) = true;

bool paramTypeMatches(floatLit(_), ptFloat()) = true;

bool paramTypeMatches(boolLit(_), ptBool()) = true;

bool paramTypeMatches(strLit(strLit(str s)), ptEnum(set[str] allowed)) = s in allowed;

default bool paramTypeMatches(Lit _, ParamType _) = false;

Task taskOf(algoLR()) = regression();

Task taskOf(algoRF()) = classification();

Task taskOf(algoLogReg()) = classification();