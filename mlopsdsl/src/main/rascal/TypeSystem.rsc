module TypeSystem

import IO;
import String;
import List;
import Set;
import Map;
import util::ShellExec;
import lang::json::IO;
import ParseTree;
import util::IDEServices;

import AST;
import Syntax;
import PythonBridge;
import Store;

data TypeException
    = unknownColumn(str cause)
    | incompatibleTransform(str cause)
    | incompatibleTarget(str cause)
    | incompatibleMetric(str cause)
    | unknownHyperparam(str cause)
    | hyperparamTypeMismatch(str cause)
    | schemaInferenceFailed(str cause)
    | databaseConnectionFailed(str cause)
    | incompatibleMethod(str cause)
    ;

Schema inferSchema(loc csvPath, PID pid) {
    str rawResponse = "";
    int tries = 0;
    while (rawResponse == "" && tries < 60) {
        rawResponse = readWithWait(pid, 500);
        tries += 1;
        if (!isAlive(pid) && rawResponse == "") {
            rawResponse = readWithWait(pid, 200);
            if (rawResponse == "") {
                str err = readFromErr(pid);
                throw schemaInferenceFailed("Schema inference process crashed: <err>");
            }
            
        }
    }
    if (rawResponse == "") {
        killProcess(pid, force=true);
        throw schemaInferenceFailed("Timeout: schema inference did not respond for <csvPath>");
    }

    killProcess(pid, force=true);

    SchemaJson resp = parseJSON(#SchemaJson, trim(rawResponse));
    if (resp.status == "FILE_ERROR") {
        throw schemaInferenceFailed(resp.message);
    }
    if (resp.status == "DB_ERROR") {
        throw databaseConnectionFailed(resp.message);
    }

    Schema schema = ();
    for (str col <- resp.columns) {
        ColumnInfoJson entry = resp.columns[col];
        schema[col] = colInfo(parseColumnType(entry.\type));
    }
    return schema;
}

ColumnType parseColumnType("integer") = tInteger();

ColumnType parseColumnType("float") = tFloat();

ColumnType parseColumnType("categorical") = tCategorical();

ColumnType parseColumnType("boolean") = tBoolean();

default ColumnType parseColumnType(str _) = tUnknown();

TypeStore checkPipeline(Syntax::Pipeline pipeline) {
    pipelineAST = implode(#AST::Pipeline, pipeline);
    return checkPipeline(pipelineAST);
}

TypeStore checkPipeline(pipeline(str name, Steps steps)) {
    return checkSteps(steps);
}

TypeStore checkSteps(steps(Load load, list[Split] _, list[Select] select, list[Trans] trans, Model model, list[Eval] eval, list[Deploy] _, list[Monitor] monitor)) {
    TypeStore store = checkLoad(load);

    if (size(select) != 0) {
        store = checkSelect(select[0], store);
    }

    if (size(trans) != 0) {
        store = checkTrans(trans[0], store);
    }

    store = checkModel(model, store);

    if (size(eval) != 0) {
        checkEval(eval[0], store);
    }
    if (size(monitor) != 0) {
        checkMonitor(monitor[0], store);
    }
    return store;
}

TypeStore checkLoad(Load l:stepLoad(StrLit path, StrLit target, list[StrLit] dbURL)) {
    str p = path.content;
    loc baseDir = l.src.parent;
    loc scriptPath = getPath("schema_infer.py");
    loc csvPath = baseDir + p;
    PID pid = createProcess(PythonBridge::getPythonExecutable(), args=size(dbURL) != 0 ? [scriptPath, csvPath.top, dbURL[0].content] : [scriptPath, csvPath.top]);
    if (!isAlive(pid)) {
        throw schemaInferenceFailed("Could not start schema inference process for <csvPath>");
    }
    Schema schema = ();
    try {
        schema = inferSchema(csvPath, pid);
    }
    catch databaseConnectionFailed(str cause): {
        throw databaseConnectionFailed(cause);
    }
    if (target.content notin schema) {
        throw unknownColumn("Target column \'<target.content>\' not found in dataset.");
    }
    return tStore(target.content, schema, null());
}

TypeStore checkSelect(stepSelect(list[StrLit] features), TypeStore store) {
    Schema schema = store.schema;
    Schema newSchema = ();
    for (StrLit f <- features) {
        str feat = f.content;
        requireColumn(schema, feat);
        newSchema[feat] = schema[feat];
    }
    newSchema[store.target] = schema[store.target];
    return tStore(store.target, newSchema, null());
}

TypeStore checkTrans(stepTrans(list[PrepTransform] transforms), TypeStore store) {
    Schema schema = store.schema;
    for (PrepTransform pt <- transforms) {
        schema = checkAndApplyTransform(pt, schema);
    }
    return tStore(store.target, schema, null());
}

Schema checkAndApplyTransform(prepFill(StrLit feature, FillStrategy strategy), Schema schema) {
    str feat = feature.content;
    requireColumn(schema, feat);
    if ((strategy is fillMean || strategy is fillMedian) && (schema[feat].ctype notin {tInteger(), tFloat()})) {
        throw incompatibleTransform("fillna(mean/median) requires a numeric column, but \'<feat>\' is <schema[feat].ctype>.");
    }
    return schema;
}

Schema checkAndApplyTransform(prepScale(StrLit feature, ScaleMethod _), Schema schema) {
    str feat = feature.content;
    requireColumn(schema, feat);
    switch(schema[feat]) {
        case colInfo(ctype): {
            if (ctype notin {tInteger(), tFloat()}) {
                throw incompatibleTransform("scale(...) requires a numeric column, but \'<feat>\' is <schema[feat].ctype>.");
            }
            schema = schema + (feat: colInfo(tFloat()));
        }
        case colInfoEnc(tInteger(), method, oldType): {
            schema = schema + (feat: colInfoEnc(tFloat(), method, oldType));
        }
    }
    return schema;
}

Schema checkAndApplyTransform(prepEncode(StrLit feature, EncodingMethod method), Schema schema) {
    str feat = feature.content;
    requireColumn(schema, feat);
    ColumnType oldType = schema[feat].ctype;
    return schema + (feat: colInfoEnc(tInteger(), method, oldType));
}

private void requireColumn(Schema schema, str feat) {
    if (feat notin schema) {
        throw unknownColumn("Column \'<feat>\' not found.");
    }
    if (colInfoEnc(_, encOneHot(), _) := schema[feat]) {
        throw unknownColumn("Column \'<feat>\' is no longer present due to it being onehot encoded.");
    }
}

TypeStore checkModel(stepModel(ModelExpr expr), TypeStore store) {
    Task task = taskOf(expr.algo);
    Schema schema = store.schema;
    str target = store.target;
    ColumnInfo targetInfo = schema[target];

    switch (task) {
        case regression(): {
            if (targetInfo.ctype notin {tInteger(), tFloat()}) {
                throw incompatibleTarget("Algorithm is a regression model and requires a numeric target, but \'<target>\' is <targetInfo.ctype>.");
            }
        }
        case classification(): {
            bool ok = targetInfo.ctype == tCategorical()
                   || targetInfo.ctype == tBoolean()
                   || (targetInfo.ctype == tInteger());
            if (!ok) {
                throw incompatibleTarget(
                    "Algorithm is a classification model and requires a categorical/boolean/integer target " +
                    "but \'<target>\' is <targetInfo.ctype>."
                );
            }
        }
    }
    checkHyperparams(expr);
    return tStore(store.target, store.schema, task);
}

void checkHyperparams(modelTrain(Algorithm algo, StrLit _, list[Param] hyperParams)) {
    ParamSignature sig = signatureOf(algo);
    for (hp(str name, Lit val) <- hyperParams) {
        if (name notin sig) {
            throw unknownHyperparam("\'<name>\' is not a valid hyperparameter for <algo>.");
        }
        if (!paramTypeMatches(val, sig[name])) {
            throw hyperparamTypeMismatch("\'<name>\' has an incompatible value for <algo>: expected <sig[name]>.");
        }
    }
}

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

void checkEval(stepEval(set[EvalRule] evalRules), TypeStore store) {
    Task task = store.task;
    for (EvalRule evalRule <- evalRules) {
        if (!ruleMatchesTask(evalRule, task)) {
            throw incompatibleMetric("Metric <metricName(evalRule)> is not compatible with a <task> model.");
        }
    }
}

Task taskOf(algoLR()) = regression();

Task taskOf(algoRF()) = classification();

Task taskOf(algoLogReg()) = classification();

bool ruleMatchesTask(evalCRule(_, _), classification()) = true;

bool ruleMatchesTask(evalRRule(_, _), regression()) = true;

default bool ruleMatchesTask(EvalRule _, Task _) = false;

str metricName(evalCRule(CMetric m, _)) = m.name;

str metricName(evalRRule(RMetric m, _)) = m.name;

void checkMonitor(Monitor mon, TypeStore store) {
    Schema schema = store.schema;
    for (DriftRule driftRule <- driftRulesOf(mon)) {
        str feat = driftRule.feature.content;
        if (feat notin schema) {
            throw unknownColumn("Column \'<feat>\' not found.");
        }
        ColumnType oldType = schema[feat].ctype;
        if (colInfoEnc(_,_,oType) := schema[feat]) {
            oldType = oType;
        }
        switch(driftRule.dMethod) {
            case dmKS(): {
                if (oldType in {tInteger(), tCategorical(), tBoolean()}) {
                    showMessage(warning("The Kolmogorow-Smirnow method is normally only used for numerical features but <feat> is type <oldType>", mon.src));
                }
            }
            case dmChiSquare(): {
                if (oldType in {tInteger(), tFloat()}) {
                    showMessage(warning("The Chi² method is normally only used for categorical features but <feat> is type <oldType>", mon.src));
                }
            }
        }
    }
}

list[DriftRule] driftRulesOf(stepMonitor(list[DriftRule] drs, _)) = drs;
list[DriftRule] driftRulesOf(stepMonitorLatency(_)) = [];