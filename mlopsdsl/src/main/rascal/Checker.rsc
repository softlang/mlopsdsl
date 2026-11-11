module Checker

import ParseTree;
import util::LanguageServer;
import util::IDEServices;
import util::ShellExec;
import lang::json::IO;
import String;
import List;
import Set;
import util::Maybe;
import util::Math;

import Syntax;
import AST;
import PythonBridge;

data MLOpsStore = mStore(str targetVariable, bool dbConnection, Task task, rel[loc, Message] messages);

data ColumnType
    = tInteger()
    | tFloat()
    | tCategorical()
    | tBoolean()
    | tUnknown()
    ;

data ColumnInfo = colInfo(ColumnType ctype) | colInfoEnc(ColumnType ctype, EncodingMethod method, ColumnType oType);

alias Schema = map[str featName, ColumnInfo info];

data TypeStore = tStore(str target, Schema schema, Task task, rel[loc, Message] messages);

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

Summary mlopsAnalysisService(loc l, start[Pipeline] input) 
    = mlopsSummaryServiceSem(l, implode(#AST::Pipeline, input));

Summary mlopsBuildService(loc l , start[Pipeline] input)
    = mlopsSummaryServiceType(l, implode(#AST::Pipeline, input));

Summary mlopsSummaryServiceSem(loc l, AST::Pipeline input) {
    Summary s = summary(l);
    MLOpsStore store = initMLOpsStore();
    store = checkStepsSem(input, store);
    s.messages += store.messages;
    return s;
}

MLOpsStore initMLOpsStore() {
    return mStore("", false, null(), {});
}

MLOpsStore checkStepsSem(AST::Pipeline input, MLOpsStore store) {
    AST::Steps steps = input.steps;
    store = checkLoadSem(steps.load, store);
    if (size(steps.split) != 0) {
        store = checkSplitSem(steps.split[0], store);
    }
    if (size(steps.select) != 0) {
        store = checkSelectSem(steps.select[0], store);
    }
    if (size(steps.trans) != 0) {
        store = checkTransSem(steps.trans[0], store);
    }
    store = checkModelSem(steps.model, store);
    if (size(steps.eval) != 0) {
        store = checkEvalSem(steps.eval[0], store);
    }
    if (size(steps.monitor) != 0) {
        if (size(steps.deploy) == 0) {
            store.messages += {<steps.monitor[0].src, error("User inputs cannot be monitored without deployed model.", steps.monitor[0].src,
                fixes=prepareNoDeploymentFixes(steps.monitor[0].src))>};
        } else {
            store = checkMonitorSem(steps.monitor[0], store);
        }
    }
    return store;
}

MLOpsStore checkLoadSem(AST::Load load, MLOpsStore store) {
    return mStore(load.target.content, size(load.dbURL) != 0, store.task, store.messages);
}

MLOpsStore checkSplitSem(AST::Split split, MLOpsStore store) {
    if (split.trainSize <= 0) {
        store.messages += {<split.src, error("Train size is below 0.0",split.src)>};
    }
    if (split.trainSize >= 1) {
        store.messages += {<split.src, error("Train size exceeds 1.0",split.src)>};
    }
    return mStore(store.targetVariable, store.dbConnection, store.task, store.messages);
}

MLOpsStore checkSelectSem(AST::Select select, MLOpsStore store) {
    set[str] seen = {};
    list[loc] featureLocs = [f.src | f <- select.features];
    for (int i <- [0..size(select.features)]) {
        AST::StrLit f = select.features[i];
        if (f.content == store.targetVariable) {
            store.messages += {<f.src, error("The target column cannot be a feature.", f.src,
                fixes=removeEntryFix("Remove target from selected features", featureLocs, i))>};
        }
        if (f.content in seen) {
            store.messages += {<f.src, error("Feature \'<f.content>\' is selected more than once.", f.src,
                fixes=removeEntryFix("Remove duplicate feature", featureLocs, i))>};
        } else {
            seen += {f.content};
        }
    }
    return mStore(store.targetVariable, store.dbConnection, store.task, store.messages);
}

MLOpsStore checkTransSem(AST::Trans trans, MLOpsStore store) {
    map[str feat, lrel[str tr, loc src] entries] byFeature = ();
    list[loc] transformLocs = [pt.feature.src | pt <- trans.transforms];
    for (int i <- [0..size(trans.transforms)]) {
        AST::PrepTransform pt = trans.transforms[i];
        tuple[str tr, str feat] ptrans = evalPrepTransform(pt);
        loc src = pt.feature.src;
        if (ptrans.feat == store.targetVariable) {
            store.messages += {<src, error("The target column cannot be transformed.", src,
                fixes=removeEntryFix("Remove transformation of target column", transformLocs, i))>};
            continue;
        }
        byFeature[ptrans.feat] = (byFeature[ptrans.feat] ? []) + <ptrans.tr, src>;
    }
    for (str feat <- byFeature) {
        set[str] seenTrans = {};
        bool sawScale = false;
        bool sawEncode = false;
        for (<str tr, loc src> <- byFeature[feat]) {
            int idx = indexOf(transformLocs, src);
            if (tr in seenTrans) {
                store.messages += {<src, error("Feature must not be transformed multiple times by the same method.", src
                    fixes=removeEntryFix("Remove duplicate transformation", transformLocs, idx))>};
            } else {
                seenTrans += {tr};
            }

            if (sawScale && (tr == "fillna" || tr == "encode")) {
                store.messages += {<src, error("Scaling must be the last transformation of a feature.", src)>};
            }
            if (tr == "scale") {
                sawScale = true;
            }

            if (sawEncode && tr == "fillna") {
                store.messages += {<src, error("Missing values must be filled before encoding a feature.", src)>};
            }
            if (tr == "encode") {
                sawEncode = true;
            }
        }
    }

    return mStore(store.targetVariable, store.dbConnection, store.task, store.messages);
}

tuple[str tr, str feat] evalPrepTransform(prepFill(StrLit feature, FillStrategy _)) {
    str feat = feature.content;
    return <"fillna", feat>;
}

tuple[str tr, str feat] evalPrepTransform(prepEncode(StrLit feature, EncodingMethod _)) {
    str feat = feature.content;
    return <"encode", feat>;
}

tuple[str tr, str feat] evalPrepTransform(prepScale(StrLit feature, ScaleMethod _)) {
    str feat = feature.content;
    return <"scale", feat>;
}

MLOpsStore checkModelSem(AST::Model model, MLOpsStore store) {
    store = checkHyperparamsSem(model.expr, store);
    return mStore(store.targetVariable, store.dbConnection, taskOf(model.expr.algo), store.messages);
}

MLOpsStore checkHyperparamsSem(AST::ModelExpr expr, MLOpsStore store) {
    ParamSignature sig = signatureOf(expr.algo);
    for (Param p <- expr.hyperParams) {
        if (p.name notin sig) {
            store.messages += {<p.src, error("\'<p.name>\' is not a valid hyperparameter for <expr.algo.name>.", p.src,
                fixes=invalidParamNameFixes(p.src, p.name, sig))>};
            continue;
        }
        if (!paramTypeMatches(p.val, sig[p.name])) {
            store.messages += {<p.val.src, error("\'<p.name>\' has an incompatible value: expected <sig[p.name]>.", p.val.src)>};
        }
    }
    return mStore(store.targetVariable, store.dbConnection, store.task, store.messages);
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

MLOpsStore checkEvalSem(AST::Eval eval, MLOpsStore store) {
    Task task = store.task;
    for (EvalRule evalRule <- eval.evalRules) {
        store.messages += checkEvalRule(evalRule, task);
    }
    return mStore(store.targetVariable, store.dbConnection, store.task, store.messages);
}

rel[loc, Message] checkEvalRule(EvalRule er:evalCRule(CMetric m, real threshold), Task task) {
    rel[loc, Message] msgs = {};
    if (threshold > 1.0 || threshold < 0) {
        msgs += {<er.src, error("Threshold for \'<m.name>\' must be between 0 and 1", er.src)>};
    }
    if (regression() := task) {
        msgs += {<er.src, error("Metric <m.name> is not compatible with a <task> model.", er.src)>};
    }
    return msgs;
}

rel[loc, Message] checkEvalRule(EvalRule er:evalRRule(RMetric m, real threshold), Task task) {
    rel[loc, Message] msgs = {};
    if (classification() := task) {
        msgs += {<er.src, error("Metric <m.name> is not compatible with a <task> model.", er.src)>};
    }
    return msgs;
}

MLOpsStore checkMonitorSem(AST::Monitor monitor, MLOpsStore store) {
    if (!store.dbConnection) {
        store.messages += {<monitor.src, error("Cannot store user data for monitoring without connected database.", monitor.src,
            fixes=prepareNoConnectionFixes(monitor.src))>};
    }
    list[AST::DriftRule] driftRules = driftRulesOf(monitor);
    list[loc] ruleLocs = [dr.src | dr <- monitor.driftRules];
    set[str] seenFeatures = {};
    for (int i <- [0..size(monitor.driftRules)]) {
        AST::DriftRule driftRule = driftRules[i];
        tuple[str feat, int win, int freq, real minEffect, real threshold] dRule = <driftRule.feature.content, driftRule.window, driftRule.freq, driftRule.minEffect, driftRule.threshold>;

        if (dRule.feat == store.targetVariable) {
            store.messages += {<driftRule.src, error("The target can not be selected as a monitored feature.", driftRule.src,
                fixes=removeEntryFix("Remove monitoring of target column", ruleLocs, i))>};
        }
        if (dRule.feat in seenFeatures) {
            store.messages += {<driftRule.src, error("Feature \'<dRule.feat>\' is monitored more than once.", driftRule.src,
                fixes=removeEntryFix("Remove duplicate monitoring of feature", ruleLocs, i))>};
        } else {
            seenFeatures += {dRule.feat};
        }
        if (dRule.win < 500) {
            store.messages += {<driftRule.src, warning("The window for drift calculation is small. Recommended is 500 or higher.", driftRule.src)>};
        }
        if (dRule.threshold <= 0) {
            store.messages += {<driftRule.src, error("The threshold cannot be negative or 0.", driftRule.src)>};
        }
        if (dRule.freq <= 0) {
            store.messages += {<driftRule.src, error("The drift calculation frequency cannot be negative or 0.", driftRule.src)>};
        }
        if (dmKS() := driftRule.dMethod && dRule.minEffect >= 1) {
            store.messages += {<driftRule.src, error("The minimum effect bound cannot be greater or equal to 1", driftRule.src)>};
        }
        if (dRule.minEffect < 0) {
            store.messages += {<driftRule.src, error("The minimum effect bound cannot be smaller than 0", driftRule.src)>};
        }
    }

    Maybe[int] ms = nothing();
    list[AST::LatencyRule] latencyRule = latencyRuleOf(monitor);
    if (size(latencyRule) != 0) {
        AST::LatencyRule lRule = latencyRule[0];
        int parsedMs = lRule.ms;
        if (parsedMs <= 0) {
            store.messages += {<lRule.src, error("Latency cannot be smaller or equal to 0.", lRule.src)>};
        } else {
            ms = just(parsedMs);
        }
    }

    return mStore(store.targetVariable, store.dbConnection, store.task, store.messages);
}

list[DriftRule] driftRulesOf(stepMonitor(list[DriftRule] drs, _)) = drs;
list[DriftRule] driftRulesOf(stepMonitorLatency(_)) = [];

list[LatencyRule] latencyRuleOf(stepMonitor(_, list[LatencyRule] lr)) = lr;
list[LatencyRule] latencyRuleOf(stepMonitorLatency(LatencyRule l)) = [l];

Summary mlopsSummaryServiceType(loc l, AST::Pipeline input) {
    Summary s = summary(l);
    AST::StrLit path = getCSVPath(input);
    list[AST::StrLit] dbURL = getDBURL(input);
    PID pid = runInferenceScript(l, path, dbURL);
    SchemaJson resp = retrieveResponse(pid);
    killProcess(pid, force=true);
    if (resp.status == "FILE_ERROR") {
        s.messages += {<path.src, error(resp.message, path.src)>};
        return s;
    }
    if (resp.status == "DB_ERROR") {
        AST::StrLit dbURL = getDBURL(input)[0];
        s.messages += {<dbURL.src, error(resp.message, dbURL.src)>};
    }
    TypeStore store = initTypeStore(input, resp);
    store = checkStepsType(input, store);
    s.messages += store.messages;
    return s;
}

AST::StrLit getCSVPath(AST::Pipeline input) {
    AST::Steps steps = input.steps;
    AST::Load load = steps.load;
    return load.path;
}

list[AST::StrLit] getDBURL(AST::Pipeline input) {
    AST::Steps steps = input.steps;
    AST::Load load = steps.load;
    return load.dbURL;
}

PID runInferenceScript(loc l, AST::StrLit path, list[AST::StrLit] dbURL) {
    str p = path.content;
    loc baseDir = l.parent;
    loc csvPath = baseDir + p;
    loc inferenceScript = getPath("schema_infer.py");
    return createProcess(PythonBridge::getPythonExecutable(), args=size(dbURL) != 0 ? [inferenceScript, csvPath.top, dbURL[0].content] : [inferenceScript, csvPath.top]);
}

SchemaJson retrieveResponse(PID pid) {
    str rawResponse = "";
    while(isAlive(pid)) {
        rawResponse = readWithWait(pid, 500);
    }
    if (rawResponse == "") {
        rawResponse = readWithWait(pid, 500);
    }
    return parseJSON(#SchemaJson, trim(rawResponse));
}

TypeStore initTypeStore(AST::Pipeline input, SchemaJson resp) {
    Schema schema = buildSchema(resp);
    return tStore(input.steps.load.target.content, schema, null(), {});
}

Schema buildSchema(SchemaJson resp) {
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

TypeStore checkStepsType(AST::Pipeline input, TypeStore store) {
    AST::Steps steps = input.steps;
    store = checkLoadType(steps.load, store);
    if (size(steps.select) != 0) {
        store = checkSelectType(steps.select[0], store);
    }
    if (size(steps.trans) != 0) {
        store = checkTransType(steps.trans[0], store);
    }
    store = checkModelType(steps.model, store);
    if (size(steps.monitor) != 0) {
        store = checkMonitorTypes(steps.monitor[0], store);
    }
    return store;
}

TypeStore checkLoadType(AST::Load load, TypeStore store) {
    if (load.target.content notin store.schema) {
        store.messages += {<load.target.src, error("Target column \'<load.target.content>\' not found in dataset.", load.target.src,
            fixes=schemaSuggestionFixes(load.target.src, load.target.content, store.schema))>};
    }
    return tStore(load.target.content, store.schema, null(), store.messages);
}

TypeStore checkSelectType(AST::Select select, TypeStore store) {
    Schema schema = store.schema;
    Schema newSchema = ();
    for (StrLit f <- select.features) {
        str feat = f.content;
        rel[loc, Message] featMessages = requireColumnSelect(schema, f);
        store.messages += featMessages;
        if (size(featMessages) != 0) {
            continue;
        }
        newSchema[feat] = schema[feat];
    }
    if (store.target in schema) {
        newSchema[store.target] = schema[store.target];
    }
    return tStore(store.target, newSchema, null(), store.messages);
}

TypeStore checkTransType(AST::Trans trans, TypeStore store) {
    for (PrepTransform pt <- trans.transforms) {
        store = checkAndApplyTransform(pt, store);
    }
    return store;
}

TypeStore checkAndApplyTransform(prepFill(StrLit feature, FillStrategy strategy), TypeStore store) {
    Schema schema = store.schema;
    str feat = feature.content;
    rel[loc, Message] msgs = requireColumn(schema, feature);
    store.messages += msgs;
    if (size(msgs) != 0) {
        return store;
    }
    if ((strategy is fillMean || strategy is fillMedian) && (schema[feat].ctype notin {tInteger(), tFloat()})) {
        store.messages += {<feature.src, error("fillna(mean/median) requires a numeric column, but \'<feat>\' is <schema[feat].ctype>.", feature.src)>};
    }
    return store;
}

TypeStore checkAndApplyTransform(prepScale(StrLit feature, ScaleMethod _), TypeStore store) {
    Schema schema = store.schema;
    str feat = feature.content;
    rel[loc, Message] msgs = requireColumn(schema, feature);
    store.messages += msgs;
    if (size(msgs) != 0) {
        return store;
    }
    switch(schema[feat]) {
        case colInfo(ctype): {
            if (ctype notin {tInteger(), tFloat()}) {
                store.messages += {<feature.src, error("scale(...) requires a numeric column, but \'<feat>\' is <schema[feat].ctype>.", feature.src)>};
            }
            schema = schema + (feat: colInfo(tFloat()));
        }
        case colInfoEnc(tInteger(), method, oldType): {
            schema = schema + (feat: colInfoEnc(tFloat(), method, oldType));
        }
    }
    return tStore(store.target, schema, store.task, store.messages);
}

TypeStore checkAndApplyTransform(prepEncode(StrLit feature, EncodingMethod method), TypeStore store) {
    Schema schema = store.schema;
    str feat = feature.content;
    rel[loc, Message] msgs = requireColumn(schema, feature);
    store.messages += msgs;
    if (size(msgs) != 0) {
        return store;
    }
    ColumnType oldType = schema[feat].ctype;
    schema = schema + (feat: colInfoEnc(tInteger(), method, oldType));
    return tStore(store.target, schema, store.task, store.messages);
}

rel[loc, Message] requireColumnSelect(Schema schema, StrLit feat) {
    rel[loc, Message] messages = {};
    if (feat.content notin schema) {
        messages += {<feat.src, error("Column \'<feat.content>\' not found in dataset.", feat.src,
            fixes=schemaSuggestionFixes(feat.src, feat.content, schema))>};
    }
    return messages;
}

rel[loc, Message] requireColumn(Schema schema, StrLit feat) {
    rel[loc, Message] messages = {};
    if (feat.content notin schema) {
        messages += {<feat.src, error("Column \'<feat.content>\' not part of selected features.", feat.src,
            fixes=schemaSuggestionFixes(feat.src, feat.content, schema))>};
    } else if (colInfoEnc(_, encOneHot(), _) := schema[feat.content]){
        messages += {<feat.src, error("Column \'<feat.content>\' is no longer present due to it being onehot encoded.", feat.src)>};
    }
    return messages;
}

TypeStore checkModelType(AST::Model model, TypeStore store) {
    Task task = taskOf(model.expr.algo);
    Schema schema = store.schema;
    str target = store.target;

    if (target notin schema) {
        return tStore(store.target, store.schema, task, store.messages);
    }
    ColumnInfo targetInfo = schema[target];

    switch (task) {
        case regression(): {
            if (targetInfo.ctype notin {tInteger(), tFloat()}) {
                store.messages += {<model.expr.algo.src, error("Algorithm is a regression model and requires a numeric target, but \'<target>\' is <targetInfo.ctype>.", model.expr.algo.src)>};
            }
        }
        case classification(): {
            bool ok = targetInfo.ctype == tCategorical()
                   || targetInfo.ctype == tBoolean()
                   || (targetInfo.ctype == tInteger());
            if (!ok) {
                store.messages += {<model.expr.algo.src, error(
                    "Algorithm is a classification model and requires a categorical/boolean/integer target " +
                    "but \'<target>\' is <targetInfo.ctype>.", model.expr.algo.src)>};
            }
        }
    }
    return tStore(store.target, store.schema, task, store.messages);
}

Task taskOf(algoLR()) = regression();

Task taskOf(algoRF()) = classification();

Task taskOf(algoLogReg()) = classification();

TypeStore checkMonitorTypes(AST::Monitor monitor, TypeStore store) {
    Schema schema = store.schema;
    for (DriftRule driftRule <- monitor.driftRules) {
        StrLit feat = driftRule.feature;
        rel[loc, Message] msgs = {};
        if (feat.content notin schema) {
            msgs += {<feat.src, error("Column \'<feat.content>\' not part of selected features.", feat.src,
                fixes=schemaSuggestionFixes(feat.src, feat.content, schema))>};
        }
        store.messages += msgs;
        if (size(msgs) != 0) {
            continue;
        }
        ColumnType oldType = store.schema[feat.content].ctype;
        if (colInfoEnc(_,_,oType) := store.schema[feat.content]) {
            oldType = oType;
        }
        switch(driftRule.dMethod) {
            case dmKS(): {
                if (oldType in {tInteger(), tCategorical(), tBoolean()}) {
                    store.messages += {<driftRule.dMethod.src, warning("The Kolmogorow-Smirnow method is normally only used for numerical features but <feat.content> is type <oldType>", driftRule.dMethod.src)>};
                }
            }
            case dmChiSquare(): {
                if (oldType in {tInteger(), tFloat()}) {
                    store.messages += {<driftRule.dMethod.src, warning("The Chi² method is normally only used for categorical features but <feat.content> is type <oldType>", driftRule.dMethod.src)>};
                }
            }
        }
    }
    return tStore(store.target, store.schema, store.task, store.messages);
}

list[CodeAction] prepareNoDeploymentFixes(loc src)
    = [
        action(title="Remove monitoring step", edits=[changed(src.top, [replace(src, "")])]),
        action(title="Add deployment step before monitoring", edits=[changed(src.top, [replace(insertionPointBefore(src), deploymentSnippet(src))])])
      ];

list[CodeAction] prepareNoConnectionFixes(loc src)
    = [
        action(title="Remove monitoring step", edits=[changed(src.top, [replace(src, "")])])
      ];

loc insertionPointBefore(loc src) = src[length=0][end=src.begin];

str deploymentSnippet(loc src) {
    str ind = intercalate("", [" " | _ <- [0..src.begin.column]]);
    str innerInd = ind + "    ";
    return "deployment (\n"
         + innerInd + "port = 8000\n"
         + ind + ")\n"
         + "\n"
         + ind;
}

loc removeEntryLoc(list[loc] \all, int idx) {
    loc item = \all[idx];
    if (size(\all) == 1) {
        return item;
    }
    if (idx == size(\all) - 1) {
        loc prev = \all[idx - 1];
        int startOffset = prev.offset + prev.length;
        int endOffset = item.offset + item.length;
        return item.top[offset=startOffset][length=endOffset-startOffset][begin=prev.end][end=item.end];
    }
    loc next = \all[idx + 1];
    int startOffset = item.offset;
    int endOffset = next.offset;
    return item.top[offset=startOffset][length=endOffset-startOffset][begin=item.begin][end=next.begin];
}

list[CodeAction] removeEntryFix(str title, list[loc] \all, int idx)
    = [action(title=title, edits=[changed(\all[idx].top, [replace(removeEntryLoc(\all, idx), "")])])];

loc paramNameLoc(loc paramSrc, str name) {
    int len = size(name);
    return paramSrc[length=len][end=<paramSrc.begin.line, paramSrc.begin.column + len>];
}

list[CodeAction] invalidParamNameFixes(loc paramSrc, str invalidName, ParamSignature sig) {
    loc target = paramNameLoc(paramSrc, invalidName);
    list[str] validNames = [n | n <- sig];
    return [
        action(title="Replace with \'<validName>\'", edits=[changed(target.top, [replace(target, validName)])])
        | str validName <- validNames
    ];
}

int levenshtein(str a, str b) {
    list[int] la = chars(a);
    list[int] lb = chars(b);
    int m = size(la);
    int n = size(lb);
    list[list[int]] d = [[0 | _ <- [0..n+1]] | _ <- [0..m+1]];

    for (int i <- [0..m+1]) {
        d[i][0] = i;
    }
    for (int j <- [0..n+1]) {
        d[0][j] = j;
    }

    for (int i <- [1..m+1]) {
        for (int j <- [1..n+1]) {
            int cost = (la[i-1] == lb[j-1]) ? 0 : 1;
            d[i][j] = min([
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            ]);
        }
    }
    return d[m][n];
}

list[CodeAction] schemaSuggestionFixes(loc src, str invalidName, Schema schema) {
    list[str] names = [n | n <- schema];
    lrel[int dist, str name] ranked = sort([<levenshtein(invalidName, n), n> | n <- names]);
    list[str] topNames = [name | <int _, str name> <- ranked][0..min(5, size(ranked))];
    return [
        action(title="Replace with \'<n>\'", edits=[changed(src.top, [replace(src, "\"<n>\"")])])
        | str n <- topNames
    ];
}

set[str] mlopsHoverService(Focus focus) {
    if (focus == []) return {};
    loc cursorLoc = focus[1].src;
    AST::Pipeline input = implode(#AST::Pipeline, focus[-1]);
    return computeDocs(input)[cursorLoc];
}

set[loc] mlopsDefinitionService(Focus focus) {
    if (focus == []) return {};
    loc cursorLoc = focus[1].src;
    AST::Pipeline input = implode(#AST::Pipeline, focus[-1]);
    <defs, uses> = computeDefsUses(input);
    return (uses o defs)[cursorLoc];
}

set[loc] mlopsReferencesService(Focus focus) {
    if (focus == []) return {};
    loc cursorLoc = focus[1].src;
    AST::Pipeline input = implode(#AST::Pipeline, focus[-1]);
    <defs, uses> = computeDefsUses(input);
    return (uses o defs)<1,0>[cursorLoc];
}

tuple[rel[str, loc] defs, rel[loc, str] uses] computeDefsUses(AST::Pipeline input) {
    AST::Steps steps = input.steps;
    rel[str, loc] defs = {};
    rel[loc, str] uses = {};

    if (size(steps.select) != 0) {
        defs = {<f.content, f.src> | f <- steps.select[0].features};
    }

    if (size(defs) != 0) {
        if (size(steps.trans) != 0) {
            uses += {<pt.feature.src, pt.feature.content> | pt <- steps.trans[0].transforms};
        }
        if (size(steps.monitor) != 0) {
            uses += {<dr.feature.src, dr.feature.content> | dr <- steps.monitor[0].driftRules};
        }
    }

    return <defs, uses>;
}

rel[loc, str] computeDocs(AST::Pipeline input) {
    AST::Steps steps = input.steps;
    rel[loc, str] docs = {};

    if (size(steps.select) != 0) {
        docs += {<f.src, "*selected feature* <f.content>"> | f <- steps.select[0].features};
    }
    if (size(steps.trans) != 0) {
        docs += {<pt.feature.src, "*transformed feature* <pt.feature.content> (<evalPrepTransform(pt).tr>)">
                  | pt <- steps.trans[0].transforms};
    }
    if (size(steps.monitor) != 0) {
        docs += {<dr.feature.src, "*monitored feature* <dr.feature.content>"> | dr <- steps.monitor[0].driftRules};
    }

    return docs;
}