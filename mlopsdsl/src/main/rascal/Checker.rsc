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
import Interpreter;
import Store;
import InlineFixes;

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
    return cStore("", false, null(), {});
}

MLOpsStore checkStepsSem(AST::Pipeline input, MLOpsStore store) {
    AST::Steps steps = input.steps;
    store = evalLoad(steps.load, store, -1, false, checker=true);
    if (size(steps.split) != 0) {
        store = evalSplit(steps.split[0], store, -1, checker=true);
    }
    if (size(steps.select) != 0) {
        store = evalSelect(steps.select[0], store, -1, checker=true);
    }
    if (size(steps.trans) != 0) {
        store = evalTrans(steps.trans[0], store, -1, checker=true);
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

MLOpsStore checkModelSem(AST::Model model, MLOpsStore store) {
    store = checkHyperparamsSem(model.expr, store);
    return cStore(store.targetVariable, store.dbConnection, taskOf(model.expr.algo), store.messages);
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
    return cStore(store.targetVariable, store.dbConnection, store.task, store.messages);
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
    return cStore(store.targetVariable, store.dbConnection, store.task, store.messages);
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

    return cStore(store.targetVariable, store.dbConnection, store.task, store.messages);
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
    return tStore(input.steps.load.target.content, schema, null());
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
    return tStore(load.target.content, store.schema, null(), messages=store.messages);
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
    return tStore(store.target, newSchema, null(), messages=store.messages);
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
    return tStore(store.target, schema, store.task, messages=store.messages);
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
    return tStore(store.target, schema, store.task, messages=store.messages);
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
        return tStore(store.target, store.schema, task, messages=store.messages);
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
    return tStore(store.target, store.schema, task, messages=store.messages);
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
    return tStore(store.target, store.schema, store.task, messages=store.messages);
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
        docs += {<pt.feature.src, "*transformed feature* <pt.feature.content> (<evalPrepTransform(pt)<1>>)">
                  | pt <- steps.trans[0].transforms};
    }
    if (size(steps.monitor) != 0) {
        docs += {<dr.feature.src, "*monitored feature* <dr.feature.content>"> | dr <- steps.monitor[0].driftRules};
    }

    return docs;
}