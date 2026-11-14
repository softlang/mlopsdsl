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
import InlineFixes;

data TypeException
    = untypedColumn(str cause)
    | incompatibleTransform(str cause)
    | incompatibleTarget(str cause)
    | incompatibleMetric(str cause)
    | hyperparamTypeMismatch(str cause)
    | schemaInferenceFailed(str cause)
    | databaseConnectionFailed(str cause)
    | incompatibleMethod(str cause)
    ;

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
        checkEval(eval[0], store.task);
    }
    if (size(monitor) != 0) {
        checkMonitor(monitor[0], store);
    }
    return store;
}

TypeStore checkLoad(Load l:stepLoad(StrLit path, StrLit target, list[StrLit] dbURL), bool checker = false) {
    str p = path.content;
    loc baseDir = l.src.parent;
    loc scriptPath = getPath("schema_infer.py");
    loc csvPath = baseDir + p;
    tuple[str output, int exitCode] result = execWithCode(PythonBridge::getPythonExecutable(), args=size(dbURL) != 0 ? [scriptPath, csvPath.top, dbURL[0].content] : [scriptPath, csvPath.top]);
    SchemaJson resp = parseJSON(#SchemaJson, trim(result<0>));
    rel[loc, Message] messages = {};
    if (resp.status == "FILE_ERROR") {
        if (!checker) {
            throw schemaInferenceFailed(resp.message);
        }
        messages += {<path.src, error(resp.message, path.src)>};
        return tStore(target.content, (), null(), messages=messages);
    }
    if (resp.status == "DB_ERROR") {
        if (!checker) {
            throw databaseConnectionFailed(resp.message);
        } else {
            messages += {<dbURL[0].src, error(resp.message, dbURL[0].src)>};
        }
    }
    Schema schema = buildSchema(resp);
    if (target.content notin schema) {
        if (!checker) {
            throw untypedColumn("No type for target column \'<target.content>\' in schema.");
        }
        messages += {<target.src, error("No type for target column \'<target.content>\' in schema.", target.src,
            fixes=schemaSuggestionFixes(target.src, target.content, schema))>};
    }
    return tStore(target.content, schema, null(), messages=messages);
}

Schema buildSchema(SchemaJson resp) {
    Schema schema = ();
    for (str col <- resp.columns) {
        ColumnInfoJson entry = resp.columns[col];
        schema[col] = colInfo(parseColumnType(entry.\type));
    }
    return schema;
}

TypeStore checkSelect(stepSelect(list[StrLit] features), TypeStore store, bool checker = false) {
    Schema schema = store.schema;
    Schema newSchema = ();
    for (StrLit f <- features) {
        rel[loc, Message] featMessages = requireColumn(schema, f, checker);
        store.messages += featMessages;
        if (size(featMessages) != 0) {
            continue;
        }
        newSchema[f.content] = schema[f.content];
    }
    if (store.target in schema) {
        newSchema[store.target] = schema[store.target];
    }
    return tStore(store.target, newSchema, null(), messages=store.messages);
}

TypeStore checkTrans(stepTrans(list[PrepTransform] transforms), TypeStore store, bool checker=false) {
    for (PrepTransform pt <- transforms) {
        store = checkAndApplyTransform(pt, store, checker);
    }
    return store;
}

TypeStore checkAndApplyTransform(prepFill(StrLit feature, FillStrategy strategy), TypeStore store, bool checker) {
    Schema schema = store.schema;
    str feat = feature.content;
    rel[loc, Message] msgs = requireColumn(schema, feature, checker);
    store.messages += msgs;
    if (size(msgs) != 0) {
        return store;
    }
    if ((strategy is fillMean || strategy is fillMedian) && (schema[feat].ctype notin {tInteger(), tFloat()})) {
        if (!checker) {
            throw incompatibleTransform("fillna(mean/median) requires a numeric column, but \'<feat>\' is <schema[feat].ctype>.");
        }
        store.messages += {<feature.src, error("fillna(mean/median) requires a numeric column, but \'<feat>\' is <schema[feat].ctype>.", feature.src)>};
    }
    return store;
}

TypeStore checkAndApplyTransform(prepScale(StrLit feature, ScaleMethod _), TypeStore store, bool checker) {
    Schema schema = store.schema;
    str feat = feature.content;
    rel[loc, Message] msgs = requireColumn(store.schema, feature, checker);
    store.messages += msgs;
    if (size(msgs) != 0) {
        return store;
    }
    switch(schema[feat]) {
        case colInfo(ctype): {
            if (ctype notin {tInteger(), tFloat()}) {
                if (!checker) {
                    throw incompatibleTransform("scale(...) requires a numeric column, but \'<feat>\' is <schema[feat].ctype>."); 
                }
                store.messages += {<feature.src, error("scale(...) requires a numeric column, but \'<feat>\' is <schema[feat].ctype>.", feature.src)>};
            } else {
                schema = schema + (feat: colInfo(tFloat()));
            }
        }
        case colInfoEnc(tInteger(), method, oldType): {
            schema = schema + (feat: colInfoEnc(tFloat(), method, oldType));
        }
    }
    return tStore(store.target, schema, store.task, messages=store.messages);
}

TypeStore checkAndApplyTransform(prepEncode(StrLit feature, EncodingMethod method), TypeStore store, bool checker) {
    Schema schema = store.schema;
    str feat = feature.content;
    rel[loc, Message] msgs = requireColumn(schema, feature, checker);
    store.messages += msgs;
    if (size(msgs) != 0) {
        return store;
    }
    ColumnType oldType = schema[feat].ctype;
    schema = schema + (feat: colInfoEnc(tInteger(), method, oldType));
    return tStore(store.target, schema, store.task, messages=store.messages);
}

private rel[loc, Message] requireColumn(Schema schema, StrLit feat, bool checker) {
    rel[loc, Message] messages = {};
    if (feat.content notin schema) {
        if (!checker) {
            throw untypedColumn("No type for column \'<feat.content>\' in schema.");
        }
        messages += {<feat.src, error("No type for \'<feat.content>\' in selected features.", feat.src,
            fixes=schemaSuggestionFixes(feat.src, feat.content, schema))>};
    } else if (colInfoEnc(_, encOneHot(), _) := schema[feat.content]) {
        if (!checker) {
            throw untypedColumn("Column \'<feat.content>\' is no longer present due to it being onehot encoded.");
        }
        messages += {<feat.src, error("Column \'<feat.content>\' is no longer present due to it being onehot encoded.", feat.src)>};
    }
    return messages;
}

TypeStore checkModel(stepModel(ModelExpr expr), TypeStore store, bool checker = false) {
    Task task = taskOf(expr.algo);
    Schema schema = store.schema;
    str target = store.target;
    if (target notin schema) {
        return tStore(store.target, store.schema, task, messages=store.messages);
    }
    ColumnInfo targetInfo = schema[target];
    switch (task) {
        case regression(): {
            if (targetInfo.ctype notin {tInteger(), tFloat()}) {
                if (!checker) {
                    throw incompatibleTarget("Algorithm is a regression model and requires a numeric target, but \'<target>\' is <targetInfo.ctype>.");
                }
                store.messages += {<expr.algo.src, error("Algorithm is a regression model and requires a numeric target, but \'<target>\' is <targetInfo.ctype>.", expr.algo.src)>};
            }
        }
        case classification(): {
            bool ok = targetInfo.ctype == tCategorical()
                   || targetInfo.ctype == tBoolean()
                   || (targetInfo.ctype == tInteger());
            if (!ok) {
                if (!checker) {
                    throw incompatibleTarget(
                        "Algorithm is a classification model and requires a categorical/boolean/integer target " +
                        "but \'<target>\' is <targetInfo.ctype>."
                    );
                }
                store.messages += {<expr.algo.src, error(
                    "Algorithm is a classification model and requires a categorical/boolean/integer target " +
                    "but \'<target>\' is <targetInfo.ctype>.", expr.algo.src)>};
            }
        }
    }
    if (!checker) {
        checkHyperparams(expr);
    }
    return tStore(store.target, store.schema, task, messages=store.messages);
}

MLOpsStore checkHyperparams(modelTrain(Algorithm algo, StrLit _, list[Param] hyperParams), MLOpsStore s = noStore()) {
    ParamSignature sig = signatureOf(algo);
    for (hp(str name, Lit val) <- hyperParams) {
        if (name notin sig) {
            continue;
        }
        if (!paramTypeMatches(val, sig[name])) {
            if (noStore() := s) {
                throw hyperparamTypeMismatch("\'<name>\' has an incompatible value for <algo>: expected <sig[name]>.");
            }
            s.messages += {<val.src, error("\'<name>\' has an incompatible value: expected <sig[name]>.", val.src)>};
        }
    }
    if (noStore() := s) {
        return s;
    }
    return cStore(s.targetVariable, s.dbConnection, s.task, s.messages);
}

MLOpsStore checkEval(stepEval(set[EvalRule] evalRules), Task task, MLOpsStore s = noStore()) {
    for (EvalRule evalRule <- evalRules) {
        if (!ruleMatchesTask(evalRule, task)) {
            if (noStore() := s) {
                throw incompatibleMetric("Metric <metricName(evalRule)> is not compatible with a <task> model.");
            }
            s.messages += {<evalRule.src, error("Metric <metricName(evalRule)> is not compatible with a <task> model.", evalRule.src)>};
        }
    }
    if (noStore() := s) {
        return s;
    }
    return cStore(s.targetVariable, s.dbConnection, s.task, s.messages);
}

bool ruleMatchesTask(evalCRule(_, _), classification()) = true;

bool ruleMatchesTask(evalRRule(_, _), regression()) = true;

default bool ruleMatchesTask(EvalRule _, Task _) = false;

str metricName(evalCRule(CMetric m, _)) = m.name;

str metricName(evalRRule(RMetric m, _)) = m.name;

TypeStore checkMonitor(Monitor mon, TypeStore store, bool checker = false) {
    Schema schema = store.schema;
    for (DriftRule driftRule <- driftRulesOf(mon)) {
        StrLit feat = driftRule.feature;
        rel[loc, Message] msgs = {};
        if (feat.content notin schema) {
            if (!checker) {
                throw untypedColumn("Column \'<feat>\' not found.");
            }
            msgs += {<feat.src, error("Column \'<feat.content>\' has not type.", feat.src,
                fixes=schemaSuggestionFixes(feat.src, feat.content, schema))>};
        }
        store.messages += msgs;
        if (size(msgs) != 0) {
            continue;
        }
        ColumnType oldType = schema[feat.content].ctype;
        if (colInfoEnc(_,_,oType) := schema[feat.content]) {
            oldType = oType;
        }
        switch(driftRule.dMethod) {
            case dmKS(): {
                if (oldType in {tInteger(), tCategorical(), tBoolean()}) {
                    if (!checker) {
                        showMessage(warning("The Kolmogorow-Smirnow method is normally only used for numerical features but <feat> is type <oldType>", mon.src));
                    } else {
                        store.messages += {<driftRule.dMethod.src, warning("The Kolmogorow-Smirnow method is normally only used for numerical features but <feat.content> is type <oldType>", driftRule.dMethod.src)>};
                    }
                }
            }
            case dmChiSquare(): {
                if (oldType in {tInteger(), tFloat()}) {
                    if (!checker) {
                        showMessage(warning("The Chi² method is normally only used for categorical features but <feat> is type <oldType>", mon.src));
                    } else {
                        store.messages += {<driftRule.dMethod.src, warning("The Chi² method is normally only used for categorical features but <feat.content> is type <oldType>", driftRule.dMethod.src)>};
                    }
                }
            }
        }
    }
    return store;
}

list[DriftRule] driftRulesOf(stepMonitor(list[DriftRule] drs, _)) = drs;
list[DriftRule] driftRulesOf(stepMonitorLatency(_)) = [];