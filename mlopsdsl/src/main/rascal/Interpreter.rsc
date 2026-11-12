module Interpreter

import ParseTree;
import List;
import String;
import util::Maybe;
import ListRelation;
import Set;
import util::ShellExec;
import IO;
import util::IDEServices;
import Message;
import Exception;

import Syntax;
import AST;
import PythonBridge;
import CodeGen;
import Store;
import InlineFixes;

data RuntimeException 
    = reRunWithoutDatabase(str cause)
    | invalidTrainSize(str cause)
    | duplicateFieldSelection(str cause)
    | targetSelectedAsFeature(str cause)
    | targetSelectedForTransform(str cause)
    | duplicateTransform(str cause)
    | transformOrderViolation(str cause)
    | invalidHyperparameter(str cause)
    | targetSelectedForMonitoring(str cause)
    | invalidThreshold(str cause)
    | invalidFrequency(str cause)
    | invalidMinEffect(str cause)
    | evalThresholdNotReached(str cause)
    | noDeploymentDeclared(str cause)
    | noDBConnection(str cause)
    | duplicateMonitoredField(str cause)
    | invalidLatency(str cause)
    | pythonRuntimeError(str cause)
    | fileNotFound(str cause)
    | targetNotFound(str cause)
    | featureNotFound(str cause)
    | invalidErrorCode(str cause)
    | databaseConnectionFailed(str cause)
    | tableNotFound(str cause)
    | dropColumnFailed(str cause)
    ;

MLOpsStore evalPipeline(Syntax::Pipeline pipeline, bool reRun = false) {
    pipelineAST = implode(#AST::Pipeline, pipeline);
    return evalPipeline(pipelineAST, reRun);
}

MLOpsStore evalPipeline(pipeline(str name, Steps steps), bool reRun) {
    PID pid = startPythonWorker();
    MLOpsStore s = empty();
    try
        s = evalSteps(steps, iStore(name, uninitialized(), "", false, "", false, nothing()), pid, reRun);
    catch RuntimeException e: {
        stopPythonWorker(pid);
        throw e;
    }
    return s;
}

MLOpsStore evalSteps(steps(Load load, list[Split] split, list[Select] select, list[Trans] trans, Model model, list[Eval] eval, list[Deploy] deploy, list[Monitor] monitor), MLOpsStore s, PID pid, bool reRun) {
    s = evalLoad(load, s, pid, reRun);
    if (size(split) != 0) {
        s = evalSplit(split[0], s, pid);
    }
    if (size(select) != 0) {
        s = evalSelect(select[0], s, pid);
    }
    if (size(trans) != 0) {
        s = evalTrans(trans[0], s, pid);
    }
    s = evalModel(model, s, pid); 
    if (size(eval) != 0) {
        s = evalEval(eval[0], s, pid);
    }
    if (size(deploy) != 0) {
        s = evalDeploy(deploy[0], s);
    }
    if (size(monitor) != 0) {
        s = evalMonitor(monitor[0], s, pid);
    }
    stopPythonWorker(pid);
    if (deployed(_) := s.state) {
        finalizeDeployment(s);
    }
    return s;
}

MLOpsStore evalLoad(Load l:stepLoad(StrLit path, StrLit target, list[StrLit] dbURL), MLOpsStore s, PID pid, bool reRun, bool checker = false) {
    if (!checker && reRun && size(dbURL) == 0) {
        throw reRunWithoutDatabase("A re-run requires a database connection. Without one, run the pipeline normally with an updated CSV.");
    }
    str targetAsStr = evalStrLit(target);
    if (checker) {
        return cStore(targetAsStr, size(dbURL) != 0, s.task, s.messages);
    }
    str p = evalStrLit(path);
    str absolutePath = (l.src.parent + p).path;
    PythonCmd cmd = loadCmd("LOAD", absolutePath, targetAsStr, size(dbURL) != 0 ? evalStrLit(dbURL[0]) : "", reRun);
    PythonResponse res = sendJsonToPython(pid, cmd);
    reportResult(res, "LOAD", l.src);
    return iStore(s.name, dataLoaded(), targetAsStr, size(dbURL) != 0, s.trainedModelFilePath, s.monitored, s.latency);
}

str evalStrLit(strLit(str s)) = s;

MLOpsStore evalSplit(Split sp:stepSplit(real trainSize, list[int] randomState), MLOpsStore s, PID pid, bool checker = false) {
    if (trainSize <= 0) {
        if (!checker) {
            throw invalidTrainSize("Train size is below 0.0");
        }
        s.messages += {<sp.src, error("Train size is below 0.0",sp.src)>};
    }
    if (trainSize >= 1) {
        if (!checker) {
            throw invalidTrainSize("Train size exceeds 1.0");
        }
        s.messages += {<sp.src, error("Train size exceeds 1.0",sp.src)>};
    }
    if (checker) {
        return cStore(s.targetVariable, s.dbConnection, s.task, s.messages);
    }
    int randState = 42;
    if (size(randomState) != 0) {
        randState = randomState[0];
    }
    PythonCmd cmd = splitCmd("SPLIT", trainSize, randState);
    PythonResponse res = sendJsonToPython(pid, cmd);
    reportResult(res, "SPLIT", sp.src);
    return iStore(s.name, dataSplit(), s.targetVariable, s.dbConnection, s.trainedModelFilePath, s.monitored, s.latency);
}

MLOpsStore evalSelect(Select se:stepSelect(list[StrLit] features), MLOpsStore s, PID pid, bool checker=false) {
    set[str] seen = {};
    lrel[loc, str] featureLocs = [<f.src, evalStrLit(f)> | f <- features];
    for (int i <- [0..size(features)]) {
        tuple[loc src, str content] f = featureLocs[i];
        if (f.content == s.targetVariable) {
            if (!checker) {
                throw targetSelectedAsFeature("The target column cannot be a feature.");
            }
            s.messages += {<f.src, error("The target column cannot be a feature.", f.src,
                fixes=removeEntryFix("Remove target from selected features", featureLocs<0>, i))>};
        }
        if (f.content in seen) {
            if (!checker) {
                throw duplicateFieldSelection("Feature can only be selected once.");
            }
            s.messages += {<f.src, error("Feature \'<f.content>\' is selected more than once.", f.src,
                fixes=removeEntryFix("Remove duplicate feature", featureLocs<0>, i))>};    
        } else {
            seen += {f.content};
        }
    }
    if (checker) {
        return cStore(s.targetVariable, s.dbConnection, s.task, s.messages);
    }
    PythonCmd cmd = selectCmd("SELECT", featureLocs<1>);
    PythonResponse res = sendJsonToPython(pid, cmd);
    reportResult(res, "SELECT", se.src);
    return iStore(s.name, featuresSelected(), s.targetVariable, s.dbConnection, s.trainedModelFilePath, s.monitored, s.latency);
}

MLOpsStore evalTrans(Trans t:stepTrans(list[PrepTransform] transforms), MLOpsStore s, PID pid, bool checker=false) {
    map[str feat, lrel[str tr, loc src] entries] byFeature = ();
    lrel[loc, tuple[str tr, str feat, str strat]] transformLocs = [<pt.feature.src, evalPrepTransform(pt)> | pt <- transforms];
    for (int i <- [0..size(transforms)]) {
        tuple[str tr, str feat, str strat] ptrans = transformLocs[i]<1>;
        loc src = transformLocs[i]<0>;
        if (ptrans.feat == s.targetVariable) {
            if (!checker) {
                throw targetSelectedForTransform("The target column cannot be transformed");
            }
            s.messages += {<src, error("The target column cannot be transformed.", src,
                fixes=removeEntryFix("Remove transformation of target column", transformLocs<0>, i))>};
            continue;
        }
        byFeature[ptrans.feat] = (byFeature[ptrans.feat] ? []) + <ptrans.tr, src>;
    }
    for (str feat <- byFeature) {
        set[str] seenTrans = {};
        bool sawScale = false;
        bool sawEncode = false;
        for (<str tr, loc src> <- byFeature[feat]) {
            int idx = indexOf(transformLocs<0>, src);
            if (tr in seenTrans) {
                if (!checker) {
                    throw duplicateTransform("Feature must not be transformed multiple times by the same method.");
                }
                s.messages += {<src, error("Feature must not be transformed multiple times by the same method.", src,
                    fixes=removeEntryFix("Remove duplicate transformation", transformLocs<0>, idx))>};
            } else {
                seenTrans += {tr};
            }
            if (sawScale && (tr == "fillna" || tr == "encode")) {
                if (!checker) {
                    throw transformOrderViolation("Scaling must be the last transformation of a feature.");
                }
                s.messages += {<src, error("Scaling must be the last transformation of a feature.", src)>};
            }
            if (tr == "scale") {
                sawScale = true;
            }
            if (sawEncode && tr == "fillna") {
                if (!checker) {
                    throw transformOrderViolation("Missing values must be filled before encoding a feature.");
                }
                s.messages += {<src, error("Missing values must be filled before encoding a feature.", src)>};
            }
            if (tr == "encode") {
                sawEncode = true;
            }
        }
    }
    if (checker) {
        return cStore(s.targetVariable, s.dbConnection, s.task, s.messages);
    }
    for (tuple[str tr, str feat, str strat] ptrans <- transformLocs<1>) {
        PythonCmd cmd = transformCmd("TRANSFORM", ptrans.tr, ptrans.feat, ptrans.strat);
        PythonResponse res = sendJsonToPython(pid, cmd);
        reportResult(res, "TRANSFORM", t.src);
    }
    return iStore(s.name, transformed(), s.targetVariable, s.dbConnection, s.trainedModelFilePath, s.monitored, s.latency);
}

tuple[str tr, str feat, str strat] evalPrepTransform(prepFill(StrLit feature, FillStrategy strategy)) {
    str feat = evalStrLit(feature);
    str strat = evalFillStrategy(strategy);
    return <"fillna", feat, strat>;
}

str evalFillStrategy(fillMean()) = "mean";

str evalFillStrategy(fillMedian()) = "median";

str evalFillStrategy(fillMode()) = "most_frequent"; 

tuple[str tr, str feat, str strat] evalPrepTransform(prepEncode(StrLit feature, EncodingMethod method)) {
    str feat = evalStrLit(feature);
    str meth = evalEncodingMethod(method);
    return <"encode", feat, meth>;
}

str evalEncodingMethod(encOneHot()) = "onehot";

str evalEncodingMethod(encLabel()) = "label";

tuple[str tr, str feat, str strat] evalPrepTransform(prepScale(StrLit feature, ScaleMethod method)) {
    str feat = evalStrLit(feature);
    str meth = evalScaleMethod(method);
    return <"scale", feat, meth>;
}

str evalScaleMethod(scaleMinMax()) = "minmax";

str evalScaleMethod(scaleStd()) = "std";

MLOpsStore evalModel(Model m:stepModel(modelTrain(Algorithm algo, StrLit path, list[Param] hyperParams)), MLOpsStore s, PID pid, bool checker=false) {
    map[str, str] params = ();
    ParamSignature sig = signatureOf(algo);
    for (Param hyperParam <- hyperParams) {
        tuple[str name, str val] param = evalParam(hyperParam);
        if (hyperParam.name notin sig) {
            if (!checker) {
                throw invalidHyperparameter("\'<param.name>\' is not a valid hyperparameter for <algo.name>.");
            } 
            s.messages += {<hyperParam.src, error("\'<param.name>\' is not a valid hyperparameter for <algo.name>.", hyperParam.src,
                fixes=invalidParamNameFixes(hyperParam.src, param.name, sig))>};
        }
        params[param.name] = param.val;
    }
    if (checker) {
        return cStore(s.targetVariable, s.dbConnection, taskOf(algo), s.messages);
    }
    str p = evalStrLit(path);
    str modelDir = (m.src.parent + p).path + "/<s.name>";
    str algo_as_string = evalAlgo(algo);
    PythonCmd cmd = trainCmd("TRAIN", algo_as_string, params, modelDir);
    PythonResponse res = sendJsonToPython(pid, cmd);
    reportResult(res, "TRAIN", m.src);
    return iStore(s.name, modelTrained(), s.targetVariable, s.dbConnection, res.modelFilePath, s.monitored, s.latency);
}

str evalAlgo(algoLR()) = "LinReg";

str evalAlgo(algoRF()) = "RandomForest";

str evalAlgo(algoLogReg()) = "LogReg";

tuple[str, str] evalParam(hp(str name, Lit val)) {
    str val_as_string = evalLit(val);
    return <name,val_as_string>;
}

str evalLit(intLit(int i)) = "<i>";

str evalLit(floatLit(real f)) = "<f>";

str evalLit(strLit(StrLit s)) = evalStrLit(s);

str evalLit(boolLit(bool b)) = "<b>";

MLOpsStore evalEval(Eval e:stepEval(set[EvalRule] evalRules), MLOpsStore s, PID pid, bool checker = false) {
    for (EvalRule evalRule <- evalRules) {
        <metric_as_str, threshold, higherIsBetter> = evalEvalRule(evalRule);
        if (higherIsBetter && (threshold > 1.0 || threshold < 0)) {
            if (!checker) {
                throw invalidThreshold("Threshold for \'<metric_as_str>\' must be between 0 and 1");
            }
            s.messages += {<evalRule.src, error("Threshold for \'<metric_as_str>\' must be between 0 and 1", evalRule.src)>};
        } else if (!higherIsBetter && (threshold < 0)) {
            if (!checker) {
                throw invalidThreshold("Threshold for \'<metric_as_str>\' must be at least 0");
            }
            s.messages += {<evalRule.src, error("Threshold for \'<metric_as_str>\' must be at least 0", evalRule.src)>};
        }
    }
    if (checker) {
        return cStore(s.targetVariable, s.dbConnection, s.task, s.messages);
    }
    for (EvalRule evalRule <- evalRules) {
        <metric_as_str, threshold, higherIsBetter> = evalEvalRule(evalRule);
        PythonCmd cmd = evalCmd("EVAL", metric_as_str);
        PythonResponse res = sendJsonToPython(pid, cmd);
        reportResult(res, "EVAL", e.src);
        real val = res.evalResults[metric_as_str];
        bool reached = higherIsBetter ? val >= threshold : val <= threshold;
        if (!reached) {
            throw evalThresholdNotReached("Pipeline stopped before potential deployment since evaluation thresholds were not met.");
        }
    }
    return iStore(s.name, modelEvaluated(), s.targetVariable, s.dbConnection, s.trainedModelFilePath, s.monitored, s.latency);
}

tuple[str, real, bool] evalEvalRule(evalCRule(CMetric metric, real threshold))
    = <evalCMetric(metric), threshold, true>;

tuple[str, real, bool] evalEvalRule(evalRRule(RMetric metric, real threshold))
    = <evalRMetric(metric), threshold, false>;

str evalCMetric(mAccuracy()) = "acc";

str evalCMetric(mPrecision()) = "pre";

str evalCMetric(mRecall()) = "rec";

str evalCMetric(mF1()) = "f1";

str evalRMetric(mMSE()) = "mse";

str evalRMetric(mRMSE()) = "rmse";

MLOpsStore evalDeploy(Deploy d:stepDeploy(int port), MLOpsStore s) {
    showMessage(info("[DEPLOY] Gathered deployment information", d.src));
    return iStore(s.name, deployed(port), s.targetVariable, s.dbConnection, s.trainedModelFilePath, s.monitored, s.latency);
}

MLOpsStore evalMonitor(Monitor mon, MLOpsStore s, PID pid, bool checker = false) {
    if (!s.dbConnection) {
        if (!checker) {
            throw noDBConnection("Cannot store user data for monitoring without connected database.");
        }
        s.messages += {<mon.src, error("Cannot store user data for monitoring without connected database.", mon.src,
            fixes=prepareNoConnectionFixes(mon.src))>};
    }
    list[AST::DriftRule] driftRules = driftRulesOf(mon);
    lrel[loc, tuple[str method, str feat, int win, int freq, real minEffect, real threshold]] ruleLocs = [<dr.src, evalDriftRule(dr)> | dr <- driftRules];
    set[str] seenFeatures = {};
    for (int i <- [0..size(driftRules)]) {
        AST::DriftRule driftRule = driftRules[i];
        tuple[str method, str feat, int win, int freq, real minEffect, real threshold] dRule = ruleLocs[i]<1>;
        if (dRule.feat == s.targetVariable) {
            if (!checker) {
                throw targetSelectedForMonitoring("The target can not be selected as a monitored feature.");
            }
            s.messages += {<driftRule.src, error("The target can not be selected as a monitored feature.", driftRule.src,
                fixes=removeEntryFix("Remove monitoring of target column", ruleLocs<0>, i))>};
        }
        if (dRule.feat in seenFeatures) {
            if (!checker) {
                throw duplicateMonitoredField("Feature \'<dRule.feat>\' is monitored more than once.");
            }
            s.messages += {<driftRule.src, error("Feature \'<dRule.feat>\' is monitored more than once.", driftRule.src,
                fixes=removeEntryFix("Remove duplicate monitoring of feature", ruleLocs<0>, i))>};
        } else {
            seenFeatures += {dRule.feat};
        }
        if (dRule.win < 500) {
            if (!checker) {
                showMessage(warning("[MONITOR] Monitoring was set up with small window. Recommendation is 500 or higher.", mon.src));
            } else {
                s.messages += {<driftRule.src, warning("The window for drift calculation is small. Recommended is 500 or higher.", driftRule.src)>};
            }
        }
        if (dRule.threshold <= 0) {
            if (!checker) {
                throw invalidThreshold("The threshold cannot be negative or 0.");
            }
            s.messages += {<driftRule.src, error("The threshold cannot be negative or 0.", driftRule.src)>};
        }
        if (dRule.freq <= 0) {
            if (!checker) {
                throw invalidFrequency("The drift calculation frequency cannot be negative or 0");
            }
            s.messages += {<driftRule.src, error("The drift calculation frequency cannot be negative or 0.", driftRule.src)>};
        }
        if (dRule.method == "KS" && dRule.minEffect >= 1) {
            if (!checker) {
                throw invalidMinEffect("The minimal effect bound cannot be greater or equal to 1"); 
            }
            s.messages += {<driftRule.src, error("The minimum effect bound cannot be greater or equal to 1", driftRule.src)>};
        }
        if (dRule.minEffect < 0) {
            if (!checker) {
                throw invalidMinEffect("The minimal effect bound cannot be smaller than 0");
            }
            s.messages += {<driftRule.src, error("The minimum effect bound cannot be smaller than 0", driftRule.src)>};
        }
    }
    Maybe[int] ms = nothing();
    list[AST::LatencyRule] latencyRule = latencyRuleOf(mon);
    if (size(latencyRule) != 0) {
        AST::LatencyRule lRule = latencyRule[0];
        int parsedMs = evalLatencyRule(latencyRule[0]);
        if (parsedMs <= 0) {
            if (!checker) {
                throw invalidLatency("Latency cannot be smaller or equal to 0.");
            }
            s.messages += {<lRule.src, error("Latency cannot be smaller or equal to 0.", lRule.src)>};
        } else {
            ms = just(parsedMs);
        }
    }
    if (checker) {
        return cStore(s.targetVariable, s.dbConnection, s.task, s.messages);
    }
    if (size(driftRules) != 0) {
        PythonCmd cmd = monitorCmd("MONITOR", ruleLocs<1><0>, ruleLocs<1><1>, ruleLocs<1><2>, ruleLocs<1><3>, ruleLocs<1><4>, ruleLocs<1><5>);
        PythonResponse res = sendJsonToPython(pid, cmd);
        reportResult(res, "MONITOR", mon.src);
    }
    return iStore(s.name, s.state, s.targetVariable, s.dbConnection, s.trainedModelFilePath, size(ruleLocs<1><1>) != 0, ms);
}

list[DriftRule] driftRulesOf(stepMonitor(list[DriftRule] drs, _)) = drs;
list[DriftRule] driftRulesOf(stepMonitorLatency(_)) = [];

list[LatencyRule] latencyRuleOf(stepMonitor(_, list[LatencyRule] lr)) = lr;
list[LatencyRule] latencyRuleOf(stepMonitorLatency(LatencyRule l)) = [l];

tuple[str method, str feat, int win, int freq, real minEffect, real threshold] evalDriftRule(ruleDrift(DriftMethod dMethod, StrLit feature, int window, int freq, real minEffect, real threshold)) {
    str feat = evalStrLit(feature);
    str method = evalDriftMethod(dMethod);
    return <method, feat, window, freq, minEffect, threshold>;
}

int evalLatencyRule(ruleLatency(int ms)) {
    return ms;
}

str evalDriftMethod(dmKS()) = "KS";

str evalDriftMethod(dmChiSquare()) = "ChiSquare";

void finalizeDeployment(MLOpsStore s) {
    loc filePath = |file:///| + s.trainedModelFilePath;
    str fastAPIApp = genFastAPIApp(filePath.file, s.monitored, s.latency);
    str dockerfile = genDockerfile(filePath.file, s.state.port);
    str requirements = genRequirementsTXT(s.monitored);
    str compose = genDockerCompose(s.state.port, s.monitored);
    writeFile(filePath.parent + "app.py", fastAPIApp);
    writeFile(filePath.parent + "Dockerfile", dockerfile);
    writeFile(filePath.parent + "requirements.txt", requirements);
    writeFile(filePath.parent + "docker-compose.yml", compose);
}

void reportResult(PythonResponse res, str step, loc l) {
    if (res.status == "ERROR") {
        if (res.code == 0) {
            throw pythonRuntimeError(res.message);
        } else {
            throwErrorWithCode(res.code, res.message);
        }
    }
    showMessage(info("[<step>] <res.message>", l));
}

void throwErrorWithCode(int code, str message) {
    switch(code) {
        case 1:
            throw fileNotFound(message);
        case 2:
            throw targetNotFound(message);
        case 3:
            throw featureNotFound(message);
        case 4:
            throw databaseConnectionFailed(message);
        case 5:
            throw tableNotFound(message);
        case 6:
            throw dropColumnFailed(message);
        default:
            throw invalidErrorCode("This error code is not identified with a specified exception.");
    }
}