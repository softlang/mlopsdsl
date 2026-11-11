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
import util::Maybe;
import Exception;

import Syntax;
import AST;
import PythonBridge;
import CodeGen;

data PipelineState
    = uninitialized()
    | dataLoaded()
    | dataSplit()
    | featuresSelected()
    | transformed()
    | modelTrained()
    | modelEvaluated()
    | deployed(int port);

data MLOpsStore = store(str name, PipelineState state, str targetVariable, bool dbConnection, str trainedModelFilePath, bool monitored, Maybe[int] latency) | empty();

data RuntimeException 
    = reRunWithoutDatabase(str cause)
    | invalidTrainSize(str cause)
    | duplicateFieldSelection(str cause)
    | targetSelectedAsFeature(str cause)
    | targetSelectedForTransform(str cause)
    | duplicateTransform(str cause)
    | transformOrderViolation(str cause)
    | targetSelectedForMonitoring(str cause)
    | windowTooSmall(str cause)
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
        s = evalSteps(steps, store(name, uninitialized(), "", false, "", false, nothing()), pid, reRun);
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

MLOpsStore evalLoad(Load l:stepLoad(StrLit path, StrLit target, list[StrLit] dbURL), MLOpsStore s, PID pid, bool reRun) {
    if (reRun && size(dbURL) == 0) {
        throw reRunWithoutDatabase("A re-run requires a database connection. Without one, run the pipeline normally with an updated CSV.");
    }
    str p = evalStrLit(path);
    str absolutePath = (l.src.parent + p).path;
    str targetAsStr = evalStrLit(target);
    PythonCmd cmd = loadCmd("LOAD", absolutePath, targetAsStr, size(dbURL) != 0 ? evalStrLit(dbURL[0]) : "", reRun);
    PythonResponse res = sendJsonToPython(pid, cmd);
    reportResult(res, "LOAD", l.src);
    return store(s.name, dataLoaded(), targetAsStr, size(dbURL) != 0, s.trainedModelFilePath, s.monitored, s.latency);
}

str evalStrLit(strLit(str s)) = s;

MLOpsStore evalSplit(Split sp:stepSplit(real trainSize, list[int] randomState), MLOpsStore s, PID pid) {
    if (trainSize <= 0) {
        throw invalidTrainSize("Train size is below 0.0");
    }
    if (trainSize >= 1) {
        throw invalidTrainSize("Train size exceeds 1.0");
    }
    int randState = 42;
    if (size(randomState) != 0) {
        randState = randomState[0];
    }
    PythonCmd cmd = splitCmd("SPLIT", trainSize, randState);
    PythonResponse res = sendJsonToPython(pid, cmd);
    reportResult(res, "SPLIT", sp.src);
    return store(s.name, dataSplit(), s.targetVariable, s.dbConnection, s.trainedModelFilePath, s.monitored, s.latency);
}

MLOpsStore evalSelect(Select se:stepSelect(list[StrLit] features), MLOpsStore s, PID pid) {
    if (size(features) != size(dup(features))) {
        throw duplicateFieldSelection("Feature can only be selected once.");
    }
    list[str] strFeatures = [];
    for (StrLit feature <- features) {
        strFeatures += evalStrLit(feature);
    }
    if (s.targetVariable in strFeatures) {
        throw targetSelectedAsFeature("The target column cannot be a feature.");
    }
    PythonCmd cmd = selectCmd("SELECT", strFeatures);
    PythonResponse res = sendJsonToPython(pid, cmd);
    reportResult(res, "SELECT", se.src);
    return store(s.name, featuresSelected(), s.targetVariable, s.dbConnection, s.trainedModelFilePath, s.monitored, s.latency);
}

MLOpsStore evalTrans(Trans t:stepTrans(list[PrepTransform] transforms), MLOpsStore s, PID pid) {
    lrel[str tr, str feat, str param] trans = [];
    for (PrepTransform pt <- transforms) {
        tuple[str tr, str feat, str param] ptrans = evalPrepTransform(pt);
        if (ptrans.feat == s.targetVariable) {
            throw targetSelectedForTransform("The target column cannot be transformed");
        }
        trans += ptrans;
    }
    list[list[str]] transforms_per_features = groupDomainByRange(trans<tr,feat>);
    for (list[str] transforms_per_feature <- transforms_per_features) {
        if (size(toSet(transforms_per_feature)) != size(transforms_per_feature)) {
            throw duplicateTransform("Feature must not be transformed multiple times by the same method.");
        }
        if ([*_, "scale", *_, "fillna", *_] := transforms_per_feature || [*_, "scale", *_, "encode", *_] := transforms_per_feature) {
            throw transformOrderViolation("Scaling must be the last transformation of a feature.");
        }
        if ([*_, "encode", *_, "fillna", *_] := transforms_per_feature) {
            throw transformOrderViolation("Missing values must be filled before encoding a feature.");
        }
    }
    for (tuple[str tr, str feat, str param] ptrans <- trans) {
        PythonCmd cmd = transformCmd("TRANSFORM", ptrans.tr, ptrans.feat, ptrans.param);
        PythonResponse res = sendJsonToPython(pid, cmd);
        reportResult(res, "TRANSFORM", t.src);
    }
    return store(s.name, transformed(), s.targetVariable, s.dbConnection, s.trainedModelFilePath, s.monitored, s.latency);
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

MLOpsStore evalModel(Model m:stepModel(modelTrain(Algorithm algo, StrLit path, list[Param] hyperParams)), MLOpsStore s, PID pid) {
    str algo_as_string = evalAlgo(algo);
    map[str, str] params = (); 
    for (Param hyperParam <- hyperParams) {
        tuple[str name, str val] param = evalParam(hyperParam);
        params[param.name] = param.val;
    }
    str p = evalStrLit(path);
    str modelDir = (m.src.parent + p).path + "/<s.name>";
    PythonCmd cmd = trainCmd("TRAIN", algo_as_string, params, modelDir);
    PythonResponse res = sendJsonToPython(pid, cmd);
    reportResult(res, "TRAIN", m.src);
    return store(s.name, modelTrained(), s.targetVariable, s.dbConnection, res.modelFilePath, s.monitored, s.latency);
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

MLOpsStore evalEval(Eval e:stepEval(set[EvalRule] evalRules), MLOpsStore s, PID pid) {
    for (EvalRule evalRule <- evalRules) {
        <metric_as_str, threshold, higherIsBetter> = evalEvalRule(evalRule);

        if (higherIsBetter && (threshold > 1.0 || threshold < 0)) {
            throw invalidThreshold("Threshold for \'<metric_as_str>\' must be between 0 and 1");
        }

        PythonCmd cmd = evalCmd("EVAL", metric_as_str);
        PythonResponse res = sendJsonToPython(pid, cmd);
        real val = res.evalResults[metric_as_str];

        bool reached = higherIsBetter ? val >= threshold : val <= threshold;
        if (!reached) {
            throw evalThresholdNotReached("Pipeline stopped before potential deployment since evaluation thresholds were not met.");
        }
        reportResult(res, "EVAL", e.src);
    }
    return store(s.name, modelEvaluated(), s.targetVariable, s.dbConnection, s.trainedModelFilePath, s.monitored, s.latency);
}

tuple[str, real, bool] evalEvalRule(evalCRule(CMetric metric, real threshold))
    = <evalCMetric(metric), threshold, true>;

tuple[str, real, bool] evalEvalRule(evalRRule(RMetric metric, real threshold))
    = <evalRMetric(metric), threshold, false>;

str evalCMetric(mAccuracy())  = "acc";

str evalCMetric(mPrecision()) = "pre";

str evalCMetric(mRecall())    = "rec";

str evalCMetric(mF1())        = "f1";

str evalRMetric(mMSE())  = "mse";

str evalRMetric(mRMSE()) = "rmse";

MLOpsStore evalDeploy(Deploy d:stepDeploy(int port), MLOpsStore s) {
    showMessage(info("[DEPLOY] Gathered deployement information", d.src));
    return store(s.name, deployed(port), s.targetVariable, s.dbConnection, s.trainedModelFilePath, s.monitored, s.latency);
}

MLOpsStore evalMonitor(Monitor mon, MLOpsStore s, PID pid) {
    list[AST::DriftRule] driftRules = driftRulesOf(mon);
    list[AST::LatencyRule] latencyRule = latencyRuleOf(mon);
    if (!(deployed(_) := s.state)) {
        throw noDeploymentDeclared("User inputs cannot be monitored without deployed model.");
    }
    if (!s.dbConnection) {
        throw noDBConnection("Cannot store user data for monitoring without connected database.");
    }
    list[str] methods = [];
    list[str] monitored = [];
    list[int] windows = [];
    list[int] freqs = [];
    list[real] minEffects = [];
    list[real] thresholds = [];
    set[str] seenFeatures = {};
    for (DriftRule driftRule <- driftRules) {
        tuple[str meth, str feat, int win, int freq, real minEffect, real threshold] dRule = evalDriftRule(driftRule);
        if (dRule.feat == s.targetVariable) {
            throw targetSelectedForMonitoring("The target can not be selected as a monitored feature.");
        }
        if (dRule.feat in seenFeatures) {
            throw duplicateMonitoredField("Feature \'<dRule.feat>\' is monitored more than once.");
        }
        if (dRule.win < 500) {
            showMessage(warning("[MONITOR] Monitoring was set up with small window. Recommendation is 500 or higher.", mon.src));
        }
        if (dRule.threshold <= 0) {
            throw invalidThreshold("The threshold cannot be negative or 0.");
        }
        if (dRule.freq <= 0) {
            throw invalidFrequency("The drift calculation frequency cannot be negative or 0");
        }
        if (dRule.meth == "KS" && dRule.minEffect >= 1) {
            throw invalidMinEffect("The minimal effect bound cannot be greater or equal to 1");
        }
        if (dRule.minEffect < 0) {
            throw invalidMinEffect("The minimal effect bound cannot be smmaller than 0");
        }
        methods += dRule.meth;
        monitored += dRule.feat;
        windows += dRule.win;
        freqs += dRule.freq;
        minEffects += dRule.minEffect;
        thresholds += dRule.threshold;
        seenFeatures += {dRule.feat};
    }
    if (size(driftRules) != 0) {
        PythonCmd cmd = monitorCmd("MONITOR", methods, monitored, windows, freqs, minEffects, thresholds);
        PythonResponse res = sendJsonToPython(pid, cmd);
        reportResult(res, "MONITOR", mon.src);
    }   
    Maybe[int] ms = nothing();
    if (size(latencyRule) != 0) {
        int parsedMs = evalLatencyRule(latencyRule[0]);
        if (parsedMs <= 0) {
            throw invalidLatency("Latency cannot be smaller or equal to 0.");
        }
        ms = just(parsedMs);
    }
    return store(s.name, s.state, s.targetVariable, s.dbConnection, s.trainedModelFilePath, size(monitored) != 0, ms);
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