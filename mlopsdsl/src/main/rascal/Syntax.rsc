module Syntax

layout Layout = WhitespaceAndComment* !>> [\ \t\n\r#];
lexical WhitespaceAndComment = [\ \t\n\r] | @category="Comment" "#" ![\n]* !>> ![\n];

start syntax Pipeline = pipeline: "pipeline" Id name "{" Steps steps "}";

syntax Steps = steps: Load load Split? split Select? select Trans? trans Model model Eval? eval Deploy? deploy Monitor? monitor; 

syntax Load = stepLoad: "load" "(" "path" "=" StrLit path "," "target" "=" StrLit target ("," "databaseURL" "=" StrLit dbURL)? ")";

syntax Split = stepSplit: "split" "(" "train_size" "=" FloatLit trainSize  ("," "random_state" "=" IntLit randomState)? ")";

syntax Select = stepSelect: "select" "(" "features" "=" "[" { StrLit ","}+ features "]" ")";

syntax Trans = stepTrans: "transformation" "(" {PrepTransform ","}+ transforms ")";

syntax Model = stepModel: "model" ModelExpr expr;

syntax Eval = stepEval: "evaluation" "(" {EvalRule ","}+ evalRules ")";

syntax Deploy = stepDeploy: "deployment" "(" "port" "=" IntLit port ")";

syntax Monitor = stepMonitor: "monitoring" "(" {DriftRule ","}+ driftRules ("," LatencyRule latencyRules)? ")"
               | stepMonitorLatency: "monitoring" "(" LatencyRule latencyRule ")";

syntax PrepTransform = prepFill: "fillna" "(" StrLit feature "," FillStrategy strategy ")"
  | prepEncode: "encode" "(" StrLit feature "," EncodingMethod method ")"
  | prepScale: "scale" "(" StrLit feature "," ScaleMethod method ")";
  
syntax FillStrategy = fillMean: "mean"
                   | fillMedian: "median"
                   | fillMode: "mode";

syntax EncodingMethod = encOneHot: "onehot"
                      | encLabel: "label";

syntax ScaleMethod = scaleMinMax: "minmax"
                   | scaleStd: "std";
                  
syntax ModelExpr = modelTrain: Algorithm algo "(" "dir" "=" StrLit path ("," {Param ","}+ hyperParams)? ")";

syntax Algorithm = algoLR: "LinReg" 
               | algoRF: "RandomForest"
               | algoLogReg: "LogReg";
          
syntax Param = hp: Id name "=" Lit val;

syntax Lit = intLit: IntLit 
          | floatLit: FloatLit 
          | strLit: StrLit
          | boolLit: BoolLit;

syntax EvalRule = evalCRule: CMetric cMetric "\>=" FloatLit threshold 
                | evalRRule: RMetric rMetric "\<=" FloatLit threshold;

syntax CMetric = mAccuracy: "accuracy"
                | mPrecision: "precision"
                | mRecall: "recall"
                | mF1: "f1";

syntax RMetric = mMSE: "mse"
                | mRMSE: "rmse";

syntax DriftRule = ruleDrift: DriftMethod dMethod "(" "feature" "=" StrLit feature "," "window" "=" IntLit window "," "checkFrequency" "=" IntLit freq "," "minEffect" "=" FloatLit minEffect ")" "\<=" FloatLit threshold;

syntax DriftMethod = dmKS: "driftKS" | dmChiSquare: "driftChiSquare";

syntax LatencyRule = ruleLatency: "latency" "\<=" IntLit ms;

syntax StrLit = strLit: "\"" StrContent content "\"";
  
lexical Id = [a-zA-Z_][a-zA-Z0-9_]* !>> [a-zA-Z0-9_] \ Keywords;
lexical IntLit   = "-"? [0-9]+;
lexical FloatLit = "-"? [0-9]+ "." [0-9]+;
lexical BoolLit = "true" | "false";
lexical StrContent = ![\"\n\r]* !>> ![\"\n\r];

keyword Keywords
  = "pipeline" 
  | "load" | "path" | "target" | "databaseURL"
  | "split" | "ratio" | "random_state"
  | "select" | "features"
  | "transformation" | "fillna" | "encode" | "scale" | "mean" | "median" | "mode" | "onehot" | "label" | "std" | "minmax" 
  | "model" | "LinReg" | "RandomForest" | "LogReg" | "dir"
  | "evaluation" | "accuracy" | "precision" | "recall" | "f1" | "mse" | "rmse"
  | "deployment" | "port"
  | "monitoring" | "feature" | "window" | "checkFrequency" | "minEffect" | "driftKS" | "driftChiSquare" | "latency";