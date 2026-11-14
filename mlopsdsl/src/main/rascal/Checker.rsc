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
import TypeSystem;
import Store;
import InlineFixes;

Summary mlopsAnalysisService(loc l, start[Pipeline] input) 
    = mlopsSummaryServiceQuick(l, implode(#AST::Pipeline, input));

Summary mlopsBuildService(loc l , start[Pipeline] input)
    = mlopsSummaryServiceSlow(l, implode(#AST::Pipeline, input));

Summary mlopsSummaryServiceQuick(loc l, AST::Pipeline input) {
    Summary s = summary(l);
    MLOpsStore store = initMLOpsStore();
    store = checkStepsQuick(input, store);
    s.messages += store.messages;
    return s;
}

MLOpsStore initMLOpsStore() {
    return cStore("", false, null(), {});
}

MLOpsStore checkStepsQuick(AST::Pipeline input, MLOpsStore store) {
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
    store = evalModel(steps.model, store, -1, checker=true);
    store = checkHyperparams(steps.model.expr, s=store);
    if (size(steps.eval) != 0) {
        store = evalEval(steps.eval[0], store, -1, checker=true);
        store = checkEval(steps.eval[0], store.task, s=store);
    }
    if (size(steps.monitor) != 0) {
        if (size(steps.deploy) == 0) {
            store.messages += {<steps.monitor[0].src, error("User inputs cannot be monitored without deployed model.", steps.monitor[0].src,
                fixes=prepareNoDeploymentFixes(steps.monitor[0].src))>};
        } else {
            store = evalMonitor(steps.monitor[0], store, -1, checker=true);
        }
    }
    return store;
}   

Summary mlopsSummaryServiceSlow(loc l, AST::Pipeline input) {
    Summary s = summary(l);
    store = checkStepsSlow(input);
    s.messages += store.messages;
    return s;
}

TypeStore checkStepsSlow(AST::Pipeline input) {
    AST::Steps steps = input.steps;
    store = checkLoad(steps.load, checker=true);
    if (size(steps.select) != 0) {
        store = checkSelect(steps.select[0], store, checker=true);
    }
    if (size(steps.trans) != 0) {
        store = checkTrans(steps.trans[0], store, checker=true);
    }
    store = checkModel(steps.model, store, checker=true);
    if (size(steps.monitor) != 0) {
        store = checkMonitor(steps.monitor[0], store, checker=true);
    }
    return store;
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