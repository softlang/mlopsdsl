module Plugin

import util::Reflective;
import ParseTree;
import util::Reflective;
import util::IDEServices;
import util::LanguageServer;
import IO;
import String;

import Syntax;
import Interpreter;
import TypeSystem;
import Checker;
import Outline;

start[Pipeline] pipelineParsingService(str s, loc l) =
    parse(#start[Pipeline], s, l, allowRecovery=true);

set[LanguageService] pipelineLanguageServices() = {
    parsing(pipelineParsingService),
    analysis(mlopsAnalysisService, 
        providesHovers = false, providesDefinitions = false, 
        providesReferences = false, providesImplementations = false),
    build(mlopsBuildService, 
        providesHovers = false, providesDefinitions = false, 
        providesReferences = false, providesImplementations = false),
    hover(mlopsHoverService),
    definition(mlopsDefinitionService),
    references(mlopsReferencesService),
    documentSymbol(mlopsDocumentSymbolService),
    codeLens(pipelineCodeLenseService),
    execution(pipelineExecutionService)
};

data Command = runPipeline(start[Pipeline] pipeline)
             | reRunPipeline(start[Pipeline] pipeline)
             | typecheckPipeline(start[Pipeline] pipeline);

lrel[loc,Command] pipelineCodeLenseService(start[Pipeline] input)
    =[<input.src, runPipeline(input, title="Run MLOps pipeline")>,
    <input.src, reRunPipeline(input, title="Rerun MLOps pipeline")>,
    <input.src, typecheckPipeline(input, title="Typecheck MLOps pipeline")>];

value pipelineExecutionService(runPipeline(start[Pipeline] input)) {
    MLOpsStore store = Interpreter::evalPipeline(input.top);
    return ("result": true);
}

value pipelineExecutionService(reRunPipeline(start[Pipeline] input)) {
    MLOpsStore store = Interpreter::evalPipeline(input.top, reRun=true);
    return ("result": true);
}

value pipelineExecutionService(typecheckPipeline(start[Pipeline] input)) {
    TypeStore store = TypeSystem::checkPipeline(input.top);
    return ("result": true);
}

void main() {
    registerLanguage( 
        language(
            getProjectPathConfig(|project://mlopsdsl|, mode=interpreter_external()),
            "MLOps", 
            {"mlops"}, 
            "Plugin",  
            "pipelineLanguageServices" 
        )
    );
}