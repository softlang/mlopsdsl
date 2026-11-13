module PythonBridge

import util::ShellExec;
import util::SystemAPI;
import lang::json::IO;
import IO;
import String;
import Set;
import Exception;

data RuntimeException
    = resourceNotFound(str cause)
    | pythonEnvNotFound(str cause)
    | workerStartFailed(str cause)
    | workerCrashed(str cause)
    | workerTimeout(str cause)
    ;

data PythonCmd 
  = loadCmd(str cmd, str path, str target, str dbURL, bool reRun)
  | splitCmd(str cmd, real ratio, int randomState)
  | selectCmd(str cmd, list[str] features)
  | transformCmd(str cmd, str action, str feature, str method)
  | trainCmd(str cmd, str algo, map[str, str] hyperparameters, str modelDir)
  | evalCmd(str cmd, str metric)
  | monitorCmd(str cmd, list[str] methods, list[str] features, list[int] windows, list[int] frequencies, list[real] minEffects, list[real] thresholds)
  ;

data PythonResponse
  = response(str status, str message, str modelFilePath, map[str,real] evalResults, int code)
  ;

public loc getPath(str file) {
    set[loc] found = findResources(file);
    if (size(found) != 1) {
        throw resourceNotFound("Expected exactly one file with name <file>, found <size(found)>: <found>");
    }
    return getSingleFrom(found);
}

@synopsis{True when this evaluator is running on Windows.}
private bool isWindowsOS() = /^(?i)windows/ := getSystemProperty("os.name");

@synopsis{Relative resource path (from a source-/classpath root) of the python executable
inside the virtual environment `.mlopsenv`, made independent of the OS the extension runs on.}
private str pythonExecutableRelPath()
    = isWindowsOS() ? ".mlopsenv/Scripts/python.exe" : ".mlopsenv/bin/python3";

@synopsis{Locates the python executable of the `.mlopsenv` virtual environment.
Works both in interpreter/dev mode (where `.mlopsenv` lives next to the scripts under
`src/main/python`) and in the packaged extension (where `.mlopsenv` is created on demand
by the extension and its parent folder is added to the runtime path config), because both
modes resolve `findResources` relative to their configured source/classpath roots.}
public loc getPythonExecutable() {
    set[loc] found = findResources(pythonExecutableRelPath());
    if (size(found) != 1) {
        throw pythonEnvNotFound("Python environment (.mlopsenv) not found or ambiguous (<size(found)> matches). " +
                                "Make sure the Python environment for this extension has been set up.");
    }
    return getSingleFrom(found);
}

PID startPythonWorker() {
    PID pid = createProcess(getPythonExecutable(), args=[getPath("pipeline_worker.py")], envVars=("PYTHONIOENCODING":"utf-8"));
    if (!isAlive(pid)) throw workerStartFailed("Python worker could not be started.");
    return pid;
}

PythonResponse sendJsonToPython(PID pid, PythonCmd command, int timeoutMs = 30000) {
    str jsonPayload = asJSON(command);
    writeTo(pid, jsonPayload + "\n");
    str buffer = "";
    int waited = 0;
    while (!contains(buffer, "\n")) {
        str chunk = readWithWait(pid, 100);
        buffer += chunk;
        if (chunk == "") {
            waited += 100;
            if (!isAlive(pid)) {
                buffer += readFrom(pid);
                if (!contains(buffer, "\n")) {
                    throw workerCrashed("Python process crashed! Error: <readEntireErrStream(pid)>");
                }
            } else if (waited >= timeoutMs) {
                stopPythonWorker(pid);
                throw workerTimeout("Timeout: Python worker does not respond.");
            }
        }
        
    }
    str line = split("\n", buffer)[0];
    return parseJSON(#PythonResponse, trim(line));
}

void stopPythonWorker(PID pid) {
    try {
        if (isAlive(pid) || isZombie(pid)) {
            killProcess(pid, force=true);
        }
    } catch RuntimeException _: {
        return;   // Prozess war schon aufgeräumt
    }
}