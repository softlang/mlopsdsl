import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as cp from 'child_process';
import {
	ParameterizedLanguageServer,
	VSCodeFileSystemInRascal,
	type LanguageParameter,
} from '@usethesource/rascal-vscode-dsl-lsp-server';

/** Name of the virtual environment - identical to the name used in
 *  development mode under `src/main/python/.mlopsenv`. This keeps the
 *  relative path searched by Rascal via `findResources`
 *  (".mlopsenv/bin/python3" or ".mlopsenv/Scripts/python.exe") identical in
 *  both modes - only the root added to pathConfig differs. */
const VENV_FOLDER_NAME = '.mlopsenv';

export async function activate(context: vscode.ExtensionContext) {
	context.subscriptions.push(
		vscode.commands.registerCommand('mlopsdsl.setupPythonEnvironment', () =>
			setupPythonEnvironment(context, true))
	);

	// Checking and offering setup runs in the background and does not block activation.
	void ensurePythonEnvironment(context);

	// JAR containing the "Plugin" module (the actual language definition)
	const mlopsLSPJar = `|jar+file://${context.extensionUri.path}/assets/jars/mlopsdsl-lsp.jar!|`;
	// The Python scripts are NOT in the JAR (createProcess cannot pass
	// jar+file:// URIs to an external process), but are stored as actual
	// files in the extension directory - see assets/python
	const pythonScriptsLoc = `|file://${context.extensionUri.path}/assets/python|`;
	// Persistent location independent of the system and project, where the
	// extension creates the virtual environment after installation
	const venvRootLoc = `|file://${venvRootUriPath(context)}|`;

	// Since Rascal 0.43 / Rascal LSP 2.23 the path config must be complete: rascal and rascal-lsp
	// are no longer added implicitly. This mirrors what PathConfig produces for
	// getProjectPathConfig(..., interpreter_external()) (see PathConfig.buildNormalProjectConfig),
	// with jars standing in for the Maven artifacts:
	//  - srcs: rascal stdlib (inside rascal.jar), the sources of each dependency (jar), then our own sources
	//  - libs: the plain jar files, used for class loading
	// rascal.jar and rascal-lsp.jar are the very jars the language server runs on (copied by webpack).
	const distJars = `${context.extensionUri.path}/dist/rascal-lsp`;
	const rascalJar = `${distJars}/rascal.jar`;
	const rascalLspJar = `${distJars}/rascal-lsp.jar`;
	const dslJarPath = `${context.extensionUri.path}/assets/jars/mlopsdsl-lsp.jar`;

	const srcs = [
		`|jar+file://${rascalJar}!|`,
		`|jar+file://${rascalLspJar}!|`,
		mlopsLSPJar,
		pythonScriptsLoc,
		venvRootLoc,
	];
	const libs = [
		`|file://${rascalJar}|`,
		`|file://${rascalLspJar}|`,
		`|file://${dslJarPath}|`,
	];

	const language: LanguageParameter = {
		pathConfig: `pathConfig(srcs=[${srcs.join(', ')}], libs=[${libs.join(', ')}])`,
		name: "MLOps DSL",
		extensions: ["mlops"],
		mainModule: "Plugin",
		mainFunction: "pipelineLanguageServices"
	};

	const log = vscode.window.createOutputChannel('MLOps DSL', { log: true });

	// deployMode=false makes the client connect to a server on port 9999 instead of starting one,
	// so it is only switched off deliberately (same convention as the Rascal extension itself).
	const deployMode = (process.env['RASCAL_LSP_DEV'] ?? 'false') !== 'true';

	// Virtual file system bridge between VS Code and Rascal
	const vfs = new VSCodeFileSystemInRascal(!deployMode, log);

	// Starts the dedicated LSP server for this language and connects it to Rascal.
	// With a dedicated language no registerLanguage() call is needed.
	const lsp = new ParameterizedLanguageServer(context,
		vfs,
		calcJarPath(context),
		deployMode,
		"mlops",     // VS Code language ID (must match package.json)
		"MLOps DSL", // VS Code language title (visible in the bottom right)
		language);
	// Subscriptions ensure that everything is cleaned up correctly on deactivation
	context.subscriptions.push(log, vfs, lsp);
}

/** Folder with rascal-lsp.jar and rascal.jar, filled by CopyPlugin in webpack.config.js */
function calcJarPath(context: vscode.ExtensionContext) {
	return context.asAbsolutePath(path.join('.', 'dist', 'rascal-lsp'));
}

/** Native filesystem variant of the persistent location (for fs/child_process). */
function venvRootFsPath(context: vscode.ExtensionContext): string {
	// globalStorageUri is stable for each extension - independent of the project/
	// workspace and the computer/username of the person who installs the
	// extension. That is why it is the right portable location for a virtual
	// environment created once.
	return context.globalStorageUri.fsPath;
}

/** URI path variant of the same location (for the Rascal `pathConfig` location). */
function venvRootUriPath(context: vscode.ExtensionContext): string {
	return context.globalStorageUri.path;
}

function venvDir(context: vscode.ExtensionContext): string {
	return path.join(venvRootFsPath(context), VENV_FOLDER_NAME);
}

function pythonExecutablePath(venv: string): string {
	return process.platform === 'win32'
		? path.join(venv, 'Scripts', 'python.exe')
		: path.join(venv, 'bin', 'python3');
}

async function ensurePythonEnvironment(context: vscode.ExtensionContext) {
	const venv = venvDir(context);
	if (fs.existsSync(pythonExecutablePath(venv))) {
		return; // Already set up
	}

	const choice = await vscode.window.showInformationMessage(
		'The MLOpsDSL needs a local Python environment for all its features (schema inference, type checking, pipeline execution). Setup now?',
		'Setup now', 'Later'
	);
	if (choice === 'Setup now') {
		await setupPythonEnvironment(context, false);
	}
}

/** Indicates that the configured Python command does not exist at all
 *  (e.g. because Python is not installed on the system), as opposed
 *  to an error *during* the execution of venv/pip. */
class PythonNotFoundError extends Error {}

async function setupPythonEnvironment(context: vscode.ExtensionContext, showSuccessMessage: boolean) {
	const venv = venvDir(context);
	const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
	const requirements = context.asAbsolutePath(path.join('assets', 'python', 'requirements.txt'));

	await vscode.window.withProgress({
		location: vscode.ProgressLocation.Notification,
		title: 'MLOps DSL: Setting up Python environment…',
		cancellable: false
	}, async (progress) => {
		try {
			fs.mkdirSync(venvRootFsPath(context), { recursive: true });

			progress.report({ message: 'Create virutal environment…' });
			await run(pythonCmd, ['-m', 'venv', venv]);

			const pip = pythonExecutablePath(venv);
			progress.report({ message: 'Update pip…' });
			await run(pip, ['-m', 'pip', 'install', '--upgrade', 'pip']);

			if (fs.existsSync(requirements)) {
				progress.report({ message: 'Install dependencies…' });
				await run(pip, ['-m', 'pip', 'install', '-r', requirements]);
			}

			if (showSuccessMessage) {
				void vscode.window.showInformationMessage('MLOps DSL: Python environment setup successfull.');
			}
		} catch (err) {
			if (err instanceof PythonNotFoundError) {
				await handlePythonNotFound(context, pythonCmd, showSuccessMessage);
			} else {
				void vscode.window.showErrorMessage(`MLOps DSL: Python environment setup failed: ${err}`);
			}
		}
	});
}

async function handlePythonNotFound(context: vscode.ExtensionContext, pythonCmd: string, showSuccessMessage: boolean) {
	const choice = await vscode.window.showErrorMessage(
		`MLOps DSL: No "${pythonCmd}" was found. Please install Python 3 and try again.`,
		'Open Python download page', 'Try again'
	);

	if (choice === 'Open Python download page') {
		void vscode.env.openExternal(vscode.Uri.parse('https://www.python.org/downloads/'));
		// Offer a retry directly instead of referring the user to the command
		// palette - they do not need to know the command name
		const retry = await vscode.window.showInformationMessage(
			'MLOps DSL: When the Python installation is completed, you can try again here.',
			'Try again'
		);
		if (retry === 'Try again') {
			await setupPythonEnvironment(context, showSuccessMessage);
		}
	} else if (choice === 'Try again') {
		await setupPythonEnvironment(context, showSuccessMessage);
	}
}

function run(cmd: string, args: string[]): Promise<void> {
	return new Promise((resolve, reject) => {
		const proc = cp.spawn(cmd, args, { shell: process.platform === 'win32' });
		let stderr = '';
		proc.stderr?.on('data', (d) => { stderr += d.toString(); });
		proc.on('error', (err: NodeJS.ErrnoException) => {
			if (err.code === 'ENOENT') {
				reject(new PythonNotFoundError(`Command "${cmd}" was not found.`));
			} else {
				reject(err);
			}
		});
		proc.on('close', (code: number | null) => {
			if (code === 0) {
				resolve();
			} else {
				reject(new Error(`${cmd} ${args.join(' ')} finished with exit code ${code}: ${stderr}`));
			}
		});
	});
}

export function deactivate() {}