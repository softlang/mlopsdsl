import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as cp from 'child_process';
import { ParameterizedLanguageServer, VSCodeUriResolverServer, LanguageParameter } from '@usethesource/rascal-vscode-dsl-lsp-server';

/** Name of the virtual environment - identical to the name used in
 *  development mode under `src/main/python/.mlopsenv`. This keeps the
 *  relative path searched by Rascal via `findResources`
 *  (".mlopsenv/bin/python3" or ".mlopsenv/Scripts/python.exe") identical in
 *  both modes - only the root added to pathConfig differs. */
const VENV_FOLDER_NAME = '.mlopsenv';

/** Minimum Python version required by the dependencies in requirements.txt.
 *  Keep in sync with the pinned versions (e.g. pandas 3 needs 3.11+). */
const MIN_PYTHON: [number, number] = [3, 11];

export async function activate(context: vscode.ExtensionContext) {
	context.subscriptions.push(
		vscode.commands.registerCommand('mlopsdsl.setupPythonEnvironment', () =>
			setupPythonEnvironment(context, true)),
		vscode.commands.registerCommand('mlopsdsl.selectPythonInterpreter', async () => {
			if (await selectPythonInterpreter()) {
				const choice = await vscode.window.showInformationMessage(
					'MLOps DSL: Rebuild the Python environment with the selected interpreter now?',
					'Rebuild', 'Later');
				if (choice === 'Rebuild') {
					await setupPythonEnvironment(context, true);
				}
			}
		})
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

	const language = <LanguageParameter>{
		pathConfig: `pathConfig(srcs=[${mlopsLSPJar}, ${pythonScriptsLoc}, ${venvRootLoc}])`,
		name: "MLOps DSL",
		extensions: ["mlops"],
		mainModule: "Plugin",
		mainFunction: "pipelineLanguageServices"
	};

	// Rascal VS Code needs an instance of this class; it can be shared
	// when there are multiple languages
	const vfs = new VSCodeUriResolverServer(false);
	// Starts the LSP server and connects it to Rascal
	const lsp = new ParameterizedLanguageServer(context,
		vfs,
		calcJarPath(context),
		true,
		"mlops",     // VS Code language ID
		"MLOps DSL", // VS Code language title (visible in the bottom right)
		language);
	// Subscriptions ensure that everything is cleaned up correctly on deactivation
	context.subscriptions.push(lsp);
}

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

// ---------------------------------------------------------------------------
// Python interpreter detection and selection
// ---------------------------------------------------------------------------

interface PythonInfo { executable: string; version: string; }

/** Runs the interpreter and asks it for its real path and version.
 *  Returns undefined if it cannot be executed at all. */
function probePython(cmd: string, args: string[] = []): Promise<PythonInfo | undefined> {
	return new Promise((resolve) => {
		cp.execFile(
			cmd,
			[...args, '-c', 'import sys; print(sys.executable); print("%d.%d.%d" % sys.version_info[:3])'],
			{ timeout: 10000 },
			(err, stdout) => {
				if (err) { return resolve(undefined); }
				const [executable, version] = stdout.trim().split(/\r?\n/);
				resolve(executable && version ? { executable, version } : undefined);
			});
	});
}

function parseVersion(v: string): number[] {
	return v.split('.').map(Number);
}

function compareVersions(a: string, b: string): number {
	const [pa, pb] = [parseVersion(a), parseVersion(b)];
	for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
		const d = (pa[i] ?? 0) - (pb[i] ?? 0);
		if (d !== 0) { return d; }
	}
	return 0;
}

function isSupported(version: string): boolean {
	return compareVersions(version, `${MIN_PYTHON[0]}.${MIN_PYTHON[1]}`) >= 0;
}

/** Finds interpreters on PATH, newest first, without duplicates. */
async function detectInterpreters(): Promise<PythonInfo[]> {
	const minors = ['3.14', '3.13', '3.12', '3.11', '3.10', '3.9'];
	const candidates: [string, string[]][] = process.platform === 'win32'
		? [['python', []], ['py', ['-3']], ...minors.map((v): [string, string[]] => ['py', [`-${v}`]])]
		: [...minors.map((v): [string, string[]] => [`python${v}`, []]), ['python3', []], ['python', []]];

	const probed = await Promise.all(candidates.map(([cmd, args]) => probePython(cmd, args)));
	const seen = new Set<string>();
	const found: PythonInfo[] = [];
	for (const info of probed) {
		if (info && !seen.has(info.executable)) {
			seen.add(info.executable);
			found.push(info);
		}
	}
	return found.sort((a, b) => compareVersions(b.version, a.version));
}

/** Uses the configured interpreter, otherwise the newest supported one found. */
async function resolvePython(): Promise<{ info?: PythonInfo; problem?: string }> {
	const configured = vscode.workspace.getConfiguration('mlopsdsl')
		.get<string>('pythonPath', '').trim();

	if (configured) {
		const info = await probePython(configured);
		if (!info) {
			return { problem: `The configured interpreter "${configured}" could not be run.` };
		}
		if (!isSupported(info.version)) {
			return { problem: `The configured interpreter is Python ${info.version}, but ${MIN_PYTHON.join('.')} or newer is required.` };
		}
		return { info };
	}

	const supported = (await detectInterpreters()).filter(i => isSupported(i.version));
	return supported.length > 0
		? { info: supported[0] }
		: { problem: `No Python ${MIN_PYTHON.join('.')} or newer was found.` };
}

/** Lets the user pick an interpreter and stores it in the settings. */
async function selectPythonInterpreter(): Promise<boolean> {
	const found = await vscode.window.withProgress(
		{ location: vscode.ProgressLocation.Window, title: 'MLOps DSL: Searching for Python interpreters…' },
		() => detectInterpreters());

	type Item = vscode.QuickPickItem & { info?: PythonInfo };
	const items: Item[] = [
		...found.filter(i => isSupported(i.version)).map(i => ({
			label: `Python ${i.version}`,
			detail: i.executable,
			info: i
		})),
		{ label: '$(folder-opened) Browse for an interpreter…' }
	];

	const picked = await vscode.window.showQuickPick(items, {
		title: 'MLOps DSL: Select Python interpreter',
		placeHolder: `Python ${MIN_PYTHON.join('.')} or newer`
	});
	if (!picked) { return false; }

	let chosen = picked.info;
	if (!chosen) {
		const uris = await vscode.window.showOpenDialog({
			canSelectMany: false,
			title: 'Select a Python executable'
		});
		if (!uris || uris.length === 0) { return false; }
		chosen = await probePython(uris[0].fsPath);
		if (!chosen) {
			void vscode.window.showErrorMessage('MLOps DSL: The selected file is not a working Python interpreter.');
			return false;
		}
		if (!isSupported(chosen.version)) {
			void vscode.window.showErrorMessage(
				`MLOps DSL: That is Python ${chosen.version}, but ${MIN_PYTHON.join('.')} or newer is required.`);
			return false;
		}
	}

	await vscode.workspace.getConfiguration('mlopsdsl')
		.update('pythonPath', chosen.executable, vscode.ConfigurationTarget.Global);
	return true;
}

// ---------------------------------------------------------------------------
// Environment setup
// ---------------------------------------------------------------------------

async function setupPythonEnvironment(context: vscode.ExtensionContext, showSuccessMessage: boolean) {
	const venv = venvDir(context);
	const requirements = context.asAbsolutePath(path.join('assets', 'python', 'requirements.txt'));

	// Resolve BEFORE showing the progress notification - we may need to prompt the user
	const { info, problem } = await resolvePython();
	if (!info) {
		await handleNoInterpreter(context, problem ?? 'No suitable Python interpreter found.', showSuccessMessage);
		return;
	}

	await vscode.window.withProgress({
		location: vscode.ProgressLocation.Notification,
		title: 'MLOps DSL: Setting up Python environment…',
		cancellable: false
	}, async (progress) => {
		try {
			fs.mkdirSync(venvRootFsPath(context), { recursive: true });

			// Without this, `venv` would reuse an existing env built with an old interpreter
			fs.rmSync(venv, { recursive: true, force: true });

			progress.report({ message: `Create virtual environment (Python ${info.version})…` });
			await run(info.executable, ['-m', 'venv', venv]);

			const pip = pythonExecutablePath(venv);
			progress.report({ message: 'Update pip…' });
			await run(pip, ['-m', 'pip', 'install', '--upgrade', 'pip']);

			if (fs.existsSync(requirements)) {
				progress.report({ message: 'Install dependencies…' });
				await run(pip, ['-m', 'pip', 'install', '-r', requirements]);
			}

			if (showSuccessMessage) {
				void vscode.window.showInformationMessage('MLOps DSL: Python environment setup successful.');
			}
		} catch (err) {
			void vscode.window.showErrorMessage(`MLOps DSL: Python environment setup failed: ${err}`);
		}
	});
}

async function handleNoInterpreter(context: vscode.ExtensionContext, problem: string, showSuccessMessage: boolean) {
	const choice = await vscode.window.showErrorMessage(
		`MLOps DSL: ${problem}`,
		'Select interpreter', 'Open Python download page', 'Try again'
	);

	if (choice === 'Select interpreter') {
		if (await selectPythonInterpreter()) {
			await setupPythonEnvironment(context, showSuccessMessage);
		}
	} else if (choice === 'Open Python download page') {
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
		// No shell needed: cmd is an absolute path to an executable
		const proc = cp.spawn(cmd, args);
		let stderr = '';
		proc.stderr?.on('data', (d) => { stderr += d.toString(); });
		proc.on('error', reject);
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