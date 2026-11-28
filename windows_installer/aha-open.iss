; AHA open - Windows installer (Inno Setup 6). Built by build_installer.py, which passes every /D below; do not compile by hand.
;
; Small on purpose: this ships `uv.exe` and the AHA open wheel (no PyTorch, no model weights). On the user's machine the
; installer builds a Python environment for the gateway (needs the internet the first time). The model is downloaded later,
; from the web page, after the user has read and accepted its licence (plan/2026-11-24_windows_installer_plan.md).
;
; AppId is the upgrade key of Windows: changing it after a release gives every user TWO entries in Add/Remove Programs.

#ifndef AppVersion
  #error AppVersion is required: build with windows_installer/build_installer.py
#endif
#ifndef WheelName
  #error WheelName is required: build with windows_installer/build_installer.py
#endif
#ifndef Stage
  #error Stage is required: build with windows_installer/build_installer.py
#endif
#ifndef Root
  #error Root is required: build with windows_installer/build_installer.py
#endif
#ifndef OutDir
  #define OutDir "dist"
#endif

[Setup]
AppId={{56B3FE1F-7C37-4C1B-AF6E-9C1F28DF42B4}
AppName=AHA open
AppVersion={#AppVersion}
AppPublisher=AHA open contributors
AppPublisherURL=https://github.com/AIhugART/AHA-open-release
DefaultDirName={localappdata}\Programs\AHA open
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
; Inno 6.7 turns Windows' RedirectionGuard on for Setup by default and everything Setup starts inherits it. A guarded process may
; not follow a junction the user created, and uv's Python is reached through one (the engines' `uv venv` environments): with the guard,
; setup failed on a machine without Python (os error 448) and a gateway started from [Run] could not start a worker (os error 2).
; Setup is per-user and never elevates, so the
; guard (against a lower-privileged user redirecting an administrator's writes) protects nothing here. Observed 2026-11-28.
RedirectionGuard=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutDir}
OutputBaseFilename=AHA-Open-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
LicenseFile={#Root}\LICENSE
UninstallDisplayName=AHA open
CloseApplications=no
SetupLogging=yes

[Files]
; Needed only while installing: extracted to a temp folder by PrepareToInstall, never copied into the install folder. (A copy of uv
; ends up in <app>\venv\Scripts, next to the interpreter, where the web page looks for it.)
Source: "{#Stage}\uv.exe"; Flags: dontcopy
Source: "{#Stage}\wheel\{#WheelName}"; Flags: dontcopy
; The versions of every dependency, exported from uv.lock at build time: what was tested is what gets installed.
Source: "{#Stage}\pins.txt"; Flags: dontcopy
Source: "{#Root}\LICENSE"; DestDir: "{app}\licenses"; DestName: "AHA-open-LICENSE.txt"; Flags: ignoreversion
Source: "{#Root}\NOTICE"; DestDir: "{app}\licenses"; DestName: "AHA-open-NOTICE.txt"; Flags: ignoreversion
Source: "{#Root}\THIRD_PARTY_LICENSES.md"; DestDir: "{app}\licenses"; Flags: ignoreversion
Source: "{#Root}\windows_installer\licenses\*"; DestDir: "{app}\licenses"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\AHA open"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: "-m aha_open.cli start --gui"; WorkingDir: "{app}"; Comment: "Start AHA open and open its web page"
Name: "{autoprograms}\Stop AHA open"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: "-m aha_open.cli stop --gui"; WorkingDir: "{app}"; Comment: "Stop AHA open"

[Run]
Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: "-m aha_open.cli start --gui"; WorkingDir: "{app}"; Description: "Start AHA open now"; Flags: postinstall nowait skipifsilent

[UninstallRun]
; Stop the gateway (and an install it may be running) BEFORE the files are removed.
Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: "-m aha_open.cli stop --quiet"; WorkingDir: "{app}"; Flags: runhidden waituntilterminated skipifdoesntexist; RunOnceId: "StopGateway"

[UninstallDelete]
; The environment is created after the files are copied, so Windows does not know about it.
Type: filesandordirs; Name: "{app}\venv"
Type: files; Name: "{app}\install.log"

[Code]
function Quote(S: String): String;
begin
  Result := '"' + S + '"';
end;

function LogPath(): String;
begin
  Result := ExpandConstant('{app}\install.log');
end;

function LogText(): String;
var
  Raw: AnsiString;
begin
  if LoadStringFromFile(LogPath(), Raw) then
    Result := Lowercase(String(Raw))
  else
    Result := '';
end;

function Has(const Haystack, Needle: String): Boolean;
begin
  Result := Pos(Needle, Haystack) > 0;
end;

{ One sentence the person can act on; different causes get different sentences. }
function Explain(): String;
var
  Log: String;
begin
  Log := LogText();
  if Has(Log, 'dns error') or Has(Log, 'error sending request') or Has(Log, 'timed out') or Has(Log, 'connection refused')
     or Has(Log, 'could not resolve') or Has(Log, 'network is unreachable') or Has(Log, 'failed to fetch')
     or Has(Log, 'failed to download') or Has(Log, 'connectivity is disabled') or Has(Log, 'tunnel error')
     or Has(Log, 'network was disabled') or Has(Log, 'network is disabled') or Has(Log, 'failed to lookup')
     or Has(Log, 'tcp connect error') or Has(Log, 'could not connect') or Has(Log, 'proxy') then
    Result := 'AHA open could not reach the internet to download Python and its libraries. Check the connection '
      + '(and any proxy or firewall), then run this installer again.'
  else if Has(Log, 'os error 112') or Has(Log, 'no space left') then
    Result := 'There is not enough free disk space. Free some space, then run this installer again.'
  else if Has(Log, 'access is denied') or Has(Log, 'os error 5') then
    Result := 'Windows refused to write to the install folder. Pick a folder you own and run this installer again.'
  else
    Result := 'The Python environment could not be prepared.';
  Result := Result + #13#10#13#10 + 'Details: ' + LogPath();
end;

var
  ResultCode: Integer;

function Step(const Caption, CommandLine: String): Boolean;
var
  Code: Integer;
begin
  WizardForm.PreparingLabel.Caption := Caption;
  Result := Exec(ExpandConstant('{cmd}'), '/C "' + CommandLine + ' >> ' + Quote(LogPath()) + ' 2>&1"',
                 ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0);
end;

{ The Python environment is built HERE, before anything is copied or registered, and not after the files: a failure raised in
  CurStepChanged(ssPostInstall) is only logged by Setup, which then finishes with exit code 0 and leaves a half-installed program
  registered in Add/Remove Programs (observed 2026-11-24). A non-empty result from PrepareToInstall cancels cleanly instead. }
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Uv, Wheel, Pins, Venv, Py: String;
begin
  Result := '';
  ForceDirectories(ExpandConstant('{app}'));
  DeleteFile(LogPath());
  ExtractTemporaryFile('uv.exe');
  ExtractTemporaryFile('{#WheelName}');
  ExtractTemporaryFile('pins.txt');
  Uv := ExpandConstant('{tmp}\uv.exe');
  Wheel := ExpandConstant('{tmp}\{#WheelName}');
  Pins := ExpandConstant('{tmp}\pins.txt');
  Venv := ExpandConstant('{app}\venv');
  Py := Venv + '\Scripts\python.exe';
  { An upgrade or a repair: stop the running gateway and rebuild the environment from scratch (the user's data is elsewhere). }
  if FileExists(Venv + '\Scripts\pythonw.exe') then
    Exec(Venv + '\Scripts\pythonw.exe', '-m aha_open.cli stop --quiet', ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode);
  if DirExists(Venv) then
    DelTree(Venv, True, True, True);
  { uv finds or downloads Python, and that Python makes the environment. Not `uv venv`: for a Python it downloaded itself, uv writes a
    console program as `pythonw.exe`, and every shortcut would open a terminal window (observed 2026-11-28). }
  if not Step('Preparing Python (the first time this downloads it)...', Quote(Uv) + ' run --no-project --python 3.12 python -I -m venv --without-pip ' + Quote(Venv)) then
    Result := Explain()
  else if not FileCopy(Uv, Venv + '\Scripts\uv.exe', False) then
    Result := 'Could not copy uv into the Python environment.' + #13#10#13#10 + 'Details: ' + LogPath()
  else if not Step('Installing AHA open...', Quote(Uv) + ' pip install --python ' + Quote(Py) + ' --constraint ' + Quote(Pins) + ' ' + Quote(Wheel + '[download]')) then
    Result := Explain()
  else if not Step('Creating your settings...', Quote(Py) + ' -m aha_open.cli init --profile local') then
    Result := Explain();
  if Result <> '' then
    DelTree(Venv, True, True, True);  { never leave a half-built environment behind; install.log stays for the details }
end;

function DataDir(): String;
begin
  Result := GetEnv('AHA_OPEN_DATA_DIR');
  if Result = '' then
    Result := ExpandConstant('{localappdata}\AHA-open');
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;
  { Default is to KEEP: the model is about 18 GB and re-downloading it is the expensive mistake. A silent uninstall keeps it too. }
  if DirExists(DataDir()) then
    if SuppressibleMsgBox('Also delete your downloaded models, settings and generated images?' + #13#10#13#10 + DataDir()
                          + #13#10#13#10 + 'Choose No to keep them (re-downloading a model takes a long time).',
                          mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
      DelTree(DataDir(), True, True, True);
end;
