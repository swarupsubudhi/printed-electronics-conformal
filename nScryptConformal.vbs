' ─────────────────────────────────────────
'  nScryptConformal v2 Silent Launcher
'  Runs launch.bat invisibly (no cmd window)
'  This is what the desktop shortcut points to
' ─────────────────────────────────────────

Dim shell, batPath
Set shell = CreateObject("WScript.Shell")

' Get the folder this .vbs file lives in
batPath = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

' Run the batch file silently (0 = hidden window, False = don't wait)
shell.Run "cmd /c """ & batPath & "\launch.bat""", 0, False

Set shell = Nothing
