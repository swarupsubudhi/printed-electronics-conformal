
' nScryptConformal_v2 launcher
' Double-click this file or pin it to the taskbar / Start Menu.
' Runs main.pyw with pythonw.exe so no console window appears.

Dim oShell
Set oShell = CreateObject("WScript.Shell")

' Resolve the folder this .vbs lives in
Dim scriptDir
scriptDir = Left(WScript.ScriptFullName, _
    Len(WScript.ScriptFullName) - Len(WScript.ScriptName))

' Full path to main.pyw (must be in the same folder as this file)
Dim appPath
appPath = scriptDir & "main.pyw"

' Launch with pythonw (suppresses the console window)
' Use "pythonw" if Python is on PATH; otherwise replace with full path:
'   e.g.  "C:\Python311\pythonw.exe"
oShell.Run "pythonw """ & appPath & """", 0, False

Set oShell = Nothing
