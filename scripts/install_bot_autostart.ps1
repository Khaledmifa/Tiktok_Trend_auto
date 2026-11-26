$ErrorActionPreference = "Stop"

$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$BotScript = Join-Path $WorkspaceRoot "scripts\telegram_bot.py"
$PidFile = Join-Path $WorkspaceRoot "data\telegram_bot.pid"
$LogFile = Join-Path $WorkspaceRoot "logs\telegram_bot.log"
$TaskName = "ViralStudioPro_TelegramBot"

# Resolve pythonw.exe whose sibling python.exe has telethon & dotenv installed
$Candidates = @(
    (Join-Path $WorkspaceRoot ".venv\Scripts\pythonw.exe"),
    (Join-Path $WorkspaceRoot "venv\Scripts\pythonw.exe")
)
$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($PythonCmd -and $PythonCmd.Source) {
    $PyDir = Split-Path -Parent $PythonCmd.Source
    $Candidates += (Join-Path $PyDir "pythonw.exe")
}

$PythonW = $null
foreach ($cand in $Candidates) {
    if ($cand -and (Test-Path $cand)) {
        $sibPy = Join-Path (Split-Path -Parent $cand) "python.exe"
        if (Test-Path $sibPy) {
            cmd /c "`"$sibPy`" -c `"import telethon, dotenv`" >nul 2>nul"
            if ($LASTEXITCODE -eq 0) {
                $PythonW = $cand
                break
            }
        }
    }
}
if (-not $PythonW) {
    # Fallback to hidden powershell wrapper launching python.exe
    $PyExe = if ($PythonCmd) { $PythonCmd.Source } else { "python" }
    $Action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -WindowStyle Hidden -Command `"& '$PyExe' '$BotScript'`"" `
        -WorkingDirectory $WorkspaceRoot
    $LauncherUsed = "powershell-hidden -> $PyExe"
} else {
    $Action = New-ScheduledTaskAction -Execute $PythonW `
        -Argument "`"$BotScript`"" `
        -WorkingDirectory $WorkspaceRoot
    $LauncherUsed = $PythonW
}

$CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser
$Trigger.Delay = "PT15S"

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -RunOnlyIfNetworkAvailable `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

$Principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel Limited

# Stop any prior instance cleanly before re-registering
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
}
if (Test-Path $PidFile) {
    $OldPid = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($OldPid -match '^\d+$') {
        Stop-Process -Id ([int]$OldPid) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Viral Studio Pro 100% Free Interactive Telegram Bot (silent background pythonw.exe)" `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 4

$TaskInfo = Get-ScheduledTask -TaskName $TaskName
$ActivePid = if (Test-Path $PidFile) { (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim() } else { "" }
$ProcAlive = $false
if ($ActivePid -match '^\d+$') {
    $ProcAlive = $null -ne (Get-Process -Id ([int]$ActivePid) -ErrorAction SilentlyContinue)
}

Write-Output "TASK_NAME=$TaskName"
Write-Output "TASK_STATE=$($TaskInfo.State)"
Write-Output "LAUNCHER=$LauncherUsed"
Write-Output "PID=$ActivePid"
Write-Output "ALIVE=$ProcAlive"
if (Test-Path $LogFile) {
    Write-Output "=== LAST LOG LINES ==="
    Get-Content $LogFile -Tail 8
}
