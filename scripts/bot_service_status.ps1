$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $WorkspaceRoot "data\telegram_bot.pid"
$LogFile = Join-Path $WorkspaceRoot "logs\telegram_bot.log"
$TaskName = "ViralStudioPro_TelegramBot"

$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$TaskState = if ($Task) { $Task.State } else { "NotRegistered" }

$ActivePid = if (Test-Path $PidFile) { (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim() } else { "none" }
$Alive = $false
$ProcName = ""
if ($ActivePid -match '^\d+$') {
    $Proc = Get-Process -Id ([int]$ActivePid) -ErrorAction SilentlyContinue
    if ($Proc) {
        $Alive = $true
        $ProcName = $Proc.ProcessName
    }
}

Write-Output "TASK_NAME=$TaskName | STATE=$TaskState"
Write-Output "PID=$ActivePid | ALIVE=$Alive | PROCESS=$ProcName"
if (Test-Path $LogFile) {
    Write-Output "=== LOG TAIL ($LogFile) ==="
    Get-Content $LogFile -Tail 15
} else {
    Write-Output "LOG_NOT_FOUND: $LogFile"
}
