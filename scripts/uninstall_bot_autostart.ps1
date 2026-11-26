$ErrorActionPreference = "SilentlyContinue"

$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $WorkspaceRoot "data\telegram_bot.pid"
$TaskName = "ViralStudioPro_TelegramBot"

$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Task) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
}

if (Test-Path $PidFile) {
    $BotPid = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($BotPid -match '^\d+$') {
        Stop-Process -Id ([int]$BotPid) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# Fallback sweep for any lingering python/pythonw running scripts\telegram_bot.py
Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" |
    Where-Object { $_.CommandLine -like "*telegram_bot.py*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

if ($Task) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "UNINSTALLED: $TaskName"
} else {
    Write-Output "TASK_NOT_FOUND (processes cleaned): $TaskName"
}
