<#
.SYNOPSIS
  Text to image with Windows PowerShell 5.1 or PowerShell 7 (no modules needed).
.EXAMPLE
  .\generate.ps1 "a red fox in the snow" -Out fox.png
.EXAMPLE
  .\generate.ps1 "a red fox" -Model qwen-image-2.1 -Width 1024 -Height 1024 -Seed 7
.NOTES
  Settings: AHA_OPEN_URL (default http://127.0.0.1:8890), AHA_OPEN_TOKEN (default: the local token file).
  The token is never printed. Specification: docs/03_api_v1.md.
#>
param(
  [Parameter(Mandatory = $true, Position = 0)][string]$Prompt,
  [string]$Out = "out.png",
  [string]$Model,
  [int]$Width = 0,
  [int]$Height = 0,
  [long]$Seed = -1
)
$ErrorActionPreference = "Stop"

function Stop-WithError([string]$text) { [Console]::Error.WriteLine("error: $text"); exit 1 }

$base = if ($env:AHA_OPEN_URL) { $env:AHA_OPEN_URL.TrimEnd("/") } else { "http://127.0.0.1:8890" }
$token = $env:AHA_OPEN_TOKEN
if (-not $token) {
  $data = if ($env:AHA_OPEN_DATA_DIR) { $env:AHA_OPEN_DATA_DIR } else { Join-Path $env:LOCALAPPDATA "AHA-open" }
  $file = Join-Path $data "local_client.token"
  if (Test-Path $file) { $token = (Get-Content -Raw $file).Trim() }
}
if (-not $token) { Stop-WithError "no token. Set `$env:AHA_OPEN_TOKEN (docs/aha_open_api/01_quickstart.md)." }
$headers = @{ Authorization = "Bearer $token" }

# One place that turns an HTTP failure into the sentence the server gave (the body is {"error": {"code", "message", "retry_after_s"}}).
# It throws a plain object {Status, Code, Message, RetryAfter}; PowerShell hands it back as $_.TargetObject in a catch block.
function Invoke-Api([string]$Method, [string]$Path, $Body = $null, [string]$OutFile = $null) {
  $call = @{ Method = $Method; Uri = "$base$Path"; Headers = $headers; UseBasicParsing = $true }
  if ($null -ne $Body) {
    $call.Body = [Text.Encoding]::UTF8.GetBytes(($Body | ConvertTo-Json -Depth 8 -Compress))
    $call.ContentType = "application/json"
  }
  try {
    if ($OutFile) { Invoke-WebRequest @call -OutFile $OutFile | Out-Null; return }
    return Invoke-RestMethod @call
  } catch {
    $response = $_.Exception.Response
    $detail = $_.ErrorDetails.Message
    $failure = [pscustomobject]@{ Status = 0; Code = "unreachable"; Message = "cannot reach $base ($($_.Exception.Message))"; RetryAfter = $null }
    if ($response -and $detail) {
      try { $e = ($detail | ConvertFrom-Json).error } catch { $e = $null }
      if ($e) { $failure = [pscustomobject]@{ Status = [int]$response.StatusCode; Code = $e.code; Message = $e.message; RetryAfter = $e.retry_after_s } }
    }
    throw $failure
  }
}

# -Out may be relative (to the current PowerShell folder) or absolute; check its folder now, so a typo is not reported as a network error.
$outPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out)
if (-not (Test-Path -LiteralPath (Split-Path -Parent $outPath))) { Stop-WithError "the folder for -Out does not exist: $(Split-Path -Parent $outPath)" }

try {
  $manifest = Invoke-Api GET "/v1/manifest"
  $models = @($manifest.models | Where-Object { $_.tasks -contains "image.generate" -and (-not $Model -or $_.id -eq $Model) })
  $chosen = $models | Where-Object { $_.state -eq "ready" } | Select-Object -First 1
  if (-not $chosen) {
    $why = if ($models) { ($models | ForEach-Object { "$($_.id) is $($_.state)" }) -join ", " } else { "no model offers image.generate" }
    Stop-WithError "no model is ready ($why). Install the model or wait for it to load."
  }
  [Console]::Error.WriteLine("model $($chosen.id) - license: $($chosen.license.name) (commercial use: $($chosen.license.commercial_use))")
  if ($Width -eq 0) { $size = $chosen.capabilities.sizes[0]; $Width = $size.width; $Height = $size.height }   # sizes are never stretched
  $params = @{ width = $Width; height = $Height }
  if ($Seed -ge 0) { $params.seed = $Seed }
  $body = @{ task = "image.generate"; model = $chosen.id; prompt = $Prompt; params = $params; client_request_id = [guid]::NewGuid().ToString("N") }

  while ($true) {                       # wait out `busy`; the same client_request_id can never create two jobs
    try { $job = Invoke-Api POST "/v1/jobs" $body; break }
    catch {
      $f = $_.TargetObject
      if ($f -and $f.Code -eq "busy") { Start-Sleep -Seconds ([math]::Max(1, [double]$f.RetryAfter)) } else { throw }
    }
  }
  $pause = 250
  while ($job.status -notin @("succeeded", "failed", "cancelled")) {
    Start-Sleep -Milliseconds $pause; $pause = [math]::Min($pause * 1.5, 2000)
    $job = Invoke-Api GET "/v1/jobs/$($job.id)"
  }
  if ($job.status -ne "succeeded") { Stop-WithError "the job $($job.status): $($job.error.code): $($job.error.message)" }
  Invoke-Api GET "/v1/jobs/$($job.id)/result?index=0" $null $outPath
  $r = $job.results[0]
  [Console]::Error.WriteLine("saved ${Out}: $($r.width)x$($r.height), $($r.bytes) bytes, seed $($r.seed)")
} catch {
  $f = $_.TargetObject
  if ($f -and $f.Code) { Stop-WithError "$($f.Code): $($f.Message)" } else { Stop-WithError "$($_.Exception.Message)" }
}
