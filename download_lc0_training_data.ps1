<#
.SYNOPSIS
    Downloads Leela Chess Zero (lc0) self-play training data (.tar chunks).

.DESCRIPTION
    Fetches the directory index for a given lc0 training run from
    storage.lczero.org, picks the newest N .tar files, and downloads them
    into a local folder. Uses BITS for efficient, resumable transfers and
    skips files that are already fully downloaded (verified by size).

    Each .tar chunk is roughly 0.5-2 GB. Full runs are many terabytes, so
    by default this only grabs a small sample of the most recent chunks.

.PARAMETER Run
    The run folder on the server (e.g. test80, test79, test75). Default: test80.

.PARAMETER Count
    How many of the newest .tar files to download. Default: 2.

.PARAMETER Destination
    Local folder to save files into. Default: .\training_data

.PARAMETER BaseUrl
    Base URL of the training data index. Default is the official lc0 storage.

.EXAMPLE
    .\download_lc0_training_data.ps1
    Downloads the 2 newest test80 chunks into .\training_data

.EXAMPLE
    .\download_lc0_training_data.ps1 -Run test80 -Count 10 -Destination D:\lc0data
    Downloads the 10 newest test80 chunks into D:\lc0data
#>
[CmdletBinding()]
param(
    [string]$Run = 'test80',
    [int]$Count = 2,
    [string]$Destination = (Join-Path $PSScriptRoot 'training_data'),
    [string]$BaseUrl = 'https://storage.lczero.org/files/training_data',
    [int]$MinSizeMB = 50
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$indexUrl = "$BaseUrl/$Run/"
Write-Host "Fetching index: $indexUrl"

$index = Invoke-WebRequest -Uri $indexUrl -UseBasicParsing

# Parse each listing line for the .tar file name and its size in bytes.
# nginx autoindex format: <a href="file">file</a>   DD-Mon-YYYY HH:MM   <size>
$pattern = 'training-run1-' + [regex]::Escape($Run) +
    '-\d{8}-\d{4}\.tar</a>\s+\d{2}-\w{3}-\d{4} \d{2}:\d{2}\s+(\d+)'
$files = [regex]::Matches($index.Content, $pattern) | ForEach-Object {
    [pscustomobject]@{
        Name = ($_.Value -replace '</a>.*$', '')
        Size = [int64]$_.Groups[1].Value
    }
}

if (-not $files) {
    throw "No .tar files found at $indexUrl. Check the run name."
}

$minBytes = [int64]$MinSizeMB * 1MB
$realFiles = $files | Where-Object { $_.Size -ge $minBytes }

if (-not $realFiles) {
    throw "Only placeholder files (< $MinSizeMB MB) found in run '$Run'. Try a different run."
}

# Newest first (file names are date-sortable: YYYYMMDD-HHMM).
$selected = ($realFiles | Sort-Object Name -Descending | Select-Object -First $Count).Name

Write-Host ("Selected {0} of {1} real data chunk(s) from run '{2}':" -f $selected.Count, $realFiles.Count, $Run)
$selected | ForEach-Object { Write-Host "  $_" }

if (-not (Test-Path -LiteralPath $Destination)) {
    New-Item -ItemType Directory -Path $Destination | Out-Null
}
$Destination = (Resolve-Path -LiteralPath $Destination).Path
Write-Host "Saving to: $Destination"

foreach ($name in $selected) {
    $url = "$BaseUrl/$Run/$name"
    $outFile = Join-Path $Destination $name

    # Determine expected size from the server.
    try {
        $head = Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing
        $expected = [int64]$head.Headers['Content-Length']
    } catch {
        $expected = $null
    }

    if ((Test-Path -LiteralPath $outFile) -and $expected -ne $null -and
        ((Get-Item -LiteralPath $outFile).Length -eq $expected)) {
        Write-Host "Already complete, skipping: $name"
        continue
    }

    Write-Host "Downloading: $name"
    Start-BitsTransfer -Source $url -Destination $outFile -Description "lc0 $Run" -DisplayName $name
    Write-Host "Done: $name"
}

Write-Host "All requested downloads finished."
