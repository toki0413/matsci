<#
.SYNOPSIS
    Bump project version across all config files.
.DESCRIPTION
    Updates version in Cargo.toml (workspace), desktop/src-tauri/Cargo.toml,
    desktop/package.json, and desktop/src-tauri/tauri.conf.json in one shot.
.PARAMETER Version
    Semantic version string (e.g. "0.2.0", "1.0.0-rc.1").
.EXAMPLE
    .\scripts\bump_version.ps1 0.2.0
#>
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^\d+\.\d+\.\d+')]
    [string]$Version
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$files = @(
    @{ Path = "$root\Cargo.toml";                    Pattern = 'version = "0.1.0"' }
    @{ Path = "$root\desktop\src-tauri\Cargo.toml";  Pattern = 'version = "0.1.0"' }
    @{ Path = "$root\desktop\src-tauri\tauri.conf.json"; Pattern = '"version": "0.1.0"' }
    @{ Path = "$root\desktop\package.json";           Pattern = '"version": "0.1.0"' }
)

foreach ($f in $files) {
    if (-not (Test-Path $f.Path)) { Write-Warning "Skip (missing): $($f.Path)"; continue }
    $content = Get-Content $f.Path -Raw
    # match any version, not just 0.1.0 — so it works on subsequent bumps
    $updated = $content -replace $f.Pattern, ($f.Pattern -replace '0\.1\.0', $Version)
    if ($content -eq $updated) {
        # fallback: regex replace any semver-looking string
        $updated = $content -replace '(version\s*=?\s*"?)\d+\.\d+\.\d+', "`${1}$Version"
    }
    Set-Content $f.Path -Value $updated -NoNewline -Encoding UTF8
    Write-Output "Updated $($f.Path) -> $Version"
}

Write-Output "All version files synced to $Version"
Write-Output "Next: git add -A && git commit -m `"chore: bump version to $Version`" && git tag v$Version && git push --tags"
