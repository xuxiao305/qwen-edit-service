# SSH port-forward helper for QwenEditService.
# Forwards local 127.0.0.1:8765 to the GPU host's 127.0.0.1:8765.
#
# Edit the variables below for your host, then run:
#   .\deploy\ssh_tunnel.ps1
#
# The script blocks; keep this terminal open while you call the API.

param(
    [string]$Key = "C:\tmp\DanLu_key",
    [string]$User = "root",
    [string]$Host = "apps-sl.danlu.netease.com",
    [int]$Port = 44304,
    [int]$LocalPort = 8765,
    [int]$RemotePort = 8765
)

if (-not (Test-Path $Key)) {
    Write-Error "SSH key not found: $Key"
    exit 1
}

Write-Host "Forwarding 127.0.0.1:$LocalPort -> ${Host}:$RemotePort (via SSH ${User}@${Host}:${Port})"
ssh -i $Key -p $Port `
    -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=NUL `
    -o ServerAliveInterval=30 `
    -o ExitOnForwardFailure=yes `
    -L "${LocalPort}:127.0.0.1:${RemotePort}" `
    -N "${User}@${Host}"
