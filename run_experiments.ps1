$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=================================================================" -ForegroundColor Magenta
Write-Host "   NVIDIA CUDA & PTX KERNEL PERFORMANCE LAB (RTX 3050)          " -ForegroundColor Magenta
Write-Host "=================================================================" -ForegroundColor Magenta

Write-Host "`n>>> Step 1: Running Kernel Correctness Unit Tests <<<" -ForegroundColor Cyan
python tests/test_kernels.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "CUDA Kernel Correctness Unit Tests Failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n>>> Step 2: Running Physical GPU Microarchitecture Benchmarks <<<" -ForegroundColor Green
python src/benchmark_runner.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "GPU Benchmarks Failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[SUCCESS] All CUDA PTX Experiments and Verifications Passed!" -ForegroundColor Green
$global:LASTEXITCODE = 0
exit 0
