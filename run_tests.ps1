# run_tests.ps1

$csvFile = "results.csv"
" N, P, T_par, T_seq, Speedup, Efficiency" | Out-File -FilePath $csvFile -Encoding utf8

$Ns = @(128, 256, 512, 1024)
$Ps = @(1, 4, 9, 16, 25)

foreach ($N in $Ns) {
    foreach ($P in $Ps) {
        $p_root = [Math]::Sqrt($P)
        if ($N % $p_root -ne 0) {
            Write-Host "Skipping N=$N, P=$P (N not divisible by $p_root)" -ForegroundColor DarkGray
            continue
        }

        Write-Host "Running: N=$N, P=$P" -ForegroundColor Cyan
        $output = mpiexec -n $P .\task3.exe $N 2>$null
        $output | Out-File -FilePath $csvFile -Append -Encoding utf8
    }
}

Write-Host "All tests completed. Results saved to $csvFile" -ForegroundColor Green