@echo off
echo FedAIR Quick Test Runner
echo.
set PYTHON_PATH=C:\Users\zayan\Downloads\MlOps_Project\NLP_porj\python.exe
echo Using Python: %PYTHON_PATH%
echo.
echo Running quick test...
echo.
%PYTHON_PATH% run_quick_test.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Quick test completed successfully!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Quick test failed. Check errors above.
    echo ========================================
)
pause

