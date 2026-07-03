@echo off
echo FedAIR Training Runner
echo.
set PYTHON_PATH=C:\Users\zayan\Downloads\MlOps_Project\NLP_porj\python.exe
echo Using Python: %PYTHON_PATH%
echo.
echo Running full federated training...
echo This may take a while depending on configuration.
echo.
%PYTHON_PATH% run_training.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Training completed successfully!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Training failed. Check errors above.
    echo ========================================
)
pause

