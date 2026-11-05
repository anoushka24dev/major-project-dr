@echo off
REM ==========================================================================
REM  Diabetic Retinopathy Research Pipeline - one-click launcher
REM
REM  Double-click this file to run the complete pipeline:
REM    environment check - dataset download - prepare - train ResNet-50
REM    - train lesion U-Net - Grad-CAM - evaluation - calibration
REM    - cross-dataset evaluation - final report
REM
REM  All arguments are forwarded unchanged to the Python launcher script.
REM
REM    RUN.bat                        full pipeline (default)
REM    RUN.bat --resume               resume interrupted training
REM    RUN.bat --download-only        download + verify every missing dataset
REM    RUN.bat --download-only --datasets mmrdr ddr   only those sources
REM    RUN.bat --prepare-only         manifests + quality only
REM    RUN.bat --train-only           training only
REM    RUN.bat --train-only --cv-folds 5   grouped stratified 5-fold CV,
REM                                  out-of-fold metrics, then a final
REM                                  retrain on the development pool
REM    RUN.bat --evaluate-only        evaluation + report only
REM    RUN.bat --report-only          rebuild the report from existing outputs
REM    RUN.bat --force-retrain        ignore existing checkpoints
REM    RUN.bat --skip-download        do not transfer bytes
REM    RUN.bat --fundus-root "PATH"   use an existing Mendeley corpus
REM    RUN.bat --idrid-root "PATH"    use an existing IDRiD copy
REM    RUN.bat --aptos-root "PATH"    use an existing APTOS 2019 copy
REM    RUN.bat --ddr-root "PATH"      use an existing DDR copy
REM
REM  Dataset composition modes (whole pipeline, one classifier):
REM    RUN.bat --mode combined        train on every validated source
REM                                  (Mendeley + MMRDR + APTOS + DDR)
REM    RUN.bat --mode external        train on Mendeley + MMRDR and evaluate
REM                                  on the held-out APTOS and DDR sources
REM    RUN.bat --experiments all      run the ResNet-50 experiment series E0..E6
REM                                  (a subset is also accepted, e.g. "E0 E4")
REM    RUN.bat --mode combined --experiments all
REM    RUN.bat --mode external --experiments all
REM    RUN.bat --config "PATH"        use an alternative config.yaml
REM    RUN.bat --no-setup             skip the environment setup step
REM    RUN.bat --tests                run the test suite instead
REM    RUN.bat --help                 show the Python-level usage
REM
REM  set DR_PYTHON=C:\Python311\python.exe   force a specific interpreter
REM
REM  Exit codes: 0 success | 1 a stage failed | 2 configuration/environment
REM              3 data blocked pending manual dataset access
REM ==========================================================================

setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Diabetic Retinopathy Research Pipeline

set "SKIP_SETUP=0"

echo.
echo ==========================================================================
echo  Diabetic Retinopathy Research Pipeline
echo  folder: %~dp0
echo ==========================================================================
echo.

echo [1/4] Locating Python...
set "BASE_PY="
if defined DR_PYTHON set "BASE_PY=%DR_PYTHON%"
if defined BASE_PY goto python_found
for /f "delims=" %%P in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "BASE_PY=%%P"
if defined BASE_PY goto python_found
for /f "delims=" %%P in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "BASE_PY=%%P"
if defined BASE_PY goto python_found
goto no_python

:no_python
echo.
echo ERROR: Python was not found on this computer.
echo.
echo   1. Install 64-bit Python 3.11 or 3.12 from https://www.python.org/downloads/
echo   2. During installation tick "Add python.exe to PATH".
echo   3. Run RUN.bat again.
echo.
call :maybe_pause
exit /b 2

:python_found
echo       interpreter: %BASE_PY%
"%BASE_PY%" -c "import sys;print('      version     :',sys.version.split()[0]);print('      architecture:',('64-bit' if sys.maxsize>2**32 else '32-bit - NOT SUPPORTED'))"

REM Detect --no-setup in Python so quoted arguments cannot break the batch parser.
"%BASE_PY%" -c "import sys;sys.exit(0 if '--no-setup' in sys.argv else 1)" %*
if not errorlevel 1 set "SKIP_SETUP=1"

if "%SKIP_SETUP%"=="1" goto skip_setup
echo.
echo [2/4] Preparing the Python environment...
echo       The first run creates .venv and installs dependencies; this can take
echo       several minutes. Later runs reuse it.
echo.
"%BASE_PY%" "%~dp0scripts\setup_env.py"
if errorlevel 1 goto setup_failed
goto setup_done

:setup_failed
echo.
echo ERROR: The Python environment could not be prepared.
echo        Read the messages above. Common causes:
echo          - no internet connection during the first run
echo          - a 32-bit or very old Python installation
echo          - insufficient disk space for PyTorch and the datasets
echo        The pipeline was NOT started.
echo.
call :maybe_pause
exit /b 2

:skip_setup
echo.
echo [2/4] Environment setup skipped (--no-setup).

:setup_done
set "RUN_PY=%BASE_PY%"
if exist "%~dp0.venv\Scripts\python.exe" set "RUN_PY=%~dp0.venv\Scripts\python.exe"

echo.
echo [3/4] Running the pipeline...
echo       A complete log is written to logs\RUN_console.log
echo.
"%RUN_PY%" "%~dp0scripts\run_pipeline.py" %*
set "RC=%ERRORLEVEL%"

echo.
echo [4/4] Finished with exit code %RC%.
if "%RC%"=="0" echo        Success. See outputs\reports\final_report.html
if "%RC%"=="1" echo        A pipeline stage failed. The failing stage and its error are listed
if "%RC%"=="1" echo        above and in outputs\reports\execution_summary.json.
if "%RC%"=="2" echo        Configuration or environment problem. Fix it and run RUN.bat again.
if "%RC%"=="3" echo        One or more datasets are blocked pending your action. The exact step
if "%RC%"=="3" echo        is printed above and saved in outputs\manifests\download_manifest.json.
if "%RC%"=="3" echo        Provide the data, then run:  RUN.bat --resume

echo.
echo Results : outputs\reports\final_report.html
echo Metrics : outputs\evaluation\metrics_summary.csv
if "%DR_NO_OPEN%"=="1" goto skip_open
if not exist "%~dp0outputs\reports\final_report.html" goto skip_open
echo.
echo Opening the report...
start "" "%~dp0outputs\reports\final_report.html"
:skip_open

echo.
echo This window is kept open so you can read the result.
call :maybe_pause
endlocal & exit /b %RC%

REM ---------------------------------------------------------------- helpers
:maybe_pause
if "%DR_NO_PAUSE%"=="1" goto :eof
pause
goto :eof
