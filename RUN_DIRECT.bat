@echo off
REM ==========================================================================
RUN_DIRECT.bat  —  Diabetic Retinopathy Dataset Download Director
REM Directs the pipeline to acquire APTOS, Mendeley, or IDRiD.
REM Some steps require your action (credentials, browser download, auth).
REM==========================================================================

setlocal EnableExtensions DisableDelayedExpansion

cd /d "%~dp0"

echo.
echo ==========================================================================
echo  Diabetic Retinopathy Research Pipeline — Dataset Download Director
echo ==========================================================================
echo.

rem ---------- Helper: run a Python one-liner and capture exit ----------
python_check() {
  "%~dp0.venv\Scripts\python.exe" -c "$1" 2>nul
  return %errorlevel%
}

rem ---------- 1. APTOS ----------
:APTOS
echo.
echo [APTOS]  Requires Kaggle CLI + token + accepted competition rules.
echo.
echo   1. Install the `kaggle` CLI: https://github.com/Kaggle/kaggle-api
echo   2. Create token:  Account → Create New API Token → downloads kaggle.json
echo   3. Place kaggle.json at:  C:\Users\%USERNAME%\.kaggle\kaggle.json  (mode 600)
echo   4. Accept competition rules at:
echo         https://www.kaggle.com/c/aptos2019-blindness-detection/rules
echo.
if exist "C:\Users\%USERNAME%\.kaggle\kaggle.json" (
  echo   ✔ kaggle.json found.
  python_check "import kaggle; kaggle.dataset_list()">nul && (
    echo   ✔ Kaggle CLI works.
    echo.
    echo   Running: RUN.bat --download-only --datasets aptos
    echo.
    call :run_pipeline --download-only --datasets aptos
    goto end_direct
  ) || (
    echo   ✖ Kaggle CLI failed (403/401). Check token/terms acceptance.
  )
) else (
  echo   ✖ kaggle.json NOT found at the expected location.
)
echo.
set /p aptos_choice="Proceed with APTOS download anyway (y/n)? "
if /i "!aptos_choice!"=="y" (
  echo   Running: RUN.bat --download-only --datasets aptos
  call :run_pipeline --download-only --datasets aptos
) else (
  echo   Skipping APTOS.
)
goto end_direct

rem ---------- 2. MENDELEY ----------
:MENDELEY
echo.
echo [MENDELEY]  Public API reports size=0, files=[] — no anonymous download.
echo   You must obtain the corpus via browser 'Download All'.
echo.
echo   Steps:
echo   1. Open: https://data.mendeley.com/datasets/6ry9n79gb4/1
echo   2. Click 'Download All' (if enabled) and extract the archive
echo   3. Ensure 13 class folders sit directly inside:
echo         C:\Users\%USERNAME%\%DR_FUNDUS_ROOT%\
echo      Expected folders: 1.Dry AMD … 13.Very Severe NPDR
echo.
set /p mendley_path="Enter path to your Mendeley corpus (or press ENTER to skip): "
if defined mendley_path (
  echo   Running: RUN.bat --fundus-root "%mendley_path%" --download-only
  call :run_pipeline --fundus-root "%mendley_path%" --download-only
) else (
  echo   Skipping Mendeley. Provide a path when you have the corpus.
)
goto end_direct

rem ---------- 3. IDRiD ----------
:IDRID
echo.
echo [IDRiD]  Requires an authenticated IEEE DataPort session.
echo   Both official hosts require registration and terms acceptance.
echo.
echo   Options:
echo   A) Set environment variables and run:
echo      set DR_IDRID_URLS=<space-separated archive urls>
echo      set DR_IDRID_COOKIE=<cookie header value>
echo      RUN.bat --download-only --datasets idrid
echo   B) Supply an existing extracted copy:
echo      RUN.bat --idrid-root "PATH" --download-only
echo.
set /p idrid_choice="Choose option (A/B, default B): "
if /i "!idrid_choice!"=="a" (
  set /p idrid_urls="Enter DR_IDRID_URLS (space-separated): "
  set /p idrid_cookie="Enter DR_IDRID_COOKIE (press ENTER to skip): "
  if defined idrid_cookie (
    set "IDRID_COOKIE_ARGS=DR_IDRID_COOKIE=%idrid_cookie%"
  )
  echo   Running: RUN.bat --download-only --datasets idrid
  call :run_pipeline --download-only --datasets idrid
) else (
  set /p idrid_path="Enter path to your IDRiD extracted copy (or press ENTER to skip): "
  if defined idrid_path (
    echo   Running: RUN.bat --idrid-root "%idrid_path%" --download-only
    call :run_pipeline --idrid-root "%idrid_path%" --download-only
  ) else (
    echo   Skipping IDRiD.
  )
)
goto end_direct

rem ---------- Common pipeline runner ----------
:run_pipeline
REM Build the argument list, injecting env vars if present
set "ARGS=%*"
if defined IDRID_COOKIE (
  set "ARGS=!ARGS:DR_IDRID_COOKIE=%IDRID_COOKIE%"
)
echo   Executing Python pipeline with: %ARGS%
"%~dp0scripts\run_pipeline.py" %ARGS%
if errorlevel 1 (
  echo   Pipeline exited with error level %errorlevel%.
  echo   Check logs\RUN_console.log for details.
)
exit /b %errorlevel%

rem ---------- End ----------
:end_direct
echo.
echo ==========================================================================
echo  Download director finished.
echo ==========================================================================
echo.
endlocal & exit /b 0