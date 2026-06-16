@echo off
cd /d "%~dp0"
echo Pushing to GitHub...
git push origin main
if %errorlevel% equ 0 (
    echo.
    echo Push successful!
) else (
    echo.
    echo Push failed! Check your GitHub credentials.
)
pause
