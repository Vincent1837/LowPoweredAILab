@echo off
.\tools\python-3.8.10-embed-amd64\python.exe .\tools\get-pip.py
.\tools\python-3.8.10-embed-amd64\python.exe .\tools\init.py
if %errorlevel% equ 0 (
    .\tools\python-3.8.10-embed-amd64\python.exe .\tools\update.py
	.\tools\python-3.8.10-embed-amd64\python.exe .\vital-sign-detection-app\update.py
)
echo.
if %errorlevel% equ 0 (
    echo All done!
) else (
    echo Update failed
)
echo.
pause
