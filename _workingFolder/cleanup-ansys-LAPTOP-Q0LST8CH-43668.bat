@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="LAPTOP-Q0LST8CH" (taskkill /f /pid 43372)
if /i "%LOCALHOST%"=="LAPTOP-Q0LST8CH" (taskkill /f /pid 52324)
if /i "%LOCALHOST%"=="LAPTOP-Q0LST8CH" (taskkill /f /pid 35928)
if /i "%LOCALHOST%"=="LAPTOP-Q0LST8CH" (taskkill /f /pid 43668)

del /F cleanup-ansys-LAPTOP-Q0LST8CH-43668.bat
