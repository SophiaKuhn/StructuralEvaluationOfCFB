@echo off
echo This runs strucenglib connect server component

@echo on
set cenv=strucenglib_connect

call conda activate %cenv%
call strucenglib_server
pause
