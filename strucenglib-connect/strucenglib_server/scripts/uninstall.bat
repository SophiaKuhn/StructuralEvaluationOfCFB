@echo off
@echo on
set cenv=strucenglib_connect

call conda activate %cenv%
call python -m compas_rhino.uninstall -v 7.0 -p strucenglib_connect
call pip uninstall -y strucenglib_connect

pause
