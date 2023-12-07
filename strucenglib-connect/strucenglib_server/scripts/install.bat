@echo off
@echo on
set cenv=strucenglib_connect

call conda activate %cenv%
call python -m compas_rhino.install -v 7.0
call pip install https://github.com/kfmResearch-NumericsTeam/Struc_Eng_Library_Server/archive/feature/master.zip#subdirectory=strucenglib_connect
call python -m compas_rhino.install -v 7.0 -p strucenglib_connect
pause
