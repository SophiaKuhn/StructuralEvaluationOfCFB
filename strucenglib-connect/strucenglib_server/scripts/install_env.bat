@echo off
echo This installs venv for struceng lib connect server component

@echo on
set cenv=strucenglib_connect

call conda create -n %cenv% -c conda-forge python=3.9 compas --yes
call conda activate %cenv%
pause
