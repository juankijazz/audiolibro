@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Instalador de Audiolibro
echo ==================================================
echo   Instalador de Audiolibro  (EPUB / PDF / TXT a audio)
echo ==================================================
echo.

rem ---- 1. Buscar Python 3.10 - 3.12 con tkinter -------------------------
set "PY="
for %%V in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import tkinter, venv" >nul 2>&1
    if not errorlevel 1 set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys,tkinter,venv; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  echo [ERROR] No se ha encontrado Python 3.10, 3.11 o 3.12 con tkinter.
  echo         Instala Python 3.12 desde https://www.python.org/downloads/
  echo         y marca la casilla "Add python.exe to PATH".
  pause
  exit /b 1
)
echo Python encontrado: %PY%

rem ---- 2. Entorno virtual propio (no toca tu Python) --------------------
if not exist "venv\Scripts\python.exe" (
  echo Creando entorno virtual...
  %PY% -m venv venv
  if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual.
    pause
    exit /b 1
  )
)
set "VPY=%~dp0venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip

rem ---- 3. PyTorch: CUDA 12.8 si hay GPU NVIDIA (necesario para RTX 50xx) -
nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo No se detecta GPU NVIDIA: se usara PyTorch para CPU.
  "%VPY%" -c "import torch" >nul 2>&1
  if errorlevel 1 "%VPY%" -m pip install torch
) else (
  "%VPY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
  if errorlevel 1 (
    echo GPU NVIDIA detectada. Instalando PyTorch con CUDA 12.8 ^(descarga de varios GB^)...
    "%VPY%" -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
    "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 (
      echo [ERROR] Fallo la instalacion de PyTorch con CUDA.
      pause
      exit /b 1
    )
  ) else (
    echo PyTorch con CUDA ya instalado.
  )
)

rem ---- 4. Resto de dependencias (espeak y ffmpeg van incluidos en pip) ---
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Fallo la instalacion de dependencias.
  pause
  exit /b 1
)

rem ---- 5. Descargar modelo y voces y probar la GPU ----------------------
echo.
echo Descargando el modelo y comprobando la GPU...
"%VPY%" audiolibro.py --diagnostico
if errorlevel 1 (
  echo [AVISO] El diagnostico dio error. Copia el mensaje de arriba para pedir ayuda.
)

rem ---- 6. Acceso directo en el escritorio -------------------------------
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Audiolibro.lnk'); $s.TargetPath='%~dp0venv\Scripts\pythonw.exe'; $s.Arguments='\"%~dp0audiolibro.py\"'; $s.WorkingDirectory='%~dp0'; $s.Save()" >nul 2>&1

echo.
echo ==================================================
echo  Instalacion terminada.
echo  Abre "Audiolibro" desde el escritorio o con AUDIOLIBRO.bat
echo ==================================================
pause
