# -*- coding: utf-8 -*-
"""
Audiolibro.exe - lanzador / instalador.

Primera ejecucion: instala en %LOCALAPPDATA%\\Audiolibro todo lo necesario
(uv + Python 3.12 + PyTorch con CUDA + dependencias + modelo) y abre la app.
Siguientes ejecuciones: abre la app directamente.

  Audiolibro.exe             lanza (instala si falta algo)
  Audiolibro.exe --reparar   fuerza reinstalar dependencias
  Audiolibro.exe --sin-gui   instala en modo consola/log, sin abrir nada (CI)
"""
import hashlib
import os
import queue
import shutil
import subprocess
import sys
import threading
import traceback
import urllib.request
import zipfile
from pathlib import Path

NOMBRE = "Audiolibro"
SETUP_VERSION = "1"
INSTALL = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / NOMBRE
APP = INSTALL / "app"
VENV = INSTALL / "venv"
UVDIR = INSTALL / "uv"
UV = UVDIR / "uv.exe"
VPY = VENV / "Scripts" / "python.exe"
VPYW = VENV / "Scripts" / "pythonw.exe"
FLAG = INSTALL / "ok.flag"
LOGF = INSTALL / "setup.log"
UV_URL = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
NOWIN = 0x08000000
DETACHED = 0x00000008 | 0x00000200 | NOWIN


def recurso(nombre):
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / nombre


def firma():
    req = recurso("requirements.txt").read_text(encoding="utf-8")
    return hashlib.sha1((SETUP_VERSION + req).encode()).hexdigest()[:12]


def copiar_app():
    APP.mkdir(parents=True, exist_ok=True)
    for f in ("audiolibro.py", "requirements.txt"):
        shutil.copy2(recurso(f), APP / f)


def instalado():
    try:
        return VPYW.exists() and FLAG.read_text().strip() == firma()
    except OSError:
        return False


def lanzar():
    subprocess.Popen([str(VPYW), str(APP / "audiolibro.py")], cwd=str(APP),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=DETACHED, close_fds=True)


class Instalador:
    def __init__(self, log):
        self.log = log
        self.env = dict(os.environ, PYTHONUTF8="1", UV_LINK_MODE="copy", UV_NO_PROGRESS="1",
                        UV_PYTHON_INSTALL_DIR=str(INSTALL / "python"),
                        UV_CACHE_DIR=str(INSTALL / "uv-cache"))

    def ejecutar(self, cmd):
        self.log("> " + " ".join(str(c) for c in cmd))
        p = subprocess.Popen([str(c) for c in cmd], stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", env=self.env,
                             creationflags=NOWIN)
        for linea in p.stdout:
            linea = linea.rstrip()
            if linea:
                self.log(linea)
        if p.wait() != 0:
            raise RuntimeError(f"Fallo (codigo {p.returncode}): {' '.join(str(c) for c in cmd)[:200]}")

    def hay_gpu_nvidia(self):
        try:
            return subprocess.run(["nvidia-smi"], stdin=subprocess.DEVNULL, capture_output=True,
                                  creationflags=NOWIN).returncode == 0
        except OSError:
            return False

    def instalar(self):
        INSTALL.mkdir(parents=True, exist_ok=True)
        copiar_app()
        if not UV.exists():
            self.log("Descargando uv (gestor de Python)...")
            UVDIR.mkdir(parents=True, exist_ok=True)
            zpath = UVDIR / "uv.zip"
            urllib.request.urlretrieve(UV_URL, zpath)
            with zipfile.ZipFile(zpath) as z:
                for n in z.namelist():
                    if n.lower().endswith("uv.exe"):
                        (UVDIR / "uv.exe").write_bytes(z.read(n))
            zpath.unlink(missing_ok=True)
        if not VPY.exists():
            self.log("Creando Python 3.12 aislado (no toca tu Python)...")
            self.ejecutar([UV, "venv", "--python", "3.12", "--managed-python", VENV])
        gpu = self.hay_gpu_nvidia()
        if gpu:
            self.log("GPU NVIDIA detectada: instalando PyTorch con CUDA 12.8 (descarga grande, paciencia)...")
            self.ejecutar([UV, "pip", "install", "--python", VPY, "torch",
                           "--index-url", "https://download.pytorch.org/whl/cu128"])
        else:
            self.log("No se detecta GPU NVIDIA: se instala PyTorch para CPU.")
            self.ejecutar([UV, "pip", "install", "--python", VPY, "torch"])
        self.log("Instalando dependencias...")
        self.ejecutar([UV, "pip", "install", "--python", VPY, "-r", APP / "requirements.txt"])
        if gpu:
            r = subprocess.run([str(VPY), "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                               stdin=subprocess.DEVNULL, capture_output=True, creationflags=NOWIN)
            self.log("CUDA operativo en PyTorch: OK" if r.returncode == 0 else
                     "AVISO: hay GPU NVIDIA pero PyTorch no la ve; se usara la CPU (actualiza el driver de NVIDIA).")
        self.log("Descargando el modelo de voz (una sola vez)...")
        self.ejecutar([VPY, APP / "audiolibro.py", "--precargar"])
        FLAG.write_text(firma())
        self.log("Instalacion terminada.")


def log_a_archivo(s):
    try:
        INSTALL.mkdir(parents=True, exist_ok=True)
        with open(LOGF, "a", encoding="utf-8") as f:
            f.write(s + "\n")
    except OSError:
        pass


def modo_consola():
    def log(s):
        log_a_archivo(s)
        try:
            print(s, flush=True)
        except Exception:
            pass
    try:
        Instalador(log).instalar()
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


def modo_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox
    root = tk.Tk()
    root.title("Audiolibro - instalacion")
    root.geometry("640x380")
    ttk.Label(root, text="Preparando Audiolibro (solo la primera vez)",
              font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=12, pady=(12, 0))
    ttk.Label(root, text="Se descarga e instala todo lo necesario (varios GB con GPU). No cierres esta ventana.",
              wraplength=600).pack(anchor="w", padx=12)
    pb = ttk.Progressbar(root, mode="indeterminate")
    pb.pack(fill="x", padx=12, pady=8)
    pb.start(12)
    txt = tk.Text(root, height=14, font=("Consolas", 9), state="disabled", wrap="word")
    txt.pack(fill="both", expand=True, padx=12, pady=(0, 12))
    q = queue.Queue()

    def log(s):
        log_a_archivo(s)
        q.put(("log", s))

    def trabajo():
        try:
            Instalador(log).instalar()
            q.put(("ok", ""))
        except Exception:
            q.put(("err", traceback.format_exc()))

    def poll():
        try:
            while True:
                k, v = q.get_nowait()
                if k == "log":
                    txt.config(state="normal")
                    txt.insert("end", v + "\n")
                    txt.see("end")
                    txt.config(state="disabled")
                elif k == "ok":
                    lanzar()
                    root.destroy()
                    return
                else:
                    log_a_archivo(v)
                    messagebox.showerror("Audiolibro",
                                         "La instalacion ha fallado.\n\n" + v[-700:] +
                                         f"\n\nRegistro completo: {LOGF}")
                    root.destroy()
                    return
        except queue.Empty:
            pass
        root.after(150, poll)

    threading.Thread(target=trabajo, daemon=True).start()
    root.after(150, poll)
    root.mainloop()


def main():
    args = sys.argv[1:]
    if "--reparar" in args:
        FLAG.unlink(missing_ok=True)
    if "--sin-gui" in args:
        return modo_consola()
    if instalado():
        copiar_app()
        lanzar()
        return 0
    modo_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
