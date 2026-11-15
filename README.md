# Audiolibro

Convierte **EPUB, PDF o TXT** en audiolibro (**M4B con capítulos** o MP3) con [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), usando la GPU NVIDIA.

## Usar el .exe
1. Descarga `Audiolibro.exe` de la sección **Releases** (`latest`).
2. Doble clic. La primera vez instala en `%LOCALAPPDATA%\Audiolibro` todo lo necesario (Python aislado, PyTorch con CUDA 12.8, dependencias y modelo) y abre la aplicación. Después abre directamente.
3. Windows SmartScreen puede avisar porque el .exe no está firmado: "Más información" → "Ejecutar de todos modos".

`Audiolibro.exe --reparar` reinstala las dependencias. El registro de instalación está en `%LOCALAPPDATA%\Audiolibro\setup.log`.

## Sin el .exe
`INSTALAR.bat` (con Python 3.10–3.12) crea un entorno local y un acceso directo; `AUDIOLIBRO.bat` abre la aplicación.

## Consola
```
python audiolibro.py libro.epub --listar
python audiolibro.py libro.epub --voz em_alex --velocidad 1.1 --formato mp3
python audiolibro.py --diagnostico      # comprueba la GPU y la velocidad real
```

## Integración continua
Cada push ejecuta en Windows: pruebas, síntesis real con Kokoro (CPU), compilación del exe, prueba de extremo a extremo del exe y publicación en la release `latest`. El artefacto `muestra-voz-real` contiene el audio de prueba generado.
