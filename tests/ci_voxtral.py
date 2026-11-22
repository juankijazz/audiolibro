"""Prueba REAL de Voxtral (motor CrispASR + modelo oficial) en CI, con texto 'dificil' de un libro.
No bloquea la compilacion: deja un informe (voxtral_informe.txt) y una muestra en muestras_voz/."""
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import soundfile as sf
import audiolibro as A

TEXTO = (
    "En 1905, Albert Einstein (1879-1955) publicó cuatro artículos que cambiaron la física. "
    "«¿Qué significa E = mc²?», se preguntó el joven empleado de la oficina de patentes de Berna. "
    "Pesaba 3,5 kg al nacer; el 10 % de sus cartas se conserva. "
    "Sección 4. Capítulo II: el hombre que viajaba con un rayo de luz... "
    "—Dios no juega a los dados —dijo—. Fin."
)

informe = []
t0 = time.time()
muestras = Path("muestras_voz")
muestras.mkdir(exist_ok=True)
try:
    os.environ.setdefault("CRISPASR_ACCEPT_LICENSE", A.VX_LICENCIA)
    s = A.Sintetizador("vx:es_male", 1.0, False, lambda m: informe.append("log: " + str(m)))
    informe.append(f"motor listo en {time.time() - t0:.0f} s")
    trozos = A.trocear(A.limpiar_para_tts(TEXTO))[:8]
    partes = []
    for i, t in enumerate(trozos):
        t1 = time.time()
        a = s(t)
        informe.append(f"[{i}] {len(a) / A.SAMPLE_RATE:5.1f} s de audio en {time.time() - t1:5.0f} s | {t[:70]!r}")
        partes += [A.acondicionar(a), np.zeros(int(A.SAMPLE_RATE * 0.3), dtype="float32")]
    informe.append(f"caidas={s.caidas} vacios={s.vacios} reintentos={s.reintentos}")
    if partes:
        sf.write(str(muestras / "voxtral_es_male.wav"), np.clip(np.concatenate(partes), -1, 1),
                 A.SAMPLE_RATE, subtype="PCM_16")
    informe.append("RESULTADO: OK")
except Exception:
    informe.append("RESULTADO: FALLO\n" + traceback.format_exc())
    informe.append("--- crispasr.log (final) ---\n" + A.cola_log_crisp(3000))
(muestras / "voxtral_informe.txt").write_text("\n".join(informe), encoding="utf-8")
print("\n".join(informe))
