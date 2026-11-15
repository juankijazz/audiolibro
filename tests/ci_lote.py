"""Prueba REAL (pesos de Chatterbox, CPU) del modo por lotes: 3 trozos de distinta longitud a la vez."""
import os
import sys
import time
from pathlib import Path

os.environ["AUDIOLIBRO_LOTE"] = "3"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import audiolibro as A  # noqa: E402

textos = ["Hola, buenos días.",
          "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo.",
          "Zaragoza, cielo y gracias: así suena el castellano de España."]
s = A.Sintetizador("cb:f1", 1.0, False, print)
assert s.lote_max == 3, s.lote_max
t = time.time()
audios = s.lote(textos)
print(f"lote de 3: {time.time() - t:.0f} s, repetidos: {s.reintentos}")
for tx, a in zip(textos, audios):
    dur = len(a) / A.SAMPLE_RATE
    esperado = max(1.0, len(tx) / 15.0)
    print(f"{len(tx):3d} caracteres -> {dur:.1f} s (esperado ~{esperado:.1f} s)")
    assert 0.4 * esperado - 0.5 <= dur <= 2.0 * esperado + 1.0, "duracion implausible"
print("OK: modo por lotes con pesos reales")
