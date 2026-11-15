"""Pruebas del pipeline con voz simulada (no necesitan GPU ni modelo)."""
import re
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tests"))
import audiolibro as A          # noqa: E402
import make_samples             # noqa: E402


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(make_samples.crear(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_epub_capitulos_y_seleccion(self):
        libro = A.leer_epub(self.d / "prueba.epub")
        self.assertEqual(libro.titulo, "El Hidalgo de Prueba")
        self.assertEqual(libro.autor, "Miguel de Prueba")
        marcados = [c.titulo for c in libro.capitulos if c.seleccionado]
        self.assertEqual(len(marcados), 3)
        self.assertNotIn("Copyright", marcados)
        self.assertNotIn("Portada", marcados)

    def test_pdf_limpieza(self):
        libro = A.leer_pdf(self.d / "prueba.pdf")
        self.assertEqual(len(libro.capitulos), 3)
        texto = " ".join(" ".join(c.parrafos) for c in libro.capitulos)
        self.assertNotIn("HIDALGO DE PRUEBA", texto)     # cabecera repetida eliminada
        self.assertIn("extrajo", texto)                  # guion de fin de linea unido
        self.assertNotEqual(libro.capitulos[0].parrafos[0], "Capitulo 1")   # titulo no duplicado

    def test_pdf_escaneado(self):
        with self.assertRaises(ValueError):
            A.leer_pdf(self.d / "escaneado.pdf")

    def test_txt(self):
        libro = A.leer_txt(self.d / "prueba.txt")
        self.assertEqual([c.titulo for c in libro.capitulos], ["Capitulo 1", "Capitulo 2"])

    def test_limpieza_dialogos(self):
        t = A.limpiar_para_tts("—¿Sabe usted, Sr. García, qué hora es? —preguntó el hidalgo—. No tengo reloj.")
        self.assertEqual(t, "¿Sabe usted, Sr. García, qué hora es? preguntó el hidalgo. No tengo reloj.")

    def test_trozos_no_superan_limite_ni_parten_abreviaturas(self):
        libro = A.leer_epub(self.d / "prueba.epub")
        for c in libro.capitulos:
            for k, v in A.plan_capitulo(c, True):
                if k == "t":
                    self.assertLessEqual(len(v), A.MAX_CHARS)
        trozos = A.trocear("Dijo el Sr. García que vendría. Luego se fue. ¿Y D. Quijote? Nadie sabe.", 25)
        self.assertFalse(any(t.endswith("Sr.") or t.endswith("D.") for t in trozos))

    def test_m4b_con_capitulos_reanudar_y_cancelar(self):
        libro = A.leer_epub(self.d / "prueba.epub")
        salida = self.d / "salida"
        cfg = A.Config(salida=str(salida), falso=True)
        cancel = threading.Event()

        def prog(fr, m):
            if "Capitulo 2/3" in m and fr > 0.45:
                cancel.set()
        with self.assertRaises(A.Cancelado):
            A.convertir(libro, cfg, lambda s: None, prog, cancel)
        log = []
        dest = A.convertir(libro, cfg, log.append, None, threading.Event())
        self.assertTrue(any("reutiliza" in l for l in log))
        self.assertTrue(dest.exists() and dest.suffix == ".m4b")
        r = subprocess.run([A._ffmpeg(), "-hide_banner", "-i", str(dest)], capture_output=True,
                           text=True, errors="replace")
        self.assertEqual(len(re.findall(r"Chapter #0:\d", r.stderr)), 3)
        self.assertFalse((salida / "El Hidalgo de Prueba_temporal").exists())

    def test_mp3_por_capitulo(self):
        libro = A.leer_txt(self.d / "prueba.txt")
        cfg = A.Config(salida=str(self.d / "mp3s"), formato="mp3-caps", falso=True)
        carpeta = A.convertir(libro, cfg, lambda s: None)
        self.assertEqual(len(list(Path(carpeta).glob("*.mp3"))), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
