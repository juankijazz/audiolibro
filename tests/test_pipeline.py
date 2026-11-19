"""Pruebas del pipeline con voz simulada (no necesitan GPU ni modelo)."""
import os
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

    def test_salida_relativa(self):
        libro = A.leer_txt(self.d / "prueba.txt")
        antes = os.getcwd()
        os.chdir(self.d)
        try:
            dest = A.convertir(libro, A.Config(salida="rel_salida", falso=True), lambda s: None)
        finally:
            os.chdir(antes)
        self.assertTrue(Path(dest).exists())

    def test_mp3_por_capitulo(self):
        libro = A.leer_txt(self.d / "prueba.txt")
        cfg = A.Config(salida=str(self.d / "mp3s"), formato="mp3-caps", falso=True)
        carpeta = A.convertir(libro, cfg, lambda s: None)
        self.assertEqual(len(list(Path(carpeta).glob("*.mp3"))), 2)


    def test_motores_y_voces(self):
        self.assertEqual(A.motor_de("ef_dora"), "kokoro")
        self.assertEqual(A.motor_de("cb:f1"), "chatterbox")
        self.assertIn("cb:propia", A.CB_VOCES)
        with self.assertRaises(ValueError):
            A.referencia_cb("cb:propia", "", lambda s: None)
        with self.assertRaises(ValueError):
            A.referencia_cb("cb:inexistente", "", lambda s: None)

    def test_chatterbox_falso_con_velocidad_y_normalizado(self):
        libro = A.leer_txt(self.d / "prueba.txt")
        cfg = A.Config(voz="cb:f1", velocidad=1.2, salida=str(self.d / "cb_falso"), falso=True)
        dest = A.convertir(libro, cfg, lambda s: None)
        self.assertTrue(Path(dest).exists())
        af, escala = A._filtro_audio(cfg)
        self.assertIn("atempo=1.200", af[1])
        self.assertIn("loudnorm", af[1])
        self.assertAlmostEqual(escala, 1 / 1.2)
        af2, esc2 = A._filtro_audio(A.Config(voz="ef_dora", velocidad=1.2))
        self.assertNotIn("atempo", af2[1])
        self.assertEqual(esc2, 1.0)

    def test_acondicionar(self):
        import numpy as np
        sr = A.SAMPLE_RATE
        voz = (0.3 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype("float32")
        a = np.concatenate([np.zeros(sr, "float32"), voz, np.zeros(sr, "float32")]) + 0.05
        r = A.acondicionar(a)
        self.assertLess(len(r), len(a) * 0.5)                  # silencio recortado
        self.assertAlmostEqual(float(r[:5].max()), 0.0, delta=0.02)   # entrada suave (sin clic)
        self.assertLess(abs(float(np.mean(r))), 0.01)           # sin componente continua
        self.assertEqual(len(A.acondicionar(np.zeros(1000, "float32"))), 0)

    def test_pausas_segun_puntuacion(self):
        cap = A.Capitulo("T", ["Primera frase corta. Segunda frase, con coma"], True)
        pl = A.plan_capitulo(cap, False)
        pausas = [v for k, v in pl if k == "s"]
        self.assertTrue(all(0.1 <= p <= 1.5 for p in pausas))

    def test_voxtral_via_servidor_unsloth_falso(self):
        """Voxtral por la API /v1/audio/speech de Unsloth (servidor simulado)."""
        import http.server
        import io
        import json as _json
        import numpy as np
        import soundfile as sf
        pedidos = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if "voices" in self.path:
                    obj = {"data": [{"id": "casual_male"}, {"id": "es_female", "language": "es"}]}
                else:
                    obj = {"data": [{"id": "otro"}, {"id": "Voxtral-4B-TTS-2603-GGUF"}]}
                cuerpo = _json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(cuerpo)

            def do_POST(self):
                n = int(self.headers["Content-Length"])
                pedidos.append((self.headers.get("Authorization"), _json.loads(self.rfile.read(n))))
                t = pedidos[-1][1]["input"]
                dur = max(1.0, len(t) / 15.0)
                x = (0.2 * np.sin(2 * np.pi * 220 * np.arange(int(16000 * dur)) / 16000)).astype("float32")
                b = io.BytesIO()
                sf.write(b, x, 16000, format="WAV")
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                self.wfile.write(b.getvalue())

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        old = {k: os.environ.get(k) for k in ("AUDIOLIBRO_VOXTRAL_URL", "AUDIOLIBRO_DATOS", "UNSLOTH_API_KEY",
                                              "AUDIOLIBRO_SIN_CRISPASR")}
        with tempfile.TemporaryDirectory() as d:
            os.environ.update(AUDIOLIBRO_VOXTRAL_URL=f"http://127.0.0.1:{srv.server_port}",
                              AUDIOLIBRO_DATOS=d, UNSLOTH_API_KEY="sk-unsloth-prueba",
                              AUDIOLIBRO_SIN_CRISPASR="1")
            try:
                self.assertEqual(A.motor_de("vx:es_female"), "voxtral")
                A.guardar_ajustes(vx_modelo=None)
                s = A.Sintetizador("vx:es_female", 1.0, True, lambda *_: None)
                a = s("Hola, esto es una prueba de la voz de Voxtral en español.")
                self.assertEqual(s.modelo, "Voxtral-4B-TTS-2603-GGUF")
                self.assertGreater(len(a) / A.SAMPLE_RATE, 2.0)          # remuestreado a 24 kHz
                self.assertEqual(pedidos[0][0], "Bearer sk-unsloth-prueba")
                self.assertEqual(pedidos[0][1]["voice"], "es_female")
                self.assertEqual(pedidos[0][1]["language"], "Spanish")
                self.assertEqual(pedidos[0][1]["audio_options"], {"voice_id": "es_female"})
                # una voz que no existe en el servidor se sustituye por la española disponible
                s2 = A.Sintetizador("vx:es_male", 1.0, True, lambda *_: None)
                self.assertEqual(s2.nombre_voz, "es_female")
                self.assertEqual([v for v, _ in A.voces_servidor(s.url, s.clave)], ["casual_male", "es_female"])
                self.assertTrue(A.es_voz_espanola("es_female") and not A.es_voz_espanola("casual_male"))
                self.assertEqual(pedidos[0][1]["response_format"], "wav")
            finally:
                srv.shutdown()
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def test_voxtral_via_crispasr_falso(self):
        """Voxtral con el servidor CrispASR (simulado): la voz va tal cual en 'voice'."""
        import http.server
        import io
        import json as _json
        import numpy as np
        import soundfile as sf
        pedidos = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers["Content-Length"])
                pedidos.append(_json.loads(self.rfile.read(n)))
                dur = max(1.0, len(pedidos[-1]["input"]) / 15.0)
                x = (0.2 * np.sin(2 * np.pi * 220 * np.arange(int(24000 * dur)) / 24000)).astype("float32")
                b = io.BytesIO()
                sf.write(b, x, 24000, format="WAV")
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                self.wfile.write(b.getvalue())

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        old = os.environ.get("AUDIOLIBRO_CRISPASR_URL")
        os.environ["AUDIOLIBRO_CRISPASR_URL"] = f"http://127.0.0.1:{srv.server_port}"
        try:
            s = A.Sintetizador("vx:es_male", 1.0, True, lambda *_: None)
            a = s("Hola, esto es una prueba de la voz en español de Voxtral.")
            self.assertEqual(s.modo, "crisp")
            self.assertGreater(len(a) / A.SAMPLE_RATE, 2.0)
            self.assertEqual((pedidos[0]["voice"], pedidos[0]["language"]), ("es_male", "es"))
            self.assertIn("vx:es_female", A.VX_VOCES)
        finally:
            srv.shutdown()
            if old is None:
                os.environ.pop("AUDIOLIBRO_CRISPASR_URL", None)
            else:
                os.environ["AUDIOLIBRO_CRISPASR_URL"] = old

    def test_licencia_voxtral(self):
        old = {k: os.environ.get(k) for k in ("AUDIOLIBRO_DATOS", "CRISPASR_ACCEPT_LICENSE", "AUDIOLIBRO_SIN_CRISPASR")}
        with tempfile.TemporaryDirectory() as d:
            os.environ.update(AUDIOLIBRO_DATOS=d)
            os.environ.pop("CRISPASR_ACCEPT_LICENSE", None)
            os.environ.pop("AUDIOLIBRO_SIN_CRISPASR", None)
            try:
                self.assertFalse(A.licencia_voxtral_aceptada())
                with self.assertRaises(RuntimeError) as cm:
                    A.arrancar_crispasr(False)
                self.assertIn("CC BY-NC", str(cm.exception))
                A.guardar_ajustes(vx_licencia=A.VX_LICENCIA)
                self.assertTrue(A.licencia_voxtral_aceptada())
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def test_cuerpo_voxtral_segun_esquema(self):
        c = A.cuerpo_voxtral({}, "m", "hola", "es_male")
        self.assertEqual((c["voice"], c["language"]), ("es_male", "Spanish"))
        props = {"input": {"type": "string"}, "voice_id": {"type": "string"}, "lang": {"type": "string"},
                 "num_inference_steps": {"type": "integer"}, "voice_sample": {"type": "string"}}
        c = A.cuerpo_voxtral(props, "m", "hola", "es_male")
        self.assertEqual((c["voice"], c["voice_id"], c["lang"], c["num_inference_steps"]),
                         ("es_male", "es_male", "Spanish", 7))
        self.assertNotIn("voice_sample", c)
        self.assertNotIn("audio_options", c)          # el esquema no lo declara: no se envia
        c = A.cuerpo_voxtral({"audio_options": {"type": "object"}}, "m", "hola", "es_male")
        self.assertEqual(c["audio_options"], {"voice_id": "es_male"})
        self.assertNotIn("audio_options", A.cuerpo_voxtral({}, "m", "hola", "es_male", True, False))
        self.assertNotIn("language", A.cuerpo_voxtral(props, "m", "hola", "es_male", False))

    def test_motor_por_lotes_conserva_el_orden(self):
        """Con un motor por lotes, el audio final queda en el mismo orden que el texto."""
        import numpy as np
        import soundfile as sf
        llamadas = []

        class Lotes:
            lote_max = 3
            vacios = 0
            reintentos = 0
            dispositivo = "falso"

            def __call__(self, t):
                return A._tono(t)

            def lote(self, textos, cancel=None):
                llamadas.append(len(textos))
                return [A._tono(t) for t in reversed(textos[::-1])]

        libro = A.leer_txt(self.d / "prueba.txt")
        orig = A.Sintetizador
        try:
            A.Sintetizador = lambda *a, **k: Lotes()
            cfg = A.Config(voz="cb:f1", salida=str(self.d / "lotes"), formato="mp3", conservar_temporal=True)
            A.convertir(libro, cfg, lambda s: None)
            ref = A.Config(voz="cb:f1", salida=str(self.d / "sin_lotes"), formato="mp3", conservar_temporal=True)
            A.Sintetizador = lambda *a, **k: type("U", (Lotes,), {"lote_max": 1})()
            A.convertir(libro, ref, lambda s: None)
        finally:
            A.Sintetizador = orig
        self.assertTrue(llamadas and max(llamadas) <= max(8, 3 * 3))
        a = sorted((self.d / "lotes").rglob("cap_*.wav"))
        b = sorted((self.d / "sin_lotes").rglob("cap_*.wav"))
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            xa, _ = sf.read(str(x))
            ya, _ = sf.read(str(y))
            self.assertEqual(len(xa), len(ya))
            self.assertTrue(np.allclose(xa, ya, atol=1e-3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
