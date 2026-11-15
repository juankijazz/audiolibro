# -*- coding: utf-8 -*-
"""
Audiolibro: convierte EPUB / PDF / TXT en audiolibro (M4B o MP3).

Motores de voz (se elige en la interfaz):
  - Rapido:        Kokoro-82M, pronunciacion castellana de Espana (CPU o GPU).
  - Alta calidad:  Chatterbox Multilingual V3, paquete es-ES (castellano de Espana,
                   clona la voz de una grabacion de referencia; recomendable GPU).

  - Sin argumentos        -> abre la interfaz grafica.
  - Con un archivo        -> modo consola (ver --help).
  - --diagnostico         -> comprueba GPU y velocidad real de sintesis.

La GPU se fuerza de forma explicita (KPipeline(device="cuda")): no depende de
ningun estado global de PyTorch.
"""
import argparse
import hashlib
import math
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")     # sin consola (pythonw) no hay stderr
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _s in ("stdout", "stderr"):
    if getattr(sys, _s) is None:
        setattr(sys, _s, open(os.devnull, "w"))

SAMPLE_RATE = 24000
MAX_CHARS = 230          # tamano maximo de cada trozo enviado al modelo
APP_DIR = Path(__file__).resolve().parent

VOCES = {
    "ef_dora": "Español (pronunciación de España) - Dora, mujer",
    "em_alex": "Español (pronunciación de España) - Alex, hombre",
    "em_santa": "Español (pronunciación de España) - Santa, hombre",
    "pf_dora": "Portugues - Dora (mujer)",
    "pm_alex": "Portugues - Alex (hombre)",
    "ff_siwis": "Frances - Siwis (mujer)",
    "if_sara": "Italiano - Sara (mujer)",
    "im_nicola": "Italiano - Nicola (hombre)",
    "af_heart": "Ingles US - Heart (mujer)",
    "am_michael": "Ingles US - Michael (hombre)",
    "bf_emma": "Ingles UK - Emma (mujer)",
    "bm_george": "Ingles UK - George (hombre)",
}
MOTORES = {
    "kokoro": "Rápido - Kokoro (CPU o GPU)",
    "chatterbox": "Alta calidad - Chatterbox España (GPU)",
}
REF_URL = "https://storage.googleapis.com/chatterbox-demo-samples/mtl-v3-single-language-prompts/es-es/"
CB_VOCES = {
    "cb:f1": {"etiqueta": "España (castellano) - mujer", "archivo": "es_es_f1.wav"},
    "cb:propia": {"etiqueta": "Mi propia voz de referencia (elige un audio)", "archivo": None},
}
CB_EXAG, CB_CFG, CB_TEMP = 0.45, 0.5, 0.75   # narracion serena y estable
FORMATOS = {
    "m4b": "M4B - un archivo con capitulos (recomendado)",
    "mp3": "MP3 - un solo archivo",
    "mp3-caps": "MP3 - un archivo por capitulo",
}


class Cancelado(Exception):
    pass


@dataclass
class Capitulo:
    titulo: str
    parrafos: list
    seleccionado: bool = True

    @property
    def chars(self):
        return sum(len(p) for p in self.parrafos)


@dataclass
class Libro:
    titulo: str
    autor: str
    capitulos: list = field(default_factory=list)


@dataclass
class Config:
    voz: str = "ef_dora"      # id de Kokoro (ef_dora...) o de Chatterbox (cb:f1, cb:propia)
    voz_ref: str = ""         # audio de referencia para cb:propia
    normalizar: bool = True   # volumen uniforme y limpieza de graves en el archivo final
    velocidad: float = 1.0
    formato: str = "m4b"
    salida: str = "."
    usar_gpu: bool = True
    leer_titulos: bool = True
    conservar_temporal: bool = False
    falso: bool = False       # solo para pruebas: tono en vez de voz


# ----------------------------------------------------------------------------
# Limpieza y troceado de texto
# ----------------------------------------------------------------------------
def limpiar_espacios(t):
    t = unicodedata.normalize("NFC", t)
    t = t.replace("­", "").replace("​", "").replace("﻿", "")
    t = t.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", t).strip()


def limpiar_para_tts(t):
    t = limpiar_espacios(t)
    t = re.sub(r"https?://\S+|www\.\S+", " enlace ", t)
    t = re.sub(r"\s*\[\d{1,3}\]", "", t)                 # llamadas de nota [1]
    t = re.sub(r"^\s*[—–-]+\s*(?=\D)", "", t)    # raya de dialogo inicial
    t = re.sub(r"([.?!…])\s*[—–]\s*", r"\1 ", t)    # tras . ? ! ... la raya no lleva coma
    t = re.sub(r"\s*[—–]\s*", ", ", t)           # rayas intercaladas -> pausa
    t = re.sub(r"[\*#_~^|\\<>{}=`]+", " ", t)
    t = re.sub(r",\s*([.!?;:])", r"\1", t)
    t = re.sub(r",(\s*,)+", ",", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


ES_SEPARADOR = re.compile(r"^[\W_]{1,20}$")
_ABREV = re.compile(
    r"\b(Sr|Sra|Srta|Sres|Dr|Dra|Ud|Uds|Vd|Vds|Prof|Lic|Ing|Av|Avda|pág|págs|"
    r"núm|Sta|Sto|Gral|Dña|D|Mons|Fr|vol|art|ej)\.(?=\s)")
_FRASE = re.compile(r'.+?(?:[.!?…]+["»”’)\]]*(?=\s|$)|$)', re.S)


def _partir(s, max_chars):
    if len(s) <= max_chars:
        return [s]
    piezas = re.split(r"(?<=[,;:])\s+", s)
    res, act = [], ""
    for p in piezas:
        if len(p) > max_chars:
            if act:
                res.append(act)
                act = ""
            sub = ""
            for w in p.split(" "):
                if sub and len(sub) + 1 + len(w) > max_chars:
                    res.append(sub)
                    sub = w
                else:
                    sub = (sub + " " + w) if sub else w
            if sub:
                res.append(sub)
        elif act and len(act) + 1 + len(p) > max_chars:
            res.append(act)
            act = p
        else:
            act = (act + " " + p) if act else p
    if act:
        res.append(act)
    return res


def trocear(texto, max_chars=MAX_CHARS):
    """Parte un texto en trozos <= max_chars respetando frases (evita que el
    modelo trunque: Kokoro descarta lo que exceda ~510 fonemas)."""
    t = _ABREV.sub(lambda m: m.group(1) + "\x01", texto)
    frases = [f.strip() for f in _FRASE.findall(t) if f.strip()]
    trozos, act = [], ""
    for f in frases:
        for p in _partir(f, max_chars):
            if act and len(act) + 1 + len(p) > max_chars:
                trozos.append(act)
                act = p
            else:
                act = (act + " " + p) if act else p
    if act:
        trozos.append(act)
    return [x.replace("\x01", ".") for x in trozos]


def plan_capitulo(cap, leer_titulo):
    """Lista de pasos: ('t', texto) = hablar, ('s', segundos) = silencio."""
    pl = []
    if leer_titulo and cap.titulo and re.search(r"\w", cap.titulo):
        tit = limpiar_para_tts(cap.titulo).rstrip(".:") + "."
        for tz in trocear(tit):
            pl.append(("t", tz))
        pl.append(("s", 1.2))
    for par in cap.parrafos:
        if ES_SEPARADOR.match(par.strip()):
            pl.append(("s", 1.2))
            continue
        txt = limpiar_para_tts(par)
        if not txt or not re.search(r"\w", txt):
            continue
        for tz in trocear(txt):
            pl.append(("t", tz))
            fin = re.search(r"[.!?…:»\"”)]$", tz.strip())
            pl.append(("s", 0.35 if fin else 0.18))
        pl.append(("s", 0.6))
    return pl


# ----------------------------------------------------------------------------
# Lectura de libros
# ----------------------------------------------------------------------------
OMITIR_TITULO = re.compile(
    r"^(portada|cubierta|contraportada|portadilla|[ií]ndice|contenidos?|"
    r"tabla de contenidos?|copyright|cr[eé]ditos|derechos.*|aviso legal|legal|"
    r"colof[oó]n|sobre el autor|acerca del autor|about the author|cover|"
    r"title ?page|table of contents|contents|dedicatoria|agradecimientos|notas?|"
    r"bibliograf[ií]a|otros t[ií]tulos.*|inicio)$", re.I)


def _seleccion_por_defecto(libro):
    for c in libro.capitulos:
        c.seleccionado = c.chars >= 150 and not OMITIR_TITULO.match(c.titulo.strip())
    if not any(c.seleccionado for c in libro.capitulos):
        for c in libro.capitulos:
            c.seleccionado = c.chars > 0


_BLOQUES = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote",
            "pre", "dd", "dt", "figcaption", "td", "div"]


def _html_a_parrafos(contenido):
    import warnings
    from bs4 import BeautifulSoup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            soup = BeautifulSoup(contenido, "lxml")
        except Exception:
            soup = BeautifulSoup(contenido, "html.parser")
    titulo_html = soup.title.get_text(strip=True) if soup.title else ""
    for t in soup(["script", "style", "nav", "rt", "rp", "svg", "head"]):
        t.decompose()
    for s in soup.find_all("sup"):
        if re.fullmatch(r"[\[\(]?\d{1,3}[\]\)]?|\*+", s.get_text(strip=True)):
            s.decompose()
    cuerpo = soup.body or soup
    parrafos, encabezado, primero = [], None, True
    for el in cuerpo.find_all(_BLOQUES):
        if el.find(_BLOQUES):
            continue                      # es un contenedor; sus hijos se procesan aparte
        for br in el.find_all("br"):
            br.replace_with(" ")
        txt = limpiar_espacios(el.get_text())
        if not txt:
            continue
        if primero and encabezado is None and el.name in ("h1", "h2", "h3") and len(txt) <= 120:
            encabezado = txt              # el primer titulo se usa como titulo del capitulo
            primero = False
            continue
        primero = False
        parrafos.append(txt)
    return encabezado, titulo_html, parrafos


def leer_epub(ruta):
    import warnings
    from ebooklib import epub, ITEM_DOCUMENT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        libro_epub = epub.read_epub(str(ruta))

    def meta(clave):
        try:
            v = libro_epub.get_metadata("DC", clave)
            return v[0][0].strip() if v else ""
        except Exception:
            return ""

    toc = {}

    def reg(link):
        href, tit = getattr(link, "href", None), getattr(link, "title", None)
        if href and tit:
            toc.setdefault(os.path.basename(href.split("#")[0]), str(tit).strip())

    def recorrer(nodos):
        for n in nodos:
            if isinstance(n, (tuple, list)):
                reg(n[0])
                recorrer(n[1])
            else:
                reg(n)
    try:
        recorrer(libro_epub.toc)
    except Exception:
        pass

    libro = Libro(titulo=meta("title") or Path(ruta).stem, autor=meta("creator"))
    for idref, _ in libro_epub.spine:
        item = libro_epub.get_item_with_id(idref)
        if item is None or item.get_type() != ITEM_DOCUMENT or isinstance(item, epub.EpubNav):
            continue
        encabezado, titulo_html, parrafos = _html_a_parrafos(item.get_content())
        if not parrafos and not encabezado:
            continue
        clave = os.path.basename(item.get_name())
        titulo = toc.get(clave) or encabezado or titulo_html or f"Seccion {len(libro.capitulos) + 1}"
        libro.capitulos.append(Capitulo(limpiar_espacios(titulo), parrafos))
    if not libro.capitulos:
        raise ValueError("No se ha podido extraer texto del EPUB.")
    _seleccion_por_defecto(libro)
    return libro


_NUM_PAG = re.compile(r"^\s*(\d{1,4}|[ivxlcdm]{1,7})\s*$", re.I)


def leer_pdf(ruta):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    doc = fitz.open(str(ruta))
    n = len(doc)
    paginas = []
    for pag in doc:
        h = pag.rect.height
        bl = []
        for b in pag.get_text("blocks", sort=True):
            x0, y0, x1, y1, txt, _no, tipo = b[:7]
            txt = txt.strip()
            if tipo != 0 or not txt:
                continue
            zona = "cab" if y1 < h * 0.09 else ("pie" if y0 > h * 0.91 else None)
            bl.append((zona, txt))
        paginas.append(bl)

    def norm(t):
        return re.sub(r"\d+", "#", re.sub(r"\s+", " ", t.strip().lower()))

    cuenta = Counter()
    for bl in paginas:
        for k in {norm(t) for z, t in bl if z}:
            cuenta[k] += 1
    umbral = max(3, int(0.3 * n))
    repetidos = {k for k, v in cuenta.items() if v >= umbral}

    plano = []
    for i, bl in enumerate(paginas):
        for zona, txt in bl:
            if zona and (norm(txt) in repetidos or _NUM_PAG.match(txt)):
                continue
            t = re.sub(r"(?<=[a-záéíóúüñ])-[ \t]*\n[ \t]*(?=[a-záéíóúüñ])", "", txt)
            t = limpiar_espacios(t.replace("\n", " "))
            if t:
                plano.append([i, t])
    if sum(len(t) for _, t in plano) < 50 * max(1, n):
        raise ValueError("Este PDF parece escaneado (casi no tiene texto). Necesita OCR antes de convertirlo.")

    fus = []
    for i, t in plano:
        if fus and t[:1].islower() and not re.search(r"[.!?…:;»”\"')\]]$", fus[-1][1]):
            fus[-1][1] += " " + t
        else:
            fus.append([i, t])

    toc = [t for t in doc.get_toc(simple=True) if t[2] >= 1]
    if toc:
        niveles = sorted({t[0] for t in toc})
        nivel = next((l for l in niveles if sum(1 for t in toc if t[0] == l) >= 3), niveles[0])
        cortes = sorted({(t[2] - 1, t[1].strip()) for t in toc if t[0] == nivel})
        if cortes[0][0] > 0:
            cortes.insert(0, (0, "Inicio"))
    else:
        cortes = [(a, f"Paginas {a + 1}-{min(a + 15, n)}") for a in range(0, n, 15)]

    md = doc.metadata or {}
    libro = Libro(titulo=(md.get("title") or "").strip() or Path(ruta).stem,
                  autor=(md.get("author") or "").strip())
    for j, (p0, titulo) in enumerate(cortes):
        p1 = cortes[j + 1][0] if j + 1 < len(cortes) else n
        pars = [t for i, t in fus if p0 <= i < p1]
        if pars and norm(pars[0]) == norm(titulo):
            pars = pars[1:]               # el titulo se locuta aparte; evita repetirlo
        if pars:
            libro.capitulos.append(Capitulo(limpiar_espacios(titulo), pars))
    _seleccion_por_defecto(libro)
    return libro


def leer_txt(ruta):
    raw = Path(ruta).read_bytes()
    texto = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            texto = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    if "\n\n" in texto:
        pars = [limpiar_espacios(p) for p in re.split(r"\n\s*\n", texto)]
    else:
        pars = [limpiar_espacios(p) for p in texto.split("\n")]
    pars = [p for p in pars if p]
    cab = re.compile(r"^(cap[ií]tulo|parte|libro|chapter|part)\b.{0,60}$", re.I)
    libro = Libro(titulo=Path(ruta).stem, autor="")
    idx = [i for i, p in enumerate(pars) if cab.match(p)]
    if len(idx) >= 2:
        if idx[0] > 0:
            libro.capitulos.append(Capitulo("Inicio", pars[:idx[0]]))
        for a, b in zip(idx, idx[1:] + [len(pars)]):
            libro.capitulos.append(Capitulo(pars[a], pars[a + 1:b]))
    else:
        act, tam, k = [], 0, 1
        for p in pars:
            act.append(p)
            tam += len(p)
            if tam >= 20000:
                libro.capitulos.append(Capitulo(f"Parte {k}", act))
                act, tam, k = [], 0, k + 1
        if act:
            libro.capitulos.append(Capitulo(f"Parte {k}", act))
    if not libro.capitulos:
        raise ValueError("El archivo esta vacio.")
    _seleccion_por_defecto(libro)
    return libro


def leer_libro(ruta):
    ext = Path(ruta).suffix.lower()
    if ext == ".epub":
        return leer_epub(ruta)
    if ext == ".pdf":
        return leer_pdf(ruta)
    if ext in (".txt", ".md"):
        return leer_txt(ruta)
    raise ValueError(f"Formato no soportado: {ext} (usa EPUB, PDF o TXT)")


# ----------------------------------------------------------------------------
# Sintesis
# ----------------------------------------------------------------------------
_PIPES = {}


def info_gpu():
    """(hay_gpu, texto)"""
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"GPU: {torch.cuda.get_device_name(0)} (PyTorch {torch.__version__})"
        return False, f"GPU NO disponible (PyTorch {torch.__version__}): se usara la CPU, mucho mas lenta"
    except Exception as e:
        return False, f"PyTorch no disponible: {e}"


def _tono(texto):
    import numpy as np
    seg = max(0.3, len(texto) / 15.0)
    t = np.arange(int(SAMPLE_RATE * seg)) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32")


class SintKokoro:
    def __init__(self, voz, velocidad, usar_gpu=True, log=print, falso=False):
        self.voz, self.velocidad, self.falso, self.log = voz, velocidad, falso, log
        self.dispositivo = "falso"
        self.vacios = 0
        if falso:
            return
        import torch
        self.torch = torch
        gpu_ok = torch.cuda.is_available()
        if usar_gpu and not gpu_ok:
            log("AVISO: se pidio GPU pero PyTorch no la ve; se usa la CPU (lento).")
        self.dispositivo = "cuda" if (usar_gpu and gpu_ok) else "cpu"
        clave = (voz[0], self.dispositivo)
        if clave not in _PIPES:
            log("Cargando el modelo (la primera vez se descarga, unos 330 MB)...")
            from kokoro import KPipeline
            _PIPES[clave] = KPipeline(lang_code=voz[0], device=self.dispositivo)
        self.pipe = _PIPES[clave]
        nombre = torch.cuda.get_device_name(0) if self.dispositivo == "cuda" else "CPU"
        log(f"Sintetizando en: {self.dispositivo.upper()} ({nombre})")

    def __call__(self, texto):
        import numpy as np
        if self.falso:
            return _tono(texto)
        partes = []
        with self.torch.inference_mode():
            for r in self.pipe(texto, voice=self.voz, speed=self.velocidad, split_pattern=None):
                a = r.audio
                if a is None:
                    continue
                if hasattr(a, "detach"):
                    a = a.detach().cpu().numpy()
                partes.append(np.asarray(a, dtype="float32").reshape(-1))
        if not partes:
            self.vacios += 1
            return np.zeros(0, dtype="float32")
        return np.concatenate(partes)

# ----------------------------------------------------------------------------
# Motor de alta calidad: Chatterbox Multilingual V3 (paquete es-ES)
# ----------------------------------------------------------------------------
def _dir_datos():
    base = os.environ.get("AUDIOLIBRO_DATOS")
    if base:
        return Path(base)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Audiolibro"
    return Path.home() / ".audiolibro"


DIR_MODELOS = _dir_datos() / "modelos"


def motor_de(voz):
    return "chatterbox" if str(voz).startswith("cb:") else "kokoro"


def descargar(url, dest):
    import urllib.request
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Audiolibro/2"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def referencia_cb(voz, voz_ref, log=print):
    """Ruta del audio de referencia que define la voz (y el acento) de Chatterbox."""
    if voz == "cb:propia":
        if not voz_ref or not Path(voz_ref).is_file():
            raise ValueError("Elige un audio de referencia (WAV, MP3 o FLAC de unos 10 segundos "
                             "con la voz que quieres imitar).")
        return str(voz_ref)
    info = CB_VOCES.get(voz)
    if not info or not info["archivo"]:
        raise ValueError(f"Voz desconocida: {voz}")
    dest = DIR_MODELOS / "voces" / info["archivo"]
    if not dest.exists():
        log("Descargando la voz de referencia de España...")
        descargar(REF_URL + info["archivo"], dest)
    return str(dest)


def cargar_chatterbox(dispositivo, log=print):
    """Monta el modelo es-ES: T3 propio de España + decodificador V3 (o el estandar si no encaja)."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder

    carpeta = DIR_MODELOS / "chatterbox-es"
    carpeta.mkdir(parents=True, exist_ok=True)
    base, es = "ResembleAI/chatterbox", "ResembleAI/Chatterbox-Multilingual-es-es"

    def bajar(repo, nombre, aviso=None):
        dest = carpeta / nombre
        if not dest.exists():
            log(aviso or f"Descargando {nombre} ...")
            hf_hub_download(repo, nombre, local_dir=str(carpeta))
        return dest

    log("Preparando el modelo de alta calidad (la primera vez descarga unos 4,3 GB; solo una vez)...")
    ve_p = bajar(base, "ve.pt")
    tok_p = bajar(es, "grapheme_mtl_merged_expanded_v1.json")
    t3_p = bajar(es, "t3_es_es.safetensors", "Descargando el modelo de España (2,1 GB)...")
    mapa = torch.device("cpu") if dispositivo == "cpu" else None

    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(str(ve_p), map_location=mapa, weights_only=True))
    ve.to(dispositivo).eval()

    t3 = T3(T3Config.multilingual())
    est = load_file(str(t3_p))
    if "model" in est:
        est = est["model"][0]
    t3.load_state_dict(est)
    t3.to(dispositivo).eval()

    s3gen = S3Gen()
    try:
        s3_p = bajar(es, "s3gen_v3.safetensors", "Descargando el decodificador de voz V3 (1,1 GB)...")
        s3gen.load_state_dict(load_file(str(s3_p)))
        log("Decodificador V3 cargado.")
    except Exception as e:
        log(f"Aviso: el decodificador V3 no encaja ({type(e).__name__}); se usa el estándar.")
        s3gen = S3Gen()
        s3_p = bajar(base, "s3gen.pt", "Descargando el decodificador estándar (1 GB)...")
        s3gen.load_state_dict(torch.load(str(s3_p), map_location=mapa, weights_only=True))
    s3gen.to(dispositivo).eval()

    tok = MTLTokenizer(str(tok_p))
    return ChatterboxMultilingualTTS(t3, s3gen, ve, tok, dispositivo, conds=None)


class SintChatterbox:
    def __init__(self, voz, velocidad, usar_gpu=True, log=print, falso=False, voz_ref=""):
        self.voz, self.velocidad, self.falso, self.log = voz, velocidad, falso, log
        self.dispositivo = "falso"
        self.vacios = 0
        self.reintentos = 0
        if falso:
            return
        import torch
        self.torch = torch
        gpu_ok = torch.cuda.is_available()
        if usar_gpu and not gpu_ok:
            log("AVISO: se pidio GPU pero PyTorch no la ve; se usa la CPU (muy lento con este modelo).")
        self.dispositivo = "cuda" if (usar_gpu and gpu_ok) else "cpu"
        ref = referencia_cb(voz, voz_ref, log)
        clave = ("chatterbox", self.dispositivo)
        if clave not in _PIPES:
            _PIPES[clave] = cargar_chatterbox(self.dispositivo, log)
        self.m = _PIPES[clave]
        self.m.prepare_conditionals(ref, exaggeration=CB_EXAG)
        nombre = torch.cuda.get_device_name(0) if self.dispositivo == "cuda" else "CPU"
        log(f"Sintetizando en: {self.dispositivo.upper()} ({nombre}) - motor de alta calidad")

    def __call__(self, texto):
        import numpy as np
        if self.falso:
            return _tono(texto)
        esperado = max(1.0, len(texto) / 15.0)        # ~15 caracteres por segundo al narrar
        mejor, mejor_err = None, None
        for intento in range(3):
            self.torch.manual_seed(zlib.crc32(texto.encode("utf-8")) + intento)
            with self.torch.inference_mode():
                w = self.m.generate(texto, language_id="es", exaggeration=CB_EXAG,
                                    cfg_weight=CB_CFG, temperature=CB_TEMP)
            a = np.asarray(w.detach().cpu().numpy(), dtype="float32").reshape(-1)
            dur = len(a) / SAMPLE_RATE
            err = abs(math.log(max(dur, 0.05) / esperado))
            if mejor is None or err < mejor_err:
                mejor, mejor_err = a, err
            if 0.4 * esperado - 0.5 <= dur <= 2.0 * esperado + 1.0:
                break                                  # duracion plausible: sin alucinaciones
            self.reintentos += 1
        if mejor is None or not len(mejor):
            self.vacios += 1
            return np.zeros(0, dtype="float32")
        return mejor


def Sintetizador(voz, velocidad, usar_gpu=True, log=print, falso=False, voz_ref=""):
    if motor_de(voz) == "chatterbox":
        return SintChatterbox(voz, velocidad, usar_gpu, log, falso, voz_ref)
    return SintKokoro(voz, velocidad, usar_gpu, log, falso)


def acondicionar(a):
    """Quita la componente continua, recorta el silencio sobrante y suaviza los bordes (sin clics)."""
    import numpy as np
    if len(a) == 0:
        return a
    a = a - float(np.median(a))
    idx = np.flatnonzero(np.abs(a) > 10 ** (-52 / 20))
    if len(idx) == 0:
        return a[:0]
    margen = int(0.04 * SAMPLE_RATE)
    a = a[max(0, idx[0] - margen): min(len(a), idx[-1] + margen)].copy()
    n = min(len(a) // 2, int(0.008 * SAMPLE_RATE))
    if n > 1:
        r = np.linspace(0.0, 1.0, n, dtype="float32")
        a[:n] *= r
        a[-n:] *= r[::-1]
    return a


def nombre_seguro(s):
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s or "").strip(" .")
    return s[:80]


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise RuntimeError("No se encuentra ffmpeg. Ejecuta INSTALAR.bat de nuevo (instala imageio-ffmpeg).")


def _run(cmd):
    flags = 0x08000000 if os.name == "nt" else 0     # sin ventana de consola
    p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       errors="replace", creationflags=flags)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg fallo:\n" + p.stderr[-1500:])


def _esc_meta(s):
    return re.sub(r"([=;#\\\n])", r"\\\1", s or "")


def _filtro_audio(cfg):
    f = []
    if cfg.normalizar:
        f.append("highpass=f=70")
    atempo = motor_de(cfg.voz) == "chatterbox" and abs(cfg.velocidad - 1.0) > 0.01
    if atempo:
        f.append(f"atempo={cfg.velocidad:.3f}")
    if cfg.normalizar:
        f.append("loudnorm=I=-19:TP=-1.5:LRA=9")
    af = ["-af", ",".join(f), "-ar", str(SAMPLE_RATE)] if f else []
    return af, (1.0 / cfg.velocidad if atempo else 1.0)


def ensamblar(wavs, titulos, libro, cfg, out_dir, base, log):
    import soundfile as sf
    ff = _ffmpeg()
    af, escala_t = _filtro_audio(cfg)
    if cfg.formato == "mp3-caps":
        carpeta = out_dir / base
        carpeta.mkdir(exist_ok=True)
        for i, (w, t) in enumerate(zip(wavs, titulos), 1):
            dest = carpeta / f"{i:02d} - {nombre_seguro(t) or 'capitulo'}.mp3"
            _run([ff, "-y", "-i", str(w)] + af + ["-c:a", "libmp3lame", "-q:a", "4",
                  "-metadata", f"title={t}", "-metadata", f"album={libro.titulo}",
                  "-metadata", f"artist={libro.autor}", "-metadata", f"track={i}", str(dest)])
        return carpeta
    lista = out_dir / (base + "_lista.txt")
    lista.write_text("".join("file '%s'\n" % str(w).replace("\\", "/").replace("'", "'\\''")
                             for w in wavs), encoding="utf-8")
    try:
        if cfg.formato == "mp3":
            dest = out_dir / (base + ".mp3")
            _run([ff, "-y", "-f", "concat", "-safe", "0", "-i", str(lista)] + af +
                 ["-c:a", "libmp3lame", "-q:a", "4", "-metadata", f"title={libro.titulo}",
                  "-metadata", f"artist={libro.autor}", str(dest)])
            return dest
        meta = out_dir / (base + "_meta.txt")
        L = [";FFMETADATA1", f"title={_esc_meta(libro.titulo)}", f"artist={_esc_meta(libro.autor)}",
             f"album={_esc_meta(libro.titulo)}", "genre=Audiobook"]
        t_ms = 0
        for w, t in zip(wavs, titulos):
            dur = int(round(sf.info(str(w)).duration * 1000 * escala_t))
            L += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={t_ms}", f"END={t_ms + dur}",
                  f"title={_esc_meta(t)}"]
            t_ms += dur
        meta.write_text("\n".join(L) + "\n", encoding="utf-8")
        dest = out_dir / (base + ".m4b")
        _run([ff, "-y", "-f", "concat", "-safe", "0", "-i", str(lista), "-i", str(meta),
              "-map", "0:a", "-map_metadata", "1"] + af +
             ["-c:a", "aac", "-b:a", "64k", "-ac", "1",
              "-movflags", "+faststart", "-f", "ipod", str(dest)])
        meta.unlink(missing_ok=True)
        return dest
    finally:
        lista.unlink(missing_ok=True)


def _fmt_t(seg):
    seg = int(max(0, seg))
    return f"{seg // 3600}h {seg % 3600 // 60:02d}m" if seg >= 3600 else f"{seg // 60}m {seg % 60:02d}s"


def convertir(libro, cfg, log=print, progreso=None, cancel=None):
    import numpy as np
    import soundfile as sf
    progreso = progreso or (lambda f, m: None)
    cancel = cancel or threading.Event()
    caps = [(i, c) for i, c in enumerate(libro.capitulos) if c.seleccionado]
    if not caps:
        raise ValueError("No hay capitulos seleccionados.")
    out_dir = Path(cfg.salida).resolve()      # ruta absoluta: ffmpeg resuelve la lista concat relativa a su carpeta
    out_dir.mkdir(parents=True, exist_ok=True)
    base = nombre_seguro(libro.titulo) or "audiolibro"
    trabajo = out_dir / (base + "_temporal")
    trabajo.mkdir(exist_ok=True)

    planes = {i: plan_capitulo(c, cfg.leer_titulos) for i, c in caps}
    chars = {i: sum(len(v) for k, v in pl if k == "t") for i, pl in planes.items()}
    total = max(1, sum(chars.values()))
    log(f"{len(caps)} capitulos, {total:,} caracteres a locutar.".replace(",", "."))

    sint = Sintetizador(cfg.voz, cfg.velocidad, cfg.usar_gpu, log, cfg.falso, cfg.voz_ref)
    nativa = motor_de(cfg.voz) == "kokoro"          # Kokoro aplica la velocidad al sintetizar
    hecho = sesion = 0
    t0 = time.time()
    wavs, titulos = [], []
    for n, (i, c) in enumerate(caps, 1):
        firma = hashlib.sha1(json.dumps([planes[i], cfg.voz, cfg.velocidad if nativa else 0,
                                         cfg.voz_ref if cfg.voz == "cb:propia" else "", 2]).encode()).hexdigest()[:8]
        wav = trabajo / f"cap_{i + 1:03d}_{firma}.wav"
        titulos.append(c.titulo)
        wavs.append(wav)
        if wav.exists():
            log(f"[{n}/{len(caps)}] {c.titulo}: ya estaba hecho, se reutiliza.")
            hecho += chars[i]
            progreso(hecho / total, f"Capitulo {n}/{len(caps)} (reutilizado)")
            continue
        log(f"[{n}/{len(caps)}] {c.titulo}")
        tmp = wav.with_suffix(".part")
        with sf.SoundFile(str(tmp), mode="w", samplerate=SAMPLE_RATE, channels=1,
                          subtype="PCM_16", format="WAV") as f:
            for k, v in planes[i]:
                if cancel.is_set():
                    raise Cancelado()
                if k == "s":
                    f.write(np.zeros(int(SAMPLE_RATE * v), dtype="float32"))
                    continue
                a = acondicionar(sint(v))
                if len(a):
                    f.write(np.clip(a, -1.0, 1.0))
                hecho += len(v)
                sesion += len(v)
                el = time.time() - t0
                eta = el / sesion * (total - hecho) if sesion else 0
                progreso(hecho / total, f"Capitulo {n}/{len(caps)} - {hecho * 100 // total}% - quedan ~{_fmt_t(eta)}")
        tmp.replace(wav)
    if sint.vacios:
        log(f"Aviso: {sint.vacios} trozos no produjeron audio (texto sin pronunciacion).")
    if getattr(sint, "reintentos", 0):
        log(f"Se repitieron {sint.reintentos} trozos con duracion anomala (control anti-alucinaciones).")
    progreso(1.0, "Codificando el audiolibro...")
    log("Codificando el archivo final...")
    dest = ensamblar(wavs, titulos, libro, cfg, out_dir, base, log)
    if not cfg.conservar_temporal:
        shutil.rmtree(trabajo, ignore_errors=True)
    log(f"Terminado en {_fmt_t(time.time() - t0)}: {dest}")
    progreso(1.0, "Listo")
    return dest


def generar_muestra(texto, voz, velocidad, usar_gpu, log=print, falso=False, voz_ref="", destino=None):
    import numpy as np
    import soundfile as sf
    sint = Sintetizador(voz, velocidad, usar_gpu, log, falso, voz_ref)
    partes = []
    for t in trocear(limpiar_para_tts(texto))[:2]:
        partes += [acondicionar(sint(t)), np.zeros(int(SAMPLE_RATE * 0.3), dtype="float32")]
    ruta = Path(destino) if destino else Path(tempfile.gettempdir()) / "audiolibro_muestra.wav"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(ruta), np.clip(np.concatenate(partes), -1, 1), SAMPLE_RATE, subtype="PCM_16")
    return ruta


# ----------------------------------------------------------------------------
# Diagnostico y precarga
# ----------------------------------------------------------------------------
def diagnostico():
    print("== Diagnostico de Audiolibro ==")
    ok, txt = info_gpu()
    print(txt)
    if not ok:
        print("-> PyTorch no ve la GPU. Vuelve a ejecutar INSTALAR.bat.")
    texto = ("En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que "
             "vivia un hidalgo de los de lanza en astillero, adarga antigua, rocin flaco y galgo corredor.")
    sint = Sintetizador("ef_dora", 1.0, True, print)
    sint(texto)                                  # calentamiento
    t = time.time()
    seg_audio = sum(len(sint(texto)) for _ in range(5)) / SAMPLE_RATE
    dt = time.time() - t
    print(f"Dispositivo usado: {sint.dispositivo.upper()}")
    print(f"{seg_audio:.1f} s de audio generados en {dt:.1f} s -> {seg_audio / dt:.1f}x tiempo real")
    if sint.dispositivo == "cuda" and seg_audio / dt < 10:
        print("Aviso: va mas lento de lo esperable para una GPU; cierra otros programas que la usen.")
    return 0


def precargar_hq():
    print("Descargando el modelo de alta calidad (unos 4,3 GB, solo la primera vez)...")
    cargar_chatterbox("cpu", print)
    print("Modelo de alta calidad listo.")
    return 0


def precargar():
    print("Descargando el modelo y las voces (solo la primera vez)...")
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="e", device="cpu")
    for v in ("ef_dora", "em_alex", "em_santa"):
        pipe.load_voice(v)
    print("Modelo y voces listos.")
    return 0


# ----------------------------------------------------------------------------
# Interfaz grafica
# ----------------------------------------------------------------------------
def gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    class App:
        def __init__(self, root):
            self.root = root
            root.title("Audiolibro - EPUB / PDF / TXT a audio")
            root.geometry("900x780")
            root.minsize(820, 700)
            self.q = queue.Queue()
            self.libro = None
            self.cancel = threading.Event()
            self.ocupado = False
            self.gpu_ok = False

            self.v_archivo = tk.StringVar()
            self.v_salida = tk.StringVar()
            self.v_motor = tk.StringVar(value=MOTORES["kokoro"])
            self.v_voz = tk.StringVar(value=f"ef_dora  |  {VOCES['ef_dora']}")
            self.v_ref = tk.StringVar(value="")
            self.v_info = tk.StringVar(value="")
            self.v_norm = tk.BooleanVar(value=True)
            self.v_vel = tk.DoubleVar(value=1.0)
            self.v_fmt = tk.StringVar(value=FORMATOS["m4b"])
            self.v_gpu = tk.BooleanVar(value=True)
            self.v_tit = tk.BooleanVar(value=True)
            self.v_gpu_txt = tk.StringVar(value="Comprobando GPU...")
            self.v_estado = tk.StringVar(value="Elige un libro para empezar.")

            f = ttk.Frame(root, padding=10)
            f.pack(fill="both", expand=True)
            f.columnconfigure(1, weight=1)

            ttk.Label(f, text="Libro:").grid(row=0, column=0, sticky="w")
            ttk.Entry(f, textvariable=self.v_archivo).grid(row=0, column=1, sticky="ew", padx=6)
            ttk.Button(f, text="Examinar...", command=self.elegir).grid(row=0, column=2)
            ttk.Button(f, text="Analizar", command=self.analizar).grid(row=0, column=3, padx=(6, 0))

            cols = ("sel", "titulo", "chars")
            self.tree = ttk.Treeview(f, columns=cols, show="headings", height=9, selectmode="none")
            self.tree.heading("sel", text="Leer")
            self.tree.heading("titulo", text="Capitulo (clic para marcar / desmarcar)")
            self.tree.heading("chars", text="Caracteres")
            self.tree.column("sel", width=50, anchor="center", stretch=False)
            self.tree.column("titulo", width=560)
            self.tree.column("chars", width=90, anchor="e", stretch=False)
            self.tree.bind("<Button-1>", self.alternar)
            sb = ttk.Scrollbar(f, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=sb.set)
            self.tree.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(8, 0))
            sb.grid(row=1, column=4, sticky="ns", pady=(8, 0))
            f.rowconfigure(1, weight=2)

            bt = ttk.Frame(f)
            bt.grid(row=2, column=0, columnspan=4, sticky="w", pady=4)
            ttk.Button(bt, text="Marcar todo", command=lambda: self.marcar(True)).pack(side="left")
            ttk.Button(bt, text="Desmarcar todo", command=lambda: self.marcar(False)).pack(side="left", padx=6)

            op = ttk.LabelFrame(f, text="Opciones", padding=8)
            op.grid(row=3, column=0, columnspan=4, sticky="ew", pady=6)
            op.columnconfigure(1, weight=1)
            ttk.Label(op, text="Motor:").grid(row=0, column=0, sticky="w")
            self.cb_motor = ttk.Combobox(op, textvariable=self.v_motor, state="readonly", width=42,
                                         values=list(MOTORES.values()))
            self.cb_motor.grid(row=0, column=1, sticky="w", padx=6, pady=2)
            self.cb_motor.bind("<<ComboboxSelected>>", lambda e: self.cambia_motor())
            ttk.Label(op, text="Voz:").grid(row=1, column=0, sticky="w")
            self.cb_voz = ttk.Combobox(op, textvariable=self.v_voz, state="readonly", width=60)
            self.cb_voz.grid(row=1, column=1, sticky="w", padx=6, pady=2)
            self.cb_voz.bind("<<ComboboxSelected>>", lambda e: self.cambia_voz())
            ttk.Button(op, text="Escuchar muestra", command=self.muestra).grid(row=1, column=2, padx=6)
            ttk.Label(op, text="Audio de referencia:").grid(row=2, column=0, sticky="w")
            self.en_ref = ttk.Entry(op, textvariable=self.v_ref, state="readonly")
            self.en_ref.grid(row=2, column=1, sticky="ew", padx=6, pady=2)
            self.bt_ref = ttk.Button(op, text="Elegir...", command=self.elegir_ref)
            self.bt_ref.grid(row=2, column=2, padx=6)
            ttk.Label(op, textvariable=self.v_info, foreground="#555", wraplength=760,
                      justify="left").grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 4))
            ttk.Label(op, text="Velocidad:").grid(row=4, column=0, sticky="w", pady=4)
            fv = ttk.Frame(op)
            fv.grid(row=4, column=1, sticky="w", padx=6)
            ttk.Scale(fv, from_=0.7, to=1.4, variable=self.v_vel, length=200,
                      command=lambda _=None: self.lbl_vel.config(text=f"{self.v_vel.get():.2f}x")).pack(side="left")
            self.lbl_vel = ttk.Label(fv, text="1.00x", width=6)
            self.lbl_vel.pack(side="left", padx=6)
            ttk.Label(op, text="Formato:").grid(row=5, column=0, sticky="w")
            ttk.Combobox(op, textvariable=self.v_fmt, state="readonly", width=42,
                         values=list(FORMATOS.values())).grid(row=5, column=1, sticky="w", padx=6, pady=2)
            ttk.Label(op, text="Guardar en:").grid(row=6, column=0, sticky="w", pady=4)
            ttk.Entry(op, textvariable=self.v_salida).grid(row=6, column=1, sticky="ew", padx=6)
            ttk.Button(op, text="Cambiar...", command=self.elegir_salida).grid(row=6, column=2, padx=6)
            self.chk_gpu = ttk.Checkbutton(op, text="Usar la GPU (CUDA)", variable=self.v_gpu)
            self.chk_gpu.grid(row=7, column=0, columnspan=2, sticky="w")
            ttk.Label(op, textvariable=self.v_gpu_txt, foreground="#555").grid(row=7, column=1, columnspan=2, sticky="e")
            ttk.Checkbutton(op, text="Leer en voz alta el titulo de cada capitulo",
                            variable=self.v_tit).grid(row=8, column=0, columnspan=3, sticky="w")
            ttk.Checkbutton(op, text="Audio limpio: volumen uniforme y sin graves molestos",
                            variable=self.v_norm).grid(row=9, column=0, columnspan=3, sticky="w")
            self.cambia_motor()

            ac = ttk.Frame(f)
            ac.grid(row=4, column=0, columnspan=4, sticky="ew", pady=4)
            self.btn_go = ttk.Button(ac, text="CONVERTIR A AUDIOLIBRO", command=self.convertir)
            self.btn_go.pack(side="left")
            self.btn_stop = ttk.Button(ac, text="Cancelar", command=self.cancelar, state="disabled")
            self.btn_stop.pack(side="left", padx=8)
            self.pb = ttk.Progressbar(f, maximum=1000)
            self.pb.grid(row=5, column=0, columnspan=4, sticky="ew")
            ttk.Label(f, textvariable=self.v_estado).grid(row=6, column=0, columnspan=4, sticky="w", pady=2)
            self.log = tk.Text(f, height=7, state="disabled", wrap="word", font=("Consolas", 9))
            self.log.grid(row=7, column=0, columnspan=4, sticky="nsew")
            f.rowconfigure(7, weight=1)

            threading.Thread(target=self._detectar_gpu, daemon=True).start()
            root.after(100, self._poll)

        # --- utilidades
        def escribir(self, s):
            self.log.config(state="normal")
            self.log.insert("end", s + "\n")
            self.log.see("end")
            self.log.config(state="disabled")

        def voz(self):
            return self.v_voz.get().split("|")[0].strip()

        def motor(self):
            return next(k for k, v in MOTORES.items() if v == self.v_motor.get())

        def cambia_motor(self):
            if self.motor() == "kokoro":
                vals = [f"{k}  |  {v}" for k, v in VOCES.items()]
                self.v_info.set("Rápido: Kokoro, pronuncia con la fonética de España (c/z como 'th'). "
                                "Ideal para libros largos; funciona en CPU y GPU.")
            else:
                vals = [f"{k}  |  {v['etiqueta']}" for k, v in CB_VOCES.items()]
                self.v_info.set("Alta calidad: Chatterbox con el paquete de España. Imita la voz y el acento del "
                                "audio de referencia (puedes poner el tuyo). Más lento y necesita GPU; la primera "
                                "vez descarga unos 4,3 GB.")
            self.cb_voz.config(values=vals)
            self.v_voz.set(vals[0])
            self.cambia_voz()

        def cambia_voz(self):
            propia = self.voz() == "cb:propia"
            self.bt_ref.config(state="normal" if propia else "disabled")
            if not propia:
                self.v_ref.set("")

        def elegir_ref(self):
            r = filedialog.askopenfilename(title="Audio de referencia (voz a imitar, ~10 s)",
                                           filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg"), ("Todos", "*.*")])
            if r:
                self.v_ref.set(r)

        def formato(self):
            return next(k for k, v in FORMATOS.items() if v == self.v_fmt.get())

        def bloquear(self, ocupado):
            self.ocupado = ocupado
            self.btn_go.config(state="disabled" if ocupado else "normal")
            self.btn_stop.config(state="normal" if ocupado else "disabled")

        def _detectar_gpu(self):
            ok, txt = info_gpu()
            self.q.put(("gpu", ok, txt))

        # --- acciones
        def elegir(self):
            r = filedialog.askopenfilename(title="Elige un libro",
                                           filetypes=[("Libros", "*.epub *.pdf *.txt"), ("Todos", "*.*")])
            if r:
                self.v_archivo.set(r)
                self.v_salida.set(str(Path(r).parent))
                self.analizar()

        def elegir_salida(self):
            r = filedialog.askdirectory(title="Carpeta de salida")
            if r:
                self.v_salida.set(r)

        def analizar(self):
            ruta = self.v_archivo.get().strip().strip('"')
            if not ruta or self.ocupado:
                return
            if not self.v_salida.get():
                self.v_salida.set(str(Path(ruta).parent))
            self.v_estado.set("Leyendo el libro...")
            self.bloquear(True)
            self.btn_stop.config(state="disabled")

            def trabajo():
                try:
                    self.q.put(("libro", leer_libro(ruta)))
                except Exception as e:
                    self.q.put(("error", f"No se pudo leer el libro: {e}"))
            threading.Thread(target=trabajo, daemon=True).start()

        def poblar(self):
            self.tree.delete(*self.tree.get_children())
            for i, c in enumerate(self.libro.capitulos):
                self.tree.insert("", "end", iid=str(i),
                                 values=("[x]" if c.seleccionado else "[ ]", c.titulo, f"{c.chars:,}".replace(",", ".")))

        def alternar(self, ev):
            iid = self.tree.identify_row(ev.y)
            if not iid or not self.libro or self.ocupado:
                return
            c = self.libro.capitulos[int(iid)]
            c.seleccionado = not c.seleccionado
            self.tree.set(iid, "sel", "[x]" if c.seleccionado else "[ ]")

        def marcar(self, v):
            if not self.libro or self.ocupado:
                return
            for c in self.libro.capitulos:
                c.seleccionado = v
            self.poblar()

        def muestra(self):
            if self.ocupado:
                return
            texto = "Hola. Esta es una muestra de la voz que has elegido para tu audiolibro."
            if self.libro:
                for c in self.libro.capitulos:
                    if c.seleccionado and c.parrafos:
                        largos = [p for p in c.parrafos if len(p) > 80]
                        texto = (largos or c.parrafos)[0][:400]
                        break
            voz, vel, gpu, ref = self.voz(), self.v_vel.get(), self.v_gpu.get(), self.v_ref.get()
            if voz == "cb:propia" and not ref:
                messagebox.showinfo("Audiolibro", "Elige primero el audio de referencia.")
                return
            self.bloquear(True)
            self.btn_stop.config(state="disabled")
            self.v_estado.set("Generando muestra...")

            def trabajo():
                try:
                    r = generar_muestra(texto, voz, vel, gpu, lambda s: self.q.put(("log", s)), voz_ref=ref)
                    self.q.put(("muestra", str(r)))
                except Exception:
                    self.q.put(("error", traceback.format_exc()))
            threading.Thread(target=trabajo, daemon=True).start()

        def convertir(self):
            if not self.libro:
                messagebox.showinfo("Audiolibro", "Primero elige un libro y pulsa Analizar.")
                return
            if not any(c.seleccionado for c in self.libro.capitulos):
                messagebox.showinfo("Audiolibro", "Marca al menos un capitulo.")
                return
            if self.voz() == "cb:propia" and not self.v_ref.get():
                messagebox.showinfo("Audiolibro", "Elige primero el audio de referencia.")
                return
            cfg = Config(voz=self.voz(), voz_ref=self.v_ref.get(), velocidad=round(self.v_vel.get(), 2),
                         formato=self.formato(), salida=self.v_salida.get() or ".",
                         usar_gpu=self.v_gpu.get(), leer_titulos=self.v_tit.get(),
                         normalizar=self.v_norm.get())
            self.cancel.clear()
            self.bloquear(True)
            self.pb["value"] = 0

            def trabajo():
                try:
                    r = convertir(self.libro, cfg, lambda s: self.q.put(("log", s)),
                                  lambda fr, m: self.q.put(("prog", fr, m)), self.cancel)
                    self.q.put(("fin", str(r)))
                except Cancelado:
                    self.q.put(("cancelado",))
                except Exception:
                    self.q.put(("error", traceback.format_exc()))
            threading.Thread(target=trabajo, daemon=True).start()

        def cancelar(self):
            self.cancel.set()
            self.v_estado.set("Cancelando (se conserva lo ya hecho para reanudar)...")

        def _poll(self):
            try:
                while True:
                    m = self.q.get_nowait()
                    k = m[0]
                    if k == "log":
                        self.escribir(m[1])
                    elif k == "prog":
                        self.pb["value"] = m[1] * 1000
                        self.v_estado.set(m[2])
                    elif k == "gpu":
                        self.gpu_ok = m[1]
                        self.v_gpu_txt.set(m[2])
                        self.v_gpu.set(m[1])
                        if not m[1]:
                            self.chk_gpu.config(state="disabled")
                    elif k == "libro":
                        self.libro = m[1]
                        self.poblar()
                        n = sum(c.seleccionado for c in self.libro.capitulos)
                        self.v_estado.set(f"'{self.libro.titulo}': {len(self.libro.capitulos)} secciones, {n} marcadas.")
                        self.bloquear(False)
                    elif k == "muestra":
                        self.bloquear(False)
                        self.v_estado.set("Reproduciendo muestra...")
                        try:
                            import winsound
                            winsound.PlaySound(None, winsound.SND_PURGE)
                            winsound.PlaySound(m[1], winsound.SND_FILENAME | winsound.SND_ASYNC)
                        except Exception:
                            self.escribir(f"Muestra guardada en {m[1]}")
                    elif k == "fin":
                        self.bloquear(False)
                        self.v_estado.set("Listo.")
                        messagebox.showinfo("Audiolibro", f"Terminado:\n{m[1]}")
                        try:
                            if os.name == "nt":
                                os.startfile(str(Path(m[1]).parent))
                        except Exception:
                            pass
                    elif k == "cancelado":
                        self.bloquear(False)
                        self.v_estado.set("Cancelado. Al volver a convertir se reanuda donde se quedo.")
                    elif k == "error":
                        self.bloquear(False)
                        self.v_estado.set("Error.")
                        self.escribir(m[1])
                        try:
                            (APP_DIR / "audiolibro_error.log").write_text(m[1], encoding="utf-8")
                        except Exception:
                            pass
                        messagebox.showerror("Audiolibro", m[1][-900:])
            except queue.Empty:
                pass
            self.root.after(100, self._poll)

    root = tk.Tk()
    app = App(root)
    root.mainloop()
    return app


# ----------------------------------------------------------------------------
def _parse_sel(s, n):
    r = set()
    for parte in s.split(","):
        parte = parte.strip()
        if "-" in parte:
            a, b = parte.split("-")
            r.update(range(int(a), int(b) + 1))
        elif parte:
            r.add(int(parte))
    return {i for i in r if 1 <= i <= n}


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Convierte EPUB/PDF/TXT en audiolibro (Kokoro rapido o Chatterbox alta calidad).")
    ap.add_argument("archivo", nargs="?", help="libro; si se omite se abre la interfaz")
    ap.add_argument("--voz", default="ef_dora",
                    help="ef_dora, em_alex, em_santa (Kokoro) | cb:f1, cb:propia (Chatterbox alta calidad)")
    ap.add_argument("--voz-ref", default="", help="audio de referencia para --voz cb:propia")
    ap.add_argument("--listar-voces", action="store_true", help="muestra las voces disponibles")
    ap.add_argument("--muestra", metavar="SALIDA.wav", help="genera una muestra de la voz y termina")
    ap.add_argument("--sin-normalizar", action="store_true", help="no normalizar el volumen del archivo final")
    ap.add_argument("--precargar-hq", action="store_true", help="descarga el modelo de alta calidad")
    ap.add_argument("--velocidad", type=float, default=1.0)
    ap.add_argument("--formato", choices=list(FORMATOS), default="m4b")
    ap.add_argument("--salida", help="carpeta de salida (por defecto, junto al libro)")
    ap.add_argument("--capitulos", help="p. ej. 1,3-6 (numeros de --listar); por defecto los marcados")
    ap.add_argument("--listar", action="store_true", help="solo muestra los capitulos detectados")
    ap.add_argument("--cpu", action="store_true", help="no usar la GPU")
    ap.add_argument("--sin-titulos", action="store_true", help="no leer el titulo de cada capitulo")
    ap.add_argument("--conservar-temporal", action="store_true")
    ap.add_argument("--diagnostico", action="store_true")
    ap.add_argument("--precargar", action="store_true")
    ap.add_argument("--falso", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.diagnostico:
        return diagnostico()
    if a.precargar:
        return precargar()
    if a.precargar_hq:
        return precargar_hq()
    if a.listar_voces:
        for k, v in VOCES.items():
            print(f"{k:12s} Kokoro      {v}")
        for k, v in CB_VOCES.items():
            print(f"{k:12s} Chatterbox  {v['etiqueta']}")
        return 0
    if a.muestra:
        texto = ("En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía "
                 "un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. "
                 "Zaragoza, cielo y gracias: así suena el castellano de España.")
        r = generar_muestra(texto, a.voz, a.velocidad, not a.cpu, print, a.falso, a.voz_ref, a.muestra)
        print(f"Muestra guardada en {r}")
        return 0
    if not a.archivo:
        try:
            gui()
        except Exception:
            err = traceback.format_exc()
            try:
                (APP_DIR / "audiolibro_error.log").write_text(err, encoding="utf-8")
            except Exception:
                pass
            print(err)
            return 1
        return 0

    libro = leer_libro(a.archivo)
    if a.capitulos:
        sel = _parse_sel(a.capitulos, len(libro.capitulos))
        for i, c in enumerate(libro.capitulos, 1):
            c.seleccionado = i in sel
    if a.listar:
        print(f"{libro.titulo} - {libro.autor}")
        for i, c in enumerate(libro.capitulos, 1):
            print(f"{i:3d} [{'x' if c.seleccionado else ' '}] {c.chars:>8,}  {c.titulo}".replace(",", "."))
        return 0
    cfg = Config(voz=a.voz, voz_ref=a.voz_ref, normalizar=not a.sin_normalizar,
                 velocidad=a.velocidad, formato=a.formato,
                 salida=a.salida or str(Path(a.archivo).resolve().parent),
                 usar_gpu=not a.cpu, leer_titulos=not a.sin_titulos,
                 conservar_temporal=a.conservar_temporal, falso=a.falso)
    ultimo = [0.0]

    def prog(fr, m):
        if time.time() - ultimo[0] > 5 or fr >= 1:
            ultimo[0] = time.time()
            print(f"  {fr * 100:5.1f}%  {m}", flush=True)
    convertir(libro, cfg, print, prog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
