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
    "voxtral": "Máxima naturalidad - Voxtral de Mistral (GPU)",
}
VX_VOCES = {
    "vx:es_female": "Español - mujer (es_female)",
    "vx:es_male": "Español - hombre (es_male)",
    "vx:neutral_female": "Neutra - mujer (neutral_female)",
    "vx:neutral_male": "Neutra - hombre (neutral_male)",
    "vx:casual_female": "Informal - mujer (casual_female)",
    "vx:casual_male": "Informal - hombre (casual_male)",
    "vx:cheerful_female": "Alegre - mujer (cheerful_female)",
}
VX_MAX_CHARS = 400       # trozos algo mas largos: menos llamadas y mejor entonacion
REF_URL = "https://storage.googleapis.com/chatterbox-demo-samples/mtl-v3-single-language-prompts/es-es/"
CB_VOCES = {
    "cb:f1": {"etiqueta": "España (castellano) - mujer", "archivo": "es_es_f1.wav"},
    "cb:propia": {"etiqueta": "Mi propia voz de referencia (elige un audio)", "archivo": None},
}
CB_EXAG, CB_CFG, CB_TEMP = 0.45, 0.5, 0.75   # narracion serena y estable
CB_LOTE = 8              # trozos que Chatterbox sintetiza a la vez en la GPU (AUDIOLIBRO_LOTE lo cambia)
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


def plan_capitulo(cap, leer_titulo, max_chars=MAX_CHARS):
    """Lista de pasos: ('t', texto) = hablar, ('s', segundos) = silencio."""
    pl = []
    if leer_titulo and cap.titulo and re.search(r"\w", cap.titulo):
        tit = limpiar_para_tts(cap.titulo).rstrip(".:") + "."
        for tz in trocear(tit, max_chars):
            pl.append(("t", tz))
        pl.append(("s", 1.2))
    for par in cap.parrafos:
        if ES_SEPARADOR.match(par.strip()):
            pl.append(("s", 1.2))
            continue
        txt = limpiar_para_tts(par)
        if not txt or not re.search(r"\w", txt):
            continue
        for tz in trocear(txt, max_chars):
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
    voz = str(voz)
    if voz.startswith("cb:"):
        return "chatterbox"
    return "voxtral" if voz.startswith("vx:") else "kokoro"


# Ajustes que se recuerdan entre sesiones (rutas del motor Voxtral).
def _f_ajustes():
    return _dir_datos() / "ajustes.json"


def cargar_ajustes():
    try:
        return json.loads(_f_ajustes().read_text(encoding="utf-8"))
    except Exception:
        return {}


def guardar_ajustes(**cambios):
    a = cargar_ajustes()
    a.update({k: v for k, v in cambios.items() if v is not None})
    try:
        _f_ajustes().parent.mkdir(parents=True, exist_ok=True)
        _f_ajustes().write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


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
        r = s3gen.load_state_dict(load_file(str(s3_p)), strict=False)
        faltan = [k for k in r.missing_keys if not k.startswith("tokenizer.")]   # buffers que se calculan solos
        if faltan or r.unexpected_keys:
            raise RuntimeError(f"faltan {len(faltan)} {faltan[:4]}; "
                               f"sobran {len(r.unexpected_keys)} {r.unexpected_keys[:4]}")
        log("Decodificador V3 cargado.")
    except Exception as e:
        log(f"Aviso: el decodificador V3 no encaja ({type(e).__name__}: {str(e)[:400]}); se usa el estándar.")
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
        self.lote_max = 1
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
        env = os.environ.get("AUDIOLIBRO_LOTE")
        self.lote_max = int(env) if env else (CB_LOTE if self.dispositivo == "cuda" else 1)
        log(f"Sintetizando en: {self.dispositivo.upper()} ({nombre}) - motor de alta calidad"
            + (f", {self.lote_max} trozos a la vez" if self.lote_max > 1 else ""))

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

    @staticmethod
    def _limite_tokens(t):
        """Tope de tokens de voz (25 por segundo) segun el texto: corta las alucinaciones largas."""
        return min(1000, int(25 * (2.0 * len(t) / 15.0 + 1.0)))

    def lote(self, textos, cancel=None):
        """Sintetiza varios trozos a la vez (agrupados por longitud). Devuelve los audios en el mismo orden."""
        import numpy as np
        resultado = [None] * len(textos)
        pendientes = sorted(range(len(textos)), key=lambda i: -len(textos[i]))
        tam = max(1, self.lote_max)
        while pendientes:
            if cancel is not None and cancel.is_set():
                raise Cancelado()
            grupo = pendientes[:tam]
            if tam == 1 or len(grupo) == 1:
                for i in grupo:
                    resultado[i] = self(textos[i])
                pendientes = pendientes[len(grupo):]
                continue
            try:
                self.torch.manual_seed(zlib.crc32(textos[grupo[0]].encode("utf-8")))
                filas = self.m.generate_lote([textos[i] for i in grupo], language_id="es",
                                             exaggeration=CB_EXAG, cfg_weight=CB_CFG, temperature=CB_TEMP,
                                             limite_tokens=self._limite_tokens)
            except Exception as e:
                oom = "out of memory" in str(e).lower()
                if self.dispositivo == "cuda":
                    self.torch.cuda.empty_cache()
                if oom and tam > 1:
                    tam = max(1, tam // 2)
                    self.lote_max = tam
                    self.log(f"Poca memoria de GPU: se reduce a {tam} trozos a la vez.")
                else:
                    self.log(f"Aviso: el modo por lotes falló ({type(e).__name__}: {str(e)[:200]}); "
                             f"se sigue de uno en uno.")
                    tam = self.lote_max = 1
                continue
            for i, (a, completo) in zip(grupo, filas):
                esperado = max(1.0, len(textos[i]) / 15.0)
                dur = len(a) / SAMPLE_RATE
                if completo and 0.4 * esperado - 0.5 <= dur <= 2.0 * esperado + 1.0:
                    resultado[i] = np.asarray(a, dtype="float32")
                else:
                    self.reintentos += 1
                    resultado[i] = self(textos[i])        # repeticion individual con varias semillas
            pendientes = pendientes[len(grupo):]
        return resultado


# ----------------------------------------------------------------------------
# Motor Voxtral (Mistral Voxtral-4B-TTS-2603) a traves del binario CrispASR (formato GGUF propio)
# Ojo: estos GGUF NO sirven en LM Studio / Unsloth / llama.cpp (otra arquitectura).
# ----------------------------------------------------------------------------
def _caminar(base, patron, prof=6):
    base = Path(base)
    if not base.is_dir():
        return
    n0 = len(base.parts)
    for raiz, dirs, ficheros in os.walk(base):
        if len(Path(raiz).parts) - n0 >= prof:
            dirs[:] = []
        for f in ficheros:
            if patron(f.lower()):
                yield Path(raiz) / f


def buscar_gguf_voxtral():
    """GGUF de Voxtral TTS ya descargado: ajustes, variable de entorno o carpetas habituales."""
    for cand in (cargar_ajustes().get("vx_gguf"), os.environ.get("AUDIOLIBRO_VOXTRAL_GGUF")):
        if cand and Path(cand).is_file():
            return str(cand)
    home = Path.home()
    sitios = [DIR_MODELOS / "voxtral", home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models",
              home / ".cache" / "huggingface" / "hub", home / ".unsloth", home / ".cache" / "unsloth",
              home / "Downloads", home / "Descargas"]
    hallados = []
    for sitio in sitios:
        for f in _caminar(sitio, lambda n: n.endswith(".gguf") and "voxtral" in n and "tts" in n):
            hallados.append(f)
    orden = ("q8_0", "f16", "q6_k", "q5_k", "q4_k")       # calidad primero: caben de sobra en 16 GB

    def peso(f):
        n = f.name.lower()
        return next((i for i, q in enumerate(orden) if q in n), len(orden))
    hallados.sort(key=peso)
    return str(hallados[0]) if hallados else ""


def buscar_crispasr():
    exe = "crispasr.exe" if os.name == "nt" else "crispasr"
    for cand in (cargar_ajustes().get("vx_exe"), os.environ.get("AUDIOLIBRO_CRISPASR"),
                 _dir_datos() / "bin" / exe, APP_DIR / exe, shutil.which("crispasr")):
        if cand and Path(cand).is_file():
            return str(cand)
    return ""


def _http(url, clave="", cuerpo=None, timeout=30):
    """GET/POST JSON sencillo (sin dependencias). Devuelve (bytes, content_type)."""
    import urllib.request
    import urllib.error
    cab = {"User-Agent": "Audiolibro/2"}
    if clave:
        cab["Authorization"] = f"Bearer {clave}"
    datos = None
    if cuerpo is not None:
        datos = json.dumps(cuerpo).encode("utf-8")
        cab["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=datos, headers=cab)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"El servidor respondió {e.code}: {e.read()[:400].decode('utf-8', 'replace')}")


def buscar_servidor_unsloth():
    """URL base del servidor de Unsloth Studio (ajustes, entorno o puertos habituales)."""
    aj = cargar_ajustes()
    clave = aj.get("vx_clave") or os.environ.get("UNSLOTH_API_KEY", "")
    cands = [aj.get("vx_url"), os.environ.get("AUDIOLIBRO_VOXTRAL_URL"),
             "http://127.0.0.1:8888", "http://127.0.0.1:8000"]
    for u in cands:
        if not u:
            continue
        u = u.rstrip("/")
        try:
            _http(u + "/v1/models", clave, timeout=2)
            return u
        except RuntimeError:
            return u                      # responde (p. ej. 401 sin clave): es el servidor
        except Exception:
            continue
    return ""


def _modelo_api(url, clave):
    aj = cargar_ajustes()
    if aj.get("vx_modelo"):
        return aj["vx_modelo"]
    datos, _ = _http(url + "/v1/models", clave)
    j = json.loads(datos.decode("utf-8"))
    ids = [m.get("id", "") for m in (j.get("data", j) if isinstance(j, dict) else j)]
    for i in ids:
        if "voxtral" in i.lower():
            return i
    if ids:
        return ids[0]
    raise RuntimeError("Unsloth no tiene ningún modelo cargado: carga Voxtral TTS en Unsloth Studio.")


def voces_servidor(url, clave):
    """Voces que expone el servidor: [(id, nombre)]."""
    datos, _ = _http(url + "/v1/audio/voices", clave, timeout=10)
    j = json.loads(datos.decode("utf-8"))
    if isinstance(j, dict):
        j = next((j[k] for k in ("voices", "data", "items") if isinstance(j.get(k), list)), [])
    salida = []
    for v in j:
        if isinstance(v, str):
            salida.append((v, v))
        elif isinstance(v, dict):
            vid = v.get("id") or v.get("voice_id") or v.get("name")
            if vid:
                nombre = str(v.get("name") or vid)
                idioma = v.get("language") or v.get("lang") or ""
                salida.append((str(vid), nombre + (f"  ({idioma})" if idioma else "")))
    return salida


def es_voz_espanola(vid, nombre=""):
    t = f"{vid} {nombre}".lower()
    return bool(re.search(r"(^|[\W_])es([\W_]|$)|spanish|espa[nñ]|castellano|spain", t))


def campos_speech(url, clave):
    """Campos que admite POST /v1/audio/speech segun /openapi.json del servidor ({} si no se puede leer)."""
    try:
        datos, _ = _http(url + "/openapi.json", clave, timeout=10)
        api = json.loads(datos.decode("utf-8"))
        cuerpo = api.get("paths", {}).get("/v1/audio/speech", {}).get("post", {}).get("requestBody", {})
        esquema = (cuerpo.get("content", {}).get("application/json", {}) or {}).get("schema", {})
        ref = esquema.get("$ref", "")
        if ref:
            esquema = api.get("components", {}).get("schemas", {}).get(ref.split("/")[-1], esquema)
        props = esquema.get("properties", {})
        return props if isinstance(props, dict) else {}
    except Exception:
        return {}


def cuerpo_voxtral(props, modelo, texto, voz, con_idioma=True):
    """Cuerpo de la peticion: se adapta a los nombres de campo reales del servidor."""
    cuerpo = {"model": modelo, "input": texto, "voice": voz, "response_format": "wav"}
    for k, v in (props or {}).items():
        lk, tipo = k.lower(), (v or {}).get("type")
        if k in cuerpo:
            continue
        if ("voice" in lk or "speaker" in lk) and tipo in (None, "string") and "sample" not in lk \
                and "clon" not in lk and "ref" not in lk:
            cuerpo[k] = voz                      # p. ej. voice_id: la misma voz en todos los campos de voz
        elif con_idioma and lk in ("language", "lang", "language_id", "lang_code"):
            enum = [e for e in ((v or {}).get("enum") or []) if isinstance(e, str)]
            cuerpo[k] = next((e for e in enum if e.lower() in ("es", "spanish", "español", "es-es")),
                             enum[0] if enum else "Spanish")
        elif "step" in lk and tipo == "integer":
            cuerpo[k] = 7                        # los mismos pasos que usa el panel de Unsloth
    if con_idioma and not any(k.lower() in ("language", "lang", "language_id", "lang_code") for k in cuerpo):
        cuerpo["language"] = "Spanish"
    return cuerpo


def diagnostico_voxtral(url, clave):
    """Texto con lo que responde el servidor: voces en bruto y campos admitidos por /v1/audio/speech."""
    lineas = []
    try:
        datos, _ = _http(url + "/v1/audio/voices", clave, timeout=10)
        lineas.append("GET /v1/audio/voices -> " + datos.decode("utf-8", "replace")[:700])
    except Exception as e:
        lineas.append(f"GET /v1/audio/voices -> {e}")
    try:
        datos, _ = _http(url + "/openapi.json", clave, timeout=10)
        api = json.loads(datos.decode("utf-8"))
        cuerpo = api.get("paths", {}).get("/v1/audio/speech", {}).get("post", {}).get("requestBody", {})
        esquema = (cuerpo.get("content", {}).get("application/json", {}) or {}).get("schema", {})
        ref = esquema.get("$ref", "")
        if ref:
            esquema = api.get("components", {}).get("schemas", {}).get(ref.split("/")[-1], esquema)
        props = esquema.get("properties", {})
        lineas.append("Campos de /v1/audio/speech: " + (", ".join(
            f"{k} ({v.get('type', '?')})" for k, v in props.items()) or json.dumps(cuerpo)[:600]))
    except Exception as e:
        lineas.append(f"/openapi.json -> {e}")
    return "\n".join(lineas)


def sondear_voxtral(log=print):
    """Muestra lo que ve la aplicacion en el servidor (para diagnosticar)."""
    aj = cargar_ajustes()
    clave = aj.get("vx_clave") or os.environ.get("UNSLOTH_API_KEY", "")
    url = buscar_servidor_unsloth()
    if not url:
        log("No encuentro el servidor de Unsloth (probé 127.0.0.1:8888 y :8000).")
        return 1
    log(f"Servidor: {url}  (clave {'sí' if clave else 'NO configurada'})")
    try:
        datos, _ = _http(url + "/v1/models", clave, timeout=10)
        log(f"/v1/models: {datos.decode('utf-8', 'replace')[:1500]}")
    except Exception as e:
        log(f"/v1/models: {e}")
    log(diagnostico_voxtral(url, clave))
    return 0


class SintVoxtral:
    """Voxtral TTS: via el servidor de Unsloth Studio (/v1/audio/speech) o, si no, con CrispASR."""
    def __init__(self, voz, velocidad, usar_gpu=True, log=print, falso=False):
        self.voz, self.velocidad, self.falso, self.log = voz, velocidad, falso, log
        self.dispositivo = "falso"
        self.vacios = 0
        self.reintentos = 0
        if falso:
            return
        self.nombre_voz = str(voz)[3:]
        self.usar_gpu = usar_gpu
        aj = cargar_ajustes()
        self.clave = aj.get("vx_clave") or os.environ.get("UNSLOTH_API_KEY", "")
        self.url = buscar_servidor_unsloth()
        if self.url:
            self.modo = "api"
            if not self.clave:
                raise RuntimeError(
                    "Unsloth pide una clave API. En Unsloth Studio ve a Settings > API, crea una clave "
                    "(empieza por sk-unsloth-) y pégala en el campo 'Clave API' de Audiolibro.")
            try:
                self.modelo = _modelo_api(self.url, self.clave)
                self.props = campos_speech(self.url, self.clave)
                self._elegir_voz()
            except RuntimeError as e:
                if "401" in str(e):
                    raise RuntimeError("Unsloth rechaza la clave API (401). Crea otra en Settings > API y "
                                       "pégala de nuevo en 'Clave API'.")
                raise
            self.dispositivo = "cuda"
            ejemplo = cuerpo_voxtral(self.props, self.modelo, "...", self.nombre_voz)
            log(f"Voxtral vía Unsloth ({self.url}), modelo {self.modelo}, voz {self.nombre_voz}")
            log("Campos que se envían: " + ", ".join(f"{k}={v!r}" for k, v in ejemplo.items() if k != "input"))
            return
        self.modo = "cli"
        self.exe = buscar_crispasr()
        if not self.exe:
            raise RuntimeError(
                "No encuentro Voxtral: abre Unsloth Studio, carga Voxtral TTS, crea una clave en "
                "Ajustes > API y pégala aquí; o indica crispasr.exe.")
        self.gguf = buscar_gguf_voxtral()
        self.dispositivo = "cuda" if usar_gpu else "cpu"
        log(f"Voxtral: {Path(self.gguf).name if self.gguf else 'auto'} con {Path(self.exe).name}, "
            f"voz {self.nombre_voz}")

    def _elegir_voz(self):
        """Comprueba que la voz existe en el servidor (si no, Voxtral usaria una voz inglesa por defecto)."""
        self.con_idioma = True
        try:
            voces = voces_servidor(self.url, self.clave)
        except Exception:
            voces = []
        if not voces:
            return
        quiero = self.nombre_voz.lower()
        for vid, nombre in voces:
            if quiero in (vid.lower(), nombre.split("  (")[0].lower()):
                self.nombre_voz = vid
                return
        esp = [v for v in voces if es_voz_espanola(*v)]
        if esp:
            self.log(f"La voz '{self.nombre_voz}' no existe en tu servidor; se usa '{esp[0][0]}'.")
            self.nombre_voz = esp[0][0]
            return
        raise RuntimeError(
            f"La voz '{self.nombre_voz}' no existe en tu servidor de Unsloth y no hay ninguna voz en español. "
            f"Voces disponibles: {', '.join(v[0] for v in voces[:20])}. Elige una en la lista de voces.")

    def _leer(self, datos):
        import numpy as np
        import soundfile as sf
        import io
        a, sr = sf.read(io.BytesIO(datos), dtype="float32", always_2d=False)
        if getattr(a, "ndim", 1) > 1:
            a = a.mean(axis=1)
        if sr != SAMPLE_RATE and len(a):
            n = int(round(len(a) * SAMPLE_RATE / sr))
            a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype("float32")
        return np.asarray(a, dtype="float32").reshape(-1)

    def _sintetizar(self, texto, semilla):
        if self.modo == "api":
            cuerpo = cuerpo_voxtral(getattr(self, "props", {}), self.modelo, texto, self.nombre_voz,
                                    getattr(self, "con_idioma", True))
            try:
                datos, tipo = _http(self.url + "/v1/audio/speech", self.clave, cuerpo, timeout=900)
            except RuntimeError as e:
                if getattr(self, "con_idioma", True) and ("422" in str(e) or "400" in str(e)):
                    self.con_idioma = False                   # este servidor no admite el campo: se reintenta sin él
                    cuerpo = cuerpo_voxtral(getattr(self, "props", {}), self.modelo, texto, self.nombre_voz, False)
                    datos, tipo = _http(self.url + "/v1/audio/speech", self.clave, cuerpo, timeout=900)
                else:
                    raise
            if "json" in tipo.lower():
                raise RuntimeError(f"Respuesta inesperada de Unsloth: {datos[:300].decode('utf-8', 'replace')}")
            return self._leer(datos)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "o.wav"
            cmd = [self.exe, "--backend", "voxtral-tts"]
            cmd += ["-m", self.gguf] if self.gguf else ["-m", "auto", "--auto-download"]
            cmd += ["--tts", texto, "--voice", self.nombre_voz, "--tts-output", str(out), "--seed", str(semilla)]
            env = dict(os.environ)
            if not self.usar_gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, env=env, timeout=900,
                               creationflags=0x08000000 if os.name == "nt" else 0)
            if r.returncode != 0 or not out.exists():
                cola = (r.stderr or r.stdout or b"").decode("utf-8", "replace")[-600:]
                raise RuntimeError(f"CrispASR falló (código {r.returncode}): {cola}")
            return self._leer(out.read_bytes())

    def __call__(self, texto):
        import numpy as np
        if self.falso:
            return _tono(texto)
        esperado = max(1.0, len(texto) / 15.0)
        mejor, mejor_err = None, None
        for intento in range(3):
            a = self._sintetizar(texto, zlib.crc32(texto.encode("utf-8")) % 100000 + intento)
            dur = len(a) / SAMPLE_RATE
            err = abs(math.log(max(dur, 0.05) / esperado))
            if mejor is None or err < mejor_err:
                mejor, mejor_err = a, err
            if 0.4 * esperado - 0.5 <= dur <= 2.0 * esperado + 1.0:
                break
            self.reintentos += 1
        if mejor is None or not len(mejor):
            self.vacios += 1
            return np.zeros(0, dtype="float32")
        return mejor


def Sintetizador(voz, velocidad, usar_gpu=True, log=print, falso=False, voz_ref=""):
    if motor_de(voz) == "chatterbox":
        return SintChatterbox(voz, velocidad, usar_gpu, log, falso, voz_ref)
    if motor_de(voz) == "voxtral":
        return SintVoxtral(voz, velocidad, usar_gpu, log, falso)
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


def _pasos_sintetizados(pasos, sint, cancel):
    """Recorre los pasos de un capitulo y entrega (tipo, valor, audio). Con motor por lotes, sintetiza
    ventanas de varios trozos a la vez; si no, uno a uno."""
    if getattr(sint, "lote_max", 1) <= 1:
        for k, v in pasos:
            if cancel.is_set():
                raise Cancelado()
            yield k, v, (sint(v) if k == "t" else None)
        return
    ventana = max(8, sint.lote_max * 3)
    i, n = 0, len(pasos)
    while i < n:
        j, textos = i, []
        while j < n and len(textos) < ventana:
            if pasos[j][0] == "t":
                textos.append(pasos[j][1])
            j += 1
        audios = iter(sint.lote(textos, cancel))
        for k, v in pasos[i:j]:
            yield k, v, (next(audios) if k == "t" else None)
        i = j


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

    maxc = VX_MAX_CHARS if motor_de(cfg.voz) == "voxtral" else MAX_CHARS
    planes = {i: plan_capitulo(c, cfg.leer_titulos, maxc) for i, c in caps}
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
            for k, v, audio in _pasos_sintetizados(planes[i], sint, cancel):
                if cancel.is_set():
                    raise Cancelado()
                if k == "s":
                    f.write(np.zeros(int(SAMPLE_RATE * v), dtype="float32"))
                    continue
                a = acondicionar(audio)
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


BENCH_TEXTOS = [
    "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero.",
    "Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes.",
    "El resto della concluían sayo de velarte, calzas de velludo para las fiestas con sus pantuflos de lo mismo.",
    "Tenía en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza.",
    "Frisaba la edad de nuestro hidalgo con los cincuenta años; era de complexión recia, seco de carnes, enjuto de rostro.",
    "Es, pues, de saber que este sobredicho hidalgo, los ratos que estaba ocioso, que eran los más del año, se daba a leer libros de caballerías.",
    "Con tanta afición y gusto, que olvidó casi de todo punto el ejercicio de la caza, y aun la administración de su hacienda.",
    "Llegó a vender muchas fanegas de tierra de sembradura para comprar libros de caballerías en que leer.",
]


def bench(lotes=(1, 4, 8), usar_gpu=True, log=print):
    """Mide la velocidad real de Chatterbox con distintos tamaños de lote (16 trozos de ~130 caracteres)."""
    textos = (BENCH_TEXTOS * 2)[:16]
    sint = Sintetizador("cb:f1", 1.0, usar_gpu, log)
    if not hasattr(sint, "m"):
        log("El banco de pruebas necesita el motor Chatterbox.")
        return 1
    sint.lote_max = 1
    sint(textos[0])                                         # calentamiento
    for n in lotes:
        sint.lote_max = n
        sint.reintentos = 0
        if sint.dispositivo == "cuda":
            sint.torch.cuda.synchronize()
            sint.torch.cuda.reset_peak_memory_stats()
        t = time.time()
        audios = sint.lote(textos)
        if sint.dispositivo == "cuda":
            sint.torch.cuda.synchronize()
        dt = time.time() - t
        seg = sum(len(a) for a in audios) / SAMPLE_RATE
        vram = f", VRAM máx {sint.torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if sint.dispositivo == "cuda" else ""
        log(f"lote {n:2d}: {seg:.0f} s de audio en {dt:.0f} s -> {seg / dt:.2f}x tiempo real, "
            f"{sint.reintentos} repetidos{vram}")
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

    # Paleta calida: crema de fondo, indigo profundo en la cabecera y naranja como acento.
    C = {"fondo": "#F6F0E4", "tarjeta": "#FFFEFA", "borde": "#E6DAC3", "indigo": "#0F4C81",
         "indigo2": "#2E7DB5", "acento": "#1B9AA6", "acento_h": "#15838E", "acento_p": "#106C75",
         "boton": "#D9693A", "boton_h": "#C25A2E", "boton_p": "#A84D26", "sol": "#E9A23B",
         "texto": "#16344B", "suave": "#6B7F8C", "claro": "#CDE6F0", "fila": "#F8F2E6",
         "gris": "#ECE3D0"}
    FUENTE = "Segoe UI"
    MARCA_SI, MARCA_NO = "✔", "○"

    class App:
        def __init__(self, root):
            self.root = root
            root.title("Audiolibro - EPUB / PDF / TXT a audio")
            root.geometry("1120x700")
            root.minsize(980, 560)
            root.configure(bg=C["fondo"])
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

            self._estilos()

            # --- cabecera
            cab = tk.Frame(root, bg=C["indigo"])
            cab.pack(fill="x")
            tk.Label(cab, text="\U0001F3A7  Audiolibro", bg=C["indigo"], fg="white",
                     font=(FUENTE, 22, "bold")).pack(side="left", padx=(24, 12), pady=(14, 14))
            tk.Label(cab, text="Convierte tus libros en audio con voces naturales, en tu propio PC",
                     bg=C["indigo"], fg=C["claro"], font=(FUENTE, 11)).pack(side="left", pady=(20, 0))
            tk.Frame(root, bg=C["sol"], height=4).pack(fill="x")

            cuerpo = tk.Frame(root, bg=C["fondo"])
            cuerpo.pack(fill="both", expand=True, padx=20, pady=(12, 10))
            cuerpo.columnconfigure(0, weight=11, uniform="c")
            cuerpo.columnconfigure(1, weight=10, uniform="c")
            cuerpo.rowconfigure(0, weight=1)

            izq = tk.Frame(cuerpo, bg=C["fondo"])
            izq.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
            der = tk.Frame(cuerpo, bg=C["fondo"])
            der.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

            # --- 1. Libro
            t1 = self._tarjeta(izq, "1  ·  Tu libro")
            t1.pack(fill="x")
            t1.columnconfigure(0, weight=1)
            ttk.Entry(t1, textvariable=self.v_archivo).grid(row=0, column=0, sticky="ew", padx=(0, 8))
            ttk.Button(t1, text="Examinar...", style="Suave.TButton",
                       command=self.elegir).grid(row=0, column=1)
            ttk.Button(t1, text="Analizar", style="Suave.TButton",
                       command=self.analizar).grid(row=0, column=2, padx=(6, 0))

            # --- capitulos
            t2 = self._tarjeta(izq, "Capítulos  (clic para marcar / desmarcar)")
            t2.pack(fill="both", expand=True, pady=(12, 0))
            t2.columnconfigure(0, weight=1)
            t2.rowconfigure(0, weight=1)
            cols = ("sel", "titulo", "chars")
            self.tree = ttk.Treeview(t2, columns=cols, show="headings", height=6, selectmode="none",
                                     style="Cap.Treeview")
            self.tree.heading("sel", text="Leer")
            self.tree.heading("titulo", text="Capítulo")
            self.tree.heading("chars", text="Caracteres")
            self.tree.column("sel", width=56, anchor="center", stretch=False)
            self.tree.column("titulo", width=360)
            self.tree.column("chars", width=90, anchor="e", stretch=False)
            self.tree.tag_configure("par", background=C["tarjeta"])
            self.tree.tag_configure("impar", background=C["fila"])
            self.tree.bind("<Button-1>", self.alternar)
            sb = ttk.Scrollbar(t2, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=sb.set)
            self.tree.grid(row=0, column=0, sticky="nsew")
            sb.grid(row=0, column=1, sticky="ns")
            bt = tk.Frame(t2, bg=C["tarjeta"])
            bt.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
            ttk.Button(bt, text="Marcar todo", style="Suave.TButton",
                       command=lambda: self.marcar(True)).pack(side="left")
            ttk.Button(bt, text="Desmarcar todo", style="Suave.TButton",
                       command=lambda: self.marcar(False)).pack(side="left", padx=6)

            # --- 2. Voz
            t3 = self._tarjeta(der, "2  ·  Voz")
            t3.pack(fill="x")
            t3.columnconfigure(1, weight=1)
            ttk.Label(t3, text="Motor", style="Card.TLabel").grid(row=0, column=0, sticky="w")
            self.cb_motor = ttk.Combobox(t3, textvariable=self.v_motor, state="readonly",
                                         values=list(MOTORES.values()))
            self.cb_motor.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=3)
            self.cb_motor.bind("<<ComboboxSelected>>", lambda e: self.cambia_motor())
            ttk.Label(t3, text="Voz", style="Card.TLabel").grid(row=1, column=0, sticky="w")
            self.cb_voz = ttk.Combobox(t3, textvariable=self.v_voz, state="readonly")
            self.cb_voz.grid(row=1, column=1, sticky="ew", padx=(10, 6), pady=3)
            self.cb_voz.bind("<<ComboboxSelected>>", lambda e: self.cambia_voz())
            ttk.Button(t3, text="▶  Escuchar", style="Suave.TButton",
                       command=self.muestra).grid(row=1, column=2)
            ttk.Label(t3, text="Referencia", style="Card.TLabel").grid(row=2, column=0, sticky="w")
            self.en_ref = ttk.Entry(t3, textvariable=self.v_ref, state="readonly")
            self.en_ref.grid(row=2, column=1, sticky="ew", padx=(10, 6), pady=3)
            self.bt_ref = ttk.Button(t3, text="Elegir...", style="Suave.TButton", command=self.elegir_ref)
            self.bt_ref.grid(row=2, column=2)
            aj = cargar_ajustes()
            self.v_vx_url = tk.StringVar(value=aj.get("vx_url") or "http://127.0.0.1:8888")
            self.v_vx_clave = tk.StringVar(value=aj.get("vx_clave", ""))
            self.fila_vx = []
            e1 = ttk.Label(t3, text="Servidor Unsloth", style="Card.TLabel")
            e2 = ttk.Entry(t3, textvariable=self.v_vx_url)
            e3 = ttk.Button(t3, text="Cargar voces", style="Suave.TButton", command=self.cargar_voces_vx)
            e1.grid(row=3, column=0, sticky="w")
            e2.grid(row=3, column=1, sticky="ew", padx=(10, 6), pady=3)
            e3.grid(row=3, column=2)
            self.fila_vx.append((e1, e2, e3))
            f1 = ttk.Label(t3, text="Clave API", style="Card.TLabel")
            f2 = ttk.Entry(t3, textvariable=self.v_vx_clave, show="\u2022")
            f1.grid(row=4, column=0, sticky="w")
            f2.grid(row=4, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=3)
            self.fila_vx.append((f1, f2))
            ttk.Label(t3, textvariable=self.v_info, style="Nota.TLabel", wraplength=470,
                      justify="left").grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

            # --- 3. Ajustes
            t4 = self._tarjeta(der, "3  ·  Ajustes")
            t4.pack(fill="x", pady=(12, 0))
            t4.columnconfigure(1, weight=1)
            ttk.Label(t4, text="Velocidad", style="Card.TLabel").grid(row=0, column=0, sticky="w")
            fv = tk.Frame(t4, bg=C["tarjeta"])
            fv.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=3)
            fv.columnconfigure(0, weight=1)
            ttk.Scale(fv, from_=0.7, to=1.4, variable=self.v_vel, style="Acento.Horizontal.TScale",
                      command=lambda _=None: self.lbl_vel.config(text=f"{self.v_vel.get():.2f}x")
                      ).grid(row=0, column=0, sticky="ew")
            self.lbl_vel = ttk.Label(fv, text="1.00x", style="Valor.TLabel", width=6, anchor="e")
            self.lbl_vel.grid(row=0, column=1, padx=(8, 0))
            ttk.Label(t4, text="Formato", style="Card.TLabel").grid(row=1, column=0, sticky="w")
            ttk.Combobox(t4, textvariable=self.v_fmt, state="readonly",
                         values=list(FORMATOS.values())).grid(row=1, column=1, columnspan=2, sticky="ew",
                                                              padx=(10, 0), pady=3)
            ttk.Label(t4, text="Guardar en", style="Card.TLabel").grid(row=2, column=0, sticky="w")
            ttk.Entry(t4, textvariable=self.v_salida).grid(row=2, column=1, sticky="ew", padx=(10, 6), pady=3)
            ttk.Button(t4, text="Cambiar...", style="Suave.TButton",
                       command=self.elegir_salida).grid(row=2, column=2)
            self.chk_gpu = ttk.Checkbutton(t4, text="Usar la GPU (CUDA)", variable=self.v_gpu,
                                           style="Card.TCheckbutton")
            self.chk_gpu.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
            ttk.Label(t4, textvariable=self.v_gpu_txt, style="Nota.TLabel").grid(row=3, column=1, columnspan=2,
                                                                                 sticky="e", pady=(6, 0))
            ttk.Checkbutton(t4, text="Leer en voz alta el título de cada capítulo", variable=self.v_tit,
                            style="Card.TCheckbutton").grid(row=4, column=0, columnspan=3, sticky="w")
            ttk.Checkbutton(t4, text="Audio limpio: volumen uniforme y sin graves molestos",
                            variable=self.v_norm, style="Card.TCheckbutton"
                            ).grid(row=5, column=0, columnspan=3, sticky="w")
            self.cambia_motor()

            # --- accion, progreso y registro (ancho completo)
            pie = tk.Frame(root, bg=C["fondo"])
            pie.pack(fill="x", padx=20, pady=(0, 12))
            pie.columnconfigure(0, weight=1)
            ac = tk.Frame(pie, bg=C["fondo"])
            ac.grid(row=0, column=0, sticky="ew")
            self.btn_go = ttk.Button(ac, text="▶   CONVERTIR A AUDIOLIBRO", style="Accion.TButton",
                                     command=self.convertir)
            self.btn_go.pack(side="left")
            self.btn_stop = ttk.Button(ac, text="Cancelar", style="Suave.TButton", command=self.cancelar,
                                       state="disabled")
            self.btn_stop.pack(side="left", padx=12)
            ttk.Label(ac, textvariable=self.v_estado, style="Estado.TLabel").pack(side="left", padx=8)
            self.pb = ttk.Progressbar(pie, maximum=1000, style="Acento.Horizontal.TProgressbar")
            self.pb.grid(row=1, column=0, sticky="ew", pady=(10, 8))
            self.log = tk.Text(pie, height=5, state="disabled", wrap="word", font=("Consolas", 9),
                               bg=C["tarjeta"], fg=C["texto"], relief="flat", bd=0, padx=10, pady=8,
                               highlightthickness=1, highlightbackground=C["borde"],
                               highlightcolor=C["borde"])
            self.log.grid(row=2, column=0, sticky="ew")
            sb_log = ttk.Scrollbar(pie, orient="vertical", command=self.log.yview)
            self.log.configure(yscrollcommand=sb_log.set)
            sb_log.grid(row=2, column=1, sticky="ns")

            threading.Thread(target=self._detectar_gpu, daemon=True).start()
            root.after(100, self._poll)

        # --- aspecto
        def _estilos(self):
            s = ttk.Style(self.root)
            try:
                s.theme_use("clam")
            except Exception:
                pass
            s.configure(".", font=(FUENTE, 10), background=C["tarjeta"], foreground=C["texto"])
            s.configure("Card.TLabel", background=C["tarjeta"], foreground=C["texto"])
            s.configure("Nota.TLabel", background=C["tarjeta"], foreground=C["suave"], font=(FUENTE, 9))
            s.configure("Valor.TLabel", background=C["tarjeta"], foreground=C["acento"],
                        font=(FUENTE, 10, "bold"))
            s.configure("Estado.TLabel", background=C["fondo"], foreground=C["suave"], font=(FUENTE, 10))
            s.configure("TEntry", fieldbackground="#FFFFFF", bordercolor=C["borde"], lightcolor=C["borde"],
                        darkcolor=C["borde"], padding=5)
            s.map("TEntry", bordercolor=[("focus", C["indigo2"])], lightcolor=[("focus", C["indigo2"])])
            s.configure("TCombobox", fieldbackground="#FFFFFF", background=C["gris"], bordercolor=C["borde"],
                        lightcolor=C["borde"], darkcolor=C["borde"], arrowcolor=C["indigo"], padding=5)
            s.map("TCombobox", fieldbackground=[("readonly", "#FFFFFF")],
                  bordercolor=[("focus", C["indigo2"])], foreground=[("readonly", C["texto"])],
                  selectbackground=[("readonly", "#FFFFFF")], selectforeground=[("readonly", C["texto"])])
            self.root.option_add("*TCombobox*Listbox.background", "#FFFFFF")
            self.root.option_add("*TCombobox*Listbox.foreground", C["texto"])
            self.root.option_add("*TCombobox*Listbox.selectBackground", C["indigo2"])
            self.root.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")
            self.root.option_add("*TCombobox*Listbox.font", (FUENTE, 10))
            s.configure("Suave.TButton", background=C["gris"], foreground=C["indigo"], borderwidth=0,
                        focusthickness=0, padding=(12, 6), font=(FUENTE, 10))
            s.map("Suave.TButton", background=[("disabled", "#F3EEE6"), ("pressed", "#D8CDBC"),
                                               ("active", "#E2D8C8")],
                  foreground=[("disabled", "#B3AAA0")])
            s.configure("Accion.TButton", background=C["boton"], foreground="white", borderwidth=0,
                        focusthickness=0, padding=(26, 12), font=(FUENTE, 12, "bold"))
            s.map("Accion.TButton", background=[("disabled", "#EBC8B4"), ("pressed", C["boton_p"]),
                                                ("active", C["boton_h"])],
                  foreground=[("disabled", "#FFFFFF")])
            s.configure("Card.TCheckbutton", background=C["tarjeta"], foreground=C["texto"], padding=3)
            s.map("Card.TCheckbutton", background=[("active", C["tarjeta"])],
                  indicatorcolor=[("selected", C["acento"]), ("!selected", "#FFFFFF")],
                  indicatorbackground=[("selected", C["acento"]), ("!selected", "#FFFFFF")])
            s.configure("Acento.Horizontal.TScale", background=C["tarjeta"], troughcolor=C["gris"],
                        bordercolor=C["tarjeta"], lightcolor=C["acento"], darkcolor=C["acento"])
            s.map("Acento.Horizontal.TScale", background=[("active", C["acento_h"])])
            s.configure("Acento.Horizontal.TProgressbar", troughcolor=C["gris"], background=C["acento"],
                        bordercolor=C["fondo"], lightcolor=C["acento"], darkcolor=C["acento"], thickness=14)
            s.configure("Cap.Treeview", background=C["tarjeta"], fieldbackground=C["tarjeta"],
                        foreground=C["texto"], rowheight=28, borderwidth=0, font=(FUENTE, 10))
            s.configure("Cap.Treeview.Heading", background=C["indigo"], foreground="white", relief="flat",
                        font=(FUENTE, 10, "bold"), padding=(8, 6))
            s.map("Cap.Treeview.Heading", background=[("active", C["indigo2"])])
            s.configure("Vertical.TScrollbar", background=C["gris"], troughcolor=C["tarjeta"],
                        bordercolor=C["tarjeta"], arrowcolor=C["indigo"])

        def _tarjeta(self, padre, titulo):
            """Tarjeta blanca con borde suave y titulo en indigo; devuelve el marco interior."""
            ext = tk.Frame(padre, bg=C["tarjeta"], highlightthickness=1, highlightbackground=C["borde"])
            tk.Label(ext, text=titulo, bg=C["tarjeta"], fg=C["indigo"],
                     font=(FUENTE, 11, "bold")).pack(anchor="w", padx=16, pady=(12, 6))
            interior = tk.Frame(ext, bg=C["tarjeta"])
            interior.pack(fill="both", expand=True, padx=16, pady=(0, 14))
            # el empaquetado del marco exterior lo hace quien llama: reenviamos pack/grid
            interior.pack = ext.pack
            return interior

        # --- utilidades
        def escribir(self, s):
            self.log.config(state="normal")
            self.log.insert("end", s + "\n")
            self.log.see("end")
            self.log.config(state="disabled")

        def voz(self):
            v = self.v_voz.get().split("|")[0].strip()
            if self.motor() == "voxtral" and v and not v.startswith("vx:"):
                v = "vx:" + v          # permite escribir a mano cualquier voz (p. ej. casual_female)
            return v

        def motor(self):
            return next(k for k, v in MOTORES.items() if v == self.v_motor.get())

        def cambia_motor(self):
            vx = self.motor() == "voxtral"
            for fila in self.fila_vx:
                for w in fila:
                    w.grid() if vx else w.grid_remove()
            if self.motor() == "kokoro":
                vals = [f"{k}  |  {v}" for k, v in VOCES.items()]
                self.v_info.set("Rápido: Kokoro, pronuncia con la fonética de España (c/z como 'th'). "
                                "Ideal para libros largos; funciona en CPU y GPU.")
            elif vx:
                vals = [f"{k}  |  {v}" for k, v in VX_VOCES.items()]
                self.v_info.set("Voxtral (Mistral, 4B): voz muy natural, solo para uso personal (licencia no "
                                "comercial). Se ejecuta en Unsloth Studio: ábrelo, carga Voxtral TTS y pega aquí "
                                "la clave de Ajustes > API. Las voces son las incluidas en el modelo (no hace "
                                "falta cargarlas); puedes escribir el nombre de otra en el desplegable.")
            else:
                vals = [f"{k}  |  {v['etiqueta']}" for k, v in CB_VOCES.items()]
                self.v_info.set("Alta calidad: Chatterbox con el paquete de España. Imita la voz y el acento del "
                                "audio de referencia (puedes poner el tuyo). Más lento y necesita GPU; la primera "
                                "vez descarga unos 4,3 GB.")
            self.cb_voz.config(values=vals, state="normal" if vx else "readonly")
            self.v_voz.set(vals[0])
            self.cambia_voz()
            if vx and self.v_vx_clave.get().strip():
                self.cargar_voces_vx()

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

        def cargar_voces_vx(self):
            """Pide a Unsloth sus voces y las ofrece en la lista (la española, preseleccionada)."""
            self.guardar_vx()
            self.v_estado.set("Consultando las voces de Unsloth...")

            def trabajo():
                try:
                    clave = self.v_vx_clave.get().strip()
                    if not clave:
                        raise RuntimeError("Pega primero la clave API de Unsloth (Settings > API).")
                    url = (self.v_vx_url.get().strip() or buscar_servidor_unsloth()).rstrip("/")
                    voces = voces_servidor(url, clave)
                    self.q.put(("vx_voces", voces))
                    if not voces:
                        self.q.put(("vx_diag", diagnostico_voxtral(url, clave)))
                except Exception as e:
                    self.q.put(("vx_error", str(e)))
            threading.Thread(target=trabajo, daemon=True).start()

        def guardar_vx(self):
            if self.motor() == "voxtral":
                guardar_ajustes(vx_url=self.v_vx_url.get().strip(), vx_clave=self.v_vx_clave.get().strip())

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
                self.tree.insert("", "end", iid=str(i), tags=("par" if i % 2 == 0 else "impar",),
                                 values=(MARCA_SI if c.seleccionado else MARCA_NO, c.titulo,
                                         f"{c.chars:,}".replace(",", ".")))

        def alternar(self, ev):
            iid = self.tree.identify_row(ev.y)
            if not iid or not self.libro or self.ocupado:
                return
            c = self.libro.capitulos[int(iid)]
            c.seleccionado = not c.seleccionado
            self.tree.set(iid, "sel", MARCA_SI if c.seleccionado else MARCA_NO)

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
            self.guardar_vx()
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
            self.guardar_vx()
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
                    elif k == "vx_voces":
                        if m[1] and self.motor() == "voxtral":
                            vals = [f"vx:{vid}  |  {nombre}" for vid, nombre in m[1]]
                            self.cb_voz.config(values=vals)
                            esp = [v for v, (vid, nom) in zip(vals, m[1]) if es_voz_espanola(vid, nom)]
                            self.v_voz.set((esp or vals)[0])
                            self.v_estado.set(f"{len(vals)} voces cargadas desde Unsloth"
                                              + (f" ({len(esp)} en español)." if esp else "; ninguna es española."))
                            self.escribir("Voces de Unsloth: " + ", ".join(v[0] for v in m[1]))
                        else:
                            self.v_estado.set("Unsloth no lista voces propias: se usan las predefinidas de Voxtral.")
                    elif k == "vx_diag":
                        self.escribir(m[1])
                        try:
                            self.root.clipboard_clear()
                            self.root.clipboard_append(m[1])
                            self.v_estado.set("Diagnóstico copiado al portapapeles: pégalo en el chat.")
                        except Exception:
                            pass
                    elif k == "vx_error":
                        self.v_estado.set("No se pudieron cargar las voces.")
                        self.escribir(m[1])
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
                    help="ef_dora, em_alex, em_santa (Kokoro) | cb:f1, cb:propia (Chatterbox alta calidad) | vx:es_female, vx:es_male (Voxtral)")
    ap.add_argument("--vx-gguf", default="", help="ruta del GGUF de Voxtral TTS (si no se detecta solo)")
    ap.add_argument("--vx-exe", default="", help="ruta de crispasr.exe (alternativa a Unsloth)")
    ap.add_argument("--vx-url", default="", help="URL del servidor de Unsloth Studio, p. ej. http://127.0.0.1:8888")
    ap.add_argument("--vx-clave", default="", help="clave API de Unsloth (Ajustes > API)")
    ap.add_argument("--vx-modelo", default="", help="id del modelo en Unsloth (por defecto el primero con 'voxtral')")
    ap.add_argument("--bench", action="store_true", help="mide la velocidad de Chatterbox por tamaño de lote")
    ap.add_argument("--lote", default="", help="trozos a la vez en Chatterbox (con --bench, lista: 1,4,8,16)")
    ap.add_argument("--vx-probar", action="store_true", help="muestra modelos y voces que ve el servidor de Unsloth")
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
    if a.vx_gguf or a.vx_exe or a.vx_url or a.vx_clave or a.vx_modelo:
        guardar_ajustes(vx_gguf=a.vx_gguf or None, vx_exe=a.vx_exe or None, vx_url=a.vx_url or None,
                        vx_clave=a.vx_clave or None, vx_modelo=a.vx_modelo or None)
    if a.vx_probar:
        return sondear_voxtral(print)
    if a.bench:
        return bench([int(x) for x in (a.lote or "1,4,8").split(",")], not a.cpu, print)
    if a.lote and "," not in a.lote:
        os.environ["AUDIOLIBRO_LOTE"] = a.lote
    if a.listar_voces:
        for k, v in VOCES.items():
            print(f"{k:12s} Kokoro      {v}")
        for k, v in CB_VOCES.items():
            print(f"{k:12s} Chatterbox  {v['etiqueta']}")
        for k, v in VX_VOCES.items():
            print(f"{k:18s} Voxtral     {v}")
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
