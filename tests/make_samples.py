import os, warnings
warnings.simplefilter("ignore")
from ebooklib import epub
import pymupdf


def crear(carpeta):
    os.makedirs(carpeta, exist_ok=True)

    # ---------------- EPUB ----------------
    b = epub.EpubBook()
    b.set_identifier("t1"); b.set_title("El Hidalgo de Prueba"); b.set_language("es"); b.add_author("Miguel de Prueba")
    cover = epub.EpubHtml(title="Portada", file_name="cover.xhtml", lang="es")
    cover.content = "<html><body><h1>Portada</h1></body></html>"
    leg = epub.EpubHtml(title="Copyright", file_name="legal.xhtml", lang="es")
    leg.content = "<html><body><h1>Copyright</h1><p>Todos los derechos reservados.</p></body></html>"
    def cap(n, t, extra=""):
        c = epub.EpubHtml(title=t, file_name=f"cap{n}.xhtml", lang="es")
        c.content = (f"<html><head><title>{t}</title></head><body><h1>{t}</h1>"
          f"<p>—¿Sabe usted, Sr. García, qué hora es? —preguntó el hidalgo—. No tengo reloj.<sup>1</sup></p>"
          f"<p>En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo "
          f"de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, "
          f"salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos.<br/>"
          f"Consumían las tres partes de su hacienda.</p>"
          f"<p>* * *</p><p>{'Frase larga sin puntos ni nada que la corte pero con palabras suficientes para pasar el limite de caracteres permitido por el modelo de voz ' * 4}</p>{extra}"
          f"<blockquote><p>Cita dentro de blockquote.</p></blockquote><div>Texto en div suelto.</div></body></html>")
        return c
    caps = [cap(1, "Capítulo 1"), cap(2, "Capítulo 2"), cap(3, "Capítulo 3")]
    for c in [cover, leg] + caps: b.add_item(c)
    b.toc = [epub.Link("cover.xhtml", "Portada", "p"), epub.Link("legal.xhtml", "Copyright", "l")] + [epub.Link(c.file_name, c.title, f"c{i}") for i, c in enumerate(caps)]
    b.add_item(epub.EpubNcx()); b.add_item(epub.EpubNav())
    b.spine = ["nav", cover, leg] + caps
    epub.write_epub(os.path.join(carpeta, "prueba.epub"), b)

    # ---------------- PDF ----------------
    doc = pymupdf.open()
    cuerpo = ("El hidalgo se levantó temprano y ex-\ntrajo de la alacena un poco de pan, queso y membrillo. "
              "Luego salió al camino sin pensar en nada más que en la aventura que le esperaba más allá de la colina. ") * 6
    for p in range(12):
        page = doc.new_page()
        page.insert_text((72, 40), "EL HIDALGO DE PRUEBA - Edición 2026", fontsize=9)       # cabecera repetida
        page.insert_textbox(pymupdf.Rect(72, 90, 520, 700), (f"Capitulo {p//4+1}\n\n" if p % 4 == 0 else "") + cuerpo, fontsize=11)
        page.insert_text((290, 810), str(p + 1), fontsize=9)                                  # número de página
    doc.set_toc([[1, "Capitulo 1", 1], [1, "Capitulo 2", 5], [1, "Capitulo 3", 9]])
    doc.set_metadata({"title": "PDF de Prueba", "author": "Autor PDF"})
    doc.save(os.path.join(carpeta, "prueba.pdf"))

    # ---------------- PDF escaneado (sin texto) ----------------
    d2 = pymupdf.open()
    for _ in range(3): d2.new_page()
    d2.save(os.path.join(carpeta, "escaneado.pdf"))
    open(os.path.join(carpeta, "prueba.txt"), "w", encoding="utf-8").write("Capitulo 1\n\nHola mundo. Esto es un texto de prueba largo con varias frases para ver si se parte bien.\n\nCapitulo 2\n\nSegundo capitulo con otro parrafo de texto suficiente para pasar el filtro de tamano minimo establecido en el programa.\n")
    return carpeta


if __name__ == "__main__":
    import sys
    print(crear(sys.argv[1] if len(sys.argv) > 1 else "tests/data"))
