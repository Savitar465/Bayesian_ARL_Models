# -*- coding: utf-8 -*-
"""Genera el notebook del ejercicio Text Extraction G5 (Wikipedia) + pipeline NLP."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

# ----------------------------------------------------------------- 0. Título
md(r"""# Text Extraction — G5: Extracción y procesamiento de texto desde Wikipedia

> **Ejercicio en grupos (Grupo E)** — Dataset: *Text Extraction*. Fuente asignada: **G5 — Extraer datos de Wikipedia** (texto en **inglés**).

Aplicamos el pipeline de **Procesamiento de Lenguaje Natural (NLP)** que pide el enunciado:

1. **Tokenization**, **Stemming**, **Lemmatization** y **Canonización**
2. **Eliminar Stopwords**
3. **Detectar NER** (Named Entity Recognition)
4. **Generar PoS** (Part-of-Speech tagging)
5. **Aplicar BoW y TF-IDF**

**Herramientas:** `requests` (API de Wikipedia), **spaCy** (tokenización, lematización, PoS, NER), **NLTK** (stemming), **scikit-learn** (BoW, TF-IDF).""")

# ----------------------------------------------------------------- 1. Setup
md(r"""## 1. Instalación e imports

> En **Google Colab** o un entorno sin estas librerías, la siguiente celda las instala automáticamente. El modelo de spaCy `en_core_web_sm` se descarga la primera vez.""")
code(r"""# Instalación robusta (Colab / entorno nuevo). Si ya están, no hace nada.
%pip install -q spacy scikit-learn nltk requests

import re
import unicodedata
import requests
import pandas as pd
import matplotlib.pyplot as plt

import spacy
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Cargar el modelo de spaCy en inglés (se descarga si no existe)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

stemmer = PorterStemmer()
print("spaCy:", spacy.__version__, "| modelo:", nlp.meta["name"])
print("Pipeline spaCy:", nlp.pipe_names)""")

# ----------------------------------------------------------------- 2. Extracción
md(r"""## 2. Extracción de datos desde Wikipedia (G5)

Extraemos el texto plano de varios artículos **en inglés** usando la **API oficial de MediaWiki** (`action=query&prop=extracts`). Es la forma correcta y estable de obtener datos de Wikipedia (no requiere *scraping* con Selenium).

> Wikipedia exige una cabecera **User-Agent**; sin ella devuelve *403 Forbidden*.

Descargamos **5 artículos relacionados** para tener un *corpus* (varios documentos), necesario para que **BoW** y **TF-IDF** sean significativos.""")
code(r"""HEADERS = {"User-Agent": "TextExtractionExercise/1.0 (academico; contacto: estudiante@example.com)"}

def extraer_wikipedia(titulo, idioma="en"):
    '''Devuelve el texto plano de un artículo de Wikipedia via la API de MediaWiki.'''
    url = f"https://{idioma}.wikipedia.org/w/api.php"
    params = {
        "format": "json", "action": "query", "prop": "extracts",
        "explaintext": 1, "redirects": 1, "titles": titulo,
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    pagina = next(iter(r.json()["query"]["pages"].values()))
    return pagina["title"], pagina.get("extract", "")

TITULOS = ["Artificial intelligence", "Machine learning",
           "Natural language processing", "Computer vision", "Robotics"]

corpus_raw = {}
for t in TITULOS:
    titulo, texto = extraer_wikipedia(t)
    corpus_raw[titulo] = texto
    print(f"  {titulo:32} -> {len(texto):>7,} caracteres")

# Artículo principal sobre el que haremos el pipeline detallado
TITULO_PRINCIPAL = "Artificial intelligence"
texto_principal = corpus_raw[TITULO_PRINCIPAL]
print(f"\nArtículo principal: {TITULO_PRINCIPAL} ({len(texto_principal):,} caracteres)")""")

md("Veamos el inicio del texto crudo extraído:")
code(r"""print(texto_principal[:600], "...")""")

# ----------------------------------------------------------------- 3. Canonización
md(r"""## 3. Canonización / Normalización

La **canonización** lleva el texto a una forma estándar para reducir el ruido y que dos formas equivalentes se traten igual. Aplicamos:

- **Normalización Unicode** (NFKC) — unifica caracteres equivalentes.
- **Minúsculas** (*lowercasing*).
- Eliminación de **referencias entre corchetes** (`[1]`, `[note 2]`), **números** y **puntuación**.
- **Colapso de espacios** en blanco.""")
code(r"""def canonizar(texto):
    texto = unicodedata.normalize("NFKC", texto)      # normalización Unicode
    texto = texto.lower()                              # minúsculas
    texto = re.sub(r"\[[^\]]*\]", " ", texto)          # quita [1], [note] ...
    texto = re.sub(r"[^a-z\s]", " ", texto)            # solo letras y espacios
    texto = re.sub(r"\s+", " ", texto).strip()         # colapsa espacios
    return texto

texto_canonico = canonizar(texto_principal)
print("ANTES :", texto_principal[:160].replace(chr(10), " "))
print()
print("DESPUÉS:", texto_canonico[:160])""")

# ----------------------------------------------------------------- 4. Tokenización
md(r"""## 4. Tokenización

**Tokenizar** = dividir el texto en unidades (*tokens*): palabras y oraciones. Usamos spaCy, que además detecta límites de oración.""")
code(r"""# Procesamos con spaCy el texto ORIGINAL (conserva mayúsculas y puntuación,
# necesarias para PoS y NER). Para acelerar, tomamos una porción representativa.
doc = nlp(texto_principal[:20000])

tokens_palabra = [t.text for t in doc if not t.is_space]
oraciones = list(doc.sents)

print(f"Nº de oraciones (en la porción analizada): {len(oraciones)}")
print(f"Nº de tokens: {len(tokens_palabra)}")
print(f"\nPrimeros 25 tokens:\n{tokens_palabra[:25]}")
print(f"\nPrimera oración:\n{oraciones[0].text.strip()}")""")

# ----------------------------------------------------------------- 5. Stopwords
md(r"""## 5. Eliminación de Stopwords

Las **stopwords** son palabras muy frecuentes y poco informativas (*the, is, of, and...*). spaCy las marca con `token.is_stop`. También quitamos puntuación y espacios.""")
code(r"""tokens_utiles = [t.text.lower() for t in doc
                 if not t.is_stop and not t.is_punct and not t.is_space and t.is_alpha]

print(f"Tokens antes : {len([t for t in doc if not t.is_space])}")
print(f"Tokens después de quitar stopwords/puntuación: {len(tokens_utiles)}")
print(f"\nEjemplo de stopwords detectadas: "
      f"{sorted({t.text.lower() for t in doc if t.is_stop})[:15]}")
print(f"\nTokens útiles (primeros 25):\n{tokens_utiles[:25]}")""")

# ----------------------------------------------------------------- 6. Stemming
md(r"""## 6. Stemming

El **stemming** recorta la palabra a su *raíz* de forma heurística (puede no ser una palabra real). Usamos el **algoritmo de Porter** (NLTK).""")
code(r"""ejemplos = ["learning", "learned", "studies", "studying", "organization",
            "computers", "computing", "happily", "running", "better"]
tabla_stem = pd.DataFrame({
    "palabra": ejemplos,
    "stem (Porter)": [stemmer.stem(w) for w in ejemplos],
})
tabla_stem""")

# ----------------------------------------------------------------- 7. Lemmatization
md(r"""## 7. Lemmatization

La **lematización** reduce la palabra a su forma de diccionario (*lema*), usando contexto y morfología → siempre produce una palabra real. Comparamos **stemming vs lemmatization**:""")
code(r"""comparacion = []
for w in ejemplos:
    lema = nlp(w)[0].lemma_
    comparacion.append({"palabra": w, "stem (Porter)": stemmer.stem(w), "lema (spaCy)": lema})
pd.DataFrame(comparacion)""")

# ----------------------------------------------------------------- 8. PoS
md(r"""## 8. PoS — Part-of-Speech tagging

El **etiquetado gramatical** asigna a cada token su categoría (sustantivo, verbo, adjetivo...). spaCy da la etiqueta universal (`pos_`) y la detallada (`tag_`).""")
code(r"""filas = [{"token": t.text, "lema": t.lemma_, "PoS": t.pos_,
          "tag": t.tag_, "descripción": spacy.explain(t.pos_)}
         for t in doc[:15] if not t.is_space]
display(pd.DataFrame(filas))

# Frecuencia de categorías gramaticales
from collections import Counter
pos_freq = Counter(t.pos_ for t in doc if not t.is_space and not t.is_punct)
pos_top = dict(pos_freq.most_common(10))

plt.figure(figsize=(9, 4))
plt.bar(pos_top.keys(), pos_top.values(), color="#4c72b0")
plt.title("Frecuencia de categorías gramaticales (PoS) — " + TITULO_PRINCIPAL)
plt.ylabel("Frecuencia"); plt.xticks(rotation=30)
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------- 9. NER
md(r"""## 9. NER — Named Entity Recognition

El **reconocimiento de entidades nombradas** detecta y clasifica nombres propios: personas (PERSON), organizaciones (ORG), lugares (GPE), fechas (DATE), etc.""")
code(r"""entidades = [{"entidad": ent.text, "tipo": ent.label_,
              "descripción": spacy.explain(ent.label_)} for ent in doc.ents]
df_ent = pd.DataFrame(entidades).drop_duplicates().reset_index(drop=True)
print(f"Total de entidades detectadas: {len(doc.ents)} "
      f"({df_ent.shape[0]} únicas)\n")
display(df_ent.head(20))

# Conteo por tipo de entidad
ner_freq = Counter(ent.label_ for ent in doc.ents)
plt.figure(figsize=(9, 4))
plt.bar(ner_freq.keys(), ner_freq.values(), color="#dd8452")
plt.title("Entidades por tipo (NER) — " + TITULO_PRINCIPAL)
plt.ylabel("Frecuencia"); plt.xticks(rotation=30)
plt.tight_layout(); plt.show()""")

md("**Visualización de NER** sobre las primeras oraciones (spaCy `displacy`):")
code(r"""from spacy import displacy
from IPython.display import HTML

doc_corto = nlp(" ".join(s.text for s in list(doc.sents)[:4]))
html = displacy.render(doc_corto, style="ent", jupyter=False)
HTML(html)""")

# ----------------------------------------------------------------- 10. BoW
md(r"""## 10. BoW — Bag of Words

El modelo **Bolsa de Palabras** representa cada documento como un vector con la **frecuencia** de cada palabra del vocabulario (ignora el orden). Lo aplicamos sobre el **corpus de 5 artículos** ya canonizados.""")
code(r"""# Canonizamos todo el corpus
docs = [canonizar(t) for t in corpus_raw.values()]
nombres_docs = list(corpus_raw.keys())

bow_vec = CountVectorizer(stop_words="english", min_df=2)
X_bow = bow_vec.fit_transform(docs)
vocab = bow_vec.get_feature_names_out()

print(f"Documentos: {X_bow.shape[0]} | Tamaño del vocabulario: {X_bow.shape[1]}")
df_bow = pd.DataFrame(X_bow.toarray(), index=nombres_docs, columns=vocab)

# Top 15 palabras más frecuentes en todo el corpus
top_bow = df_bow.sum().sort_values(ascending=False).head(15)
plt.figure(figsize=(10, 4))
plt.bar(top_bow.index, top_bow.values, color="#55a868")
plt.title("Top 15 palabras por frecuencia (BoW) — corpus completo")
plt.ylabel("Frecuencia total"); plt.xticks(rotation=45, ha="right")
plt.tight_layout(); plt.show()

print("\nMatriz documento-término (muestra 8 columnas):")
display(df_bow[top_bow.index[:8]])""")

# ----------------------------------------------------------------- 11. TF-IDF
md(r"""## 11. TF-IDF — Term Frequency × Inverse Document Frequency

**TF-IDF** pondera cada palabra por su frecuencia en el documento (**TF**) pero la penaliza si aparece en muchos documentos (**IDF**). Así resalta las palabras **distintivas** de cada artículo.""")
code(r"""tfidf_vec = TfidfVectorizer(stop_words="english", min_df=2)
X_tfidf = tfidf_vec.fit_transform(docs)
terminos = tfidf_vec.get_feature_names_out()
df_tfidf = pd.DataFrame(X_tfidf.toarray(), index=nombres_docs, columns=terminos)

print("Palabras más DISTINTIVAS de cada artículo (mayor TF-IDF):\n")
for doc_nombre in nombres_docs:
    top = df_tfidf.loc[doc_nombre].sort_values(ascending=False).head(8)
    print(f"• {doc_nombre}:")
    print("   " + ", ".join(f"{w} ({v:.3f})" for w, v in top.items()))
    print()""")

code(r"""# Comparación visual: top términos TF-IDF del artículo principal
top_tfidf = df_tfidf.loc[TITULO_PRINCIPAL].sort_values(ascending=False).head(15)
plt.figure(figsize=(10, 4))
plt.bar(top_tfidf.index, top_tfidf.values, color="#c44e52")
plt.title("Top 15 términos por TF-IDF — " + TITULO_PRINCIPAL)
plt.ylabel("Peso TF-IDF"); plt.xticks(rotation=45, ha="right")
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------- 12. Conclusión
md(r"""## 12. Conclusión

Completamos el ejercicio **G5 — Text Extraction desde Wikipedia** aplicando todo el pipeline de NLP solicitado:

| Paso | Herramienta | Qué hicimos |
|---|---|---|
| **Extracción** | `requests` + API de Wikipedia | 5 artículos en inglés a texto plano |
| **Canonización** | `re` + `unicodedata` | minúsculas, sin puntuación/números, Unicode NFKC |
| **Tokenización** | spaCy | división en oraciones y tokens |
| **Stopwords** | spaCy (`is_stop`) | eliminación de palabras vacías |
| **Stemming** | NLTK (Porter) | raíz heurística de las palabras |
| **Lemmatization** | spaCy | forma de diccionario (lema) |
| **PoS** | spaCy | etiquetado gramatical + frecuencias |
| **NER** | spaCy | entidades (PERSON, ORG, GPE, DATE...) |
| **BoW** | scikit-learn | matriz documento-término por frecuencia |
| **TF-IDF** | scikit-learn | términos distintivos por documento |

**Observaciones:**
- **Stemming vs Lemmatization:** el stemming es más agresivo y puede generar raíces no válidas (*studi*), mientras que la lematización devuelve palabras reales (*study*).
- **BoW vs TF-IDF:** BoW resalta palabras frecuentes globales; **TF-IDF** resalta lo **característico** de cada artículo (p. ej. *image* en *Computer vision*, *robot* en *Robotics*).
- La extracción vía **API** es más robusta y reproducible que el *web scraping* con Selenium.""")

nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
}
nbf.validate(nb)
out = "ejercicios_clase/text_extraction_wikipedia.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook escrito:", out, "(", len(cells), "celdas )")
