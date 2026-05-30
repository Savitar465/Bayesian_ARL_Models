# Rúbrica — Documentación de Algoritmos en Jupyter Notebook

Guía de estructura obligatoria que debe seguir cada notebook de algoritmo asignado.
Basada en las diapositivas del enunciado (págs. 23–28).

---

## 1. Requerimientos generales

### Objetivos
1. Describir los detalles del algoritmo asignado.
2. Aprender las herramientas:
   - **ChatGPT** — apoyo para investigación y redacción.
   - **[scholar.google.com](https://scholar.google.com)** — búsqueda de referencias académicas.
   - **Bibtex to APA** — conversor de citas.

### Tiempo de Trabajo
- 1 hora de elaboración.
- 10 minutos de presentación oral.

### Pasos
1. Crear una **Jupyter Notebook** en **Google Colab** o **Anaconda**.
2. Tener cuenta en <https://chat.openai.com/chat>.
3. Usar el conversor Bibtex → APA para las referencias.

---

## 2. Estructura obligatoria del Notebook

Cada notebook debe contener **estas secciones, en este orden**:

### `<NOMBRE DEL ALGORITMO>`
Título principal del notebook.

### Descripción
Explicación clara del algoritmo: qué es, qué problema resuelve, en qué contexto se usa.

### Bibtex y Referencias
Cita(s) del paper original u obras de referencia, en formato BibTeX **y** su equivalente APA.

### Tipo de Modelo
Clasificación según cuatro criterios:
- **Método de aprendizaje** — supervisado / no supervisado / por refuerzo / etc.
- **Por parámetros** — paramétrico / no paramétrico.
- **Datos de aprendizaje** — offline (batch) / online (streaming).
- **Resultado del entrenamiento** — modelo matemático / reglas / árbol / red / etc.

### Algoritmo de Entrenamiento
Nombre y descripción del procedimiento que ajusta/entrena el modelo
(p. ej. Gradiente Descendente, Mínimos Cuadrados, generación de candidatos, etc.).

### Supuestos y Restricciones
Condiciones que deben cumplirse para que el algoritmo sea válido o aplicable
(p. ej. linealidad, independencia, soporte mínimo, datos categóricos, etc.).

### Tests / Métricas de validación
Pruebas estadísticas o métricas que validan el modelo
(p. ej. R², Breusch-Pagan, soporte, confianza, lift, etc.).

---

## 3. Ejemplo de referencia: Regresión Lineal

> Las diapositivas 25–28 muestran cómo se completa la estructura usando Regresión Lineal.

| Sección | Contenido |
|---|---|
| Definición | Herramienta matemática para describir la relación entre una variable dependiente y una o más independientes. |
| Bibtex | `@article{fisher1919xv, ...}` (Fisher, 1919). |
| Tipo de Modelo | Regresión · Offline · Matemático · Paramétrico. |
| Alg. Entrenamiento | Gradiente Descendente · Mínimos Cuadrados Ordinarios. |
| Fórmula | $Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i$ |
| Supuestos | Linealidad, independencia de errores, homocedasticidad, normalidad de residuales, no multicolinealidad, no autocorrelación. |
| Tests | Scatter plot, $R^2$, normalidad de residuos, Breusch-Pagan, White. |

---

## 4. Entregable

- Un archivo `.ipynb` por algoritmo asignado.
- Cada sección de la estructura como celda **Markdown**.
- Si el algoritmo lo permite, incluir celdas de **código** con una implementación y métricas sobre un dataset de ejemplo.
- Referencias correctamente formateadas (BibTeX + APA).
