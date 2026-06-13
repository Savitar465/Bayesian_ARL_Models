# 🎤 Guía para presentar el notebook CNN a la clase

> Cómo explicar [cnn_flores.ipynb](cnn_flores.ipynb) en ~10 min, en cristiano y sin morir en el intento. xd

---

## 📦 El dataset: de dónde sale, qué es y cómo lo procesamos

### ¿De dónde viene? (origen)
Usamos **tf_flowers** (el dataset de flores de TensorFlow). Es un set de **fotografías reales** de flores publicado por el equipo de TensorFlow (2019), pensado para tutoriales de clasificación de imágenes con fotos de verdad (no imágenes de juguete como MNIST).

- Son fotos reales tomadas por personas, con **fondos, ángulos e iluminación variados**.
- **No lo descargamos a mano**: `keras.utils.get_file(...)` baja un `.tgz` desde el servidor de Google y lo descomprime en carpetas.
- Cada **carpeta = una clase**. `image_dataset_from_directory` lee las imágenes y usa el **nombre de la carpeta como etiqueta**.

> *"Es un set público de fotos reales de flores que publica el equipo de TensorFlow; lo bajamos con una línea y cada carpeta es una especie."*

### ¿De qué trata? (contenido)
Son **fotos a color de 5 especies de flor**. El objetivo: que la máquina diga **qué flor** hay en la foto.

| Característica | Valor |
|---|---|
| Total de imágenes | **~3 670 fotos reales** |
| Tamaño original | **Variable** (cada foto es distinta) → las redimensionamos a 128×128 |
| Color | **RGB (3 canales)** |
| Clases (5) | margarita, diente de león, rosa, girasol, tulipán |
| Balance | Aproximado (no exactamente igual por clase) |

> *"Cada dato es una foto a color de una flor, y su etiqueta diciendo cuál de las 5 especies es."*

**Por qué es más difícil que MNIST/CIFAR:** son **fotos reales** de tamaños distintos, con fondos, hojas, ángulos y luz variable. Y hay flores que se parecen mucho (rosa vs tulipán). Es un problema más cercano a la vida real.

### ¿Cómo lo procesamos? (los pasos antes de entrenar)
1. **Redimensionar** → todas las fotos a **128×128** (la red necesita que todas tengan el mismo tamaño).

2. **Normalizar** a **[0,1]** → dividir entre 255. Lo hacemos **dentro del modelo** con una capa `Rescaling`.

3. **¡NO aplanamos!** ← **la diferencia clave con el MLP**. Mantenemos la forma **128×128×3** para que la CNN pueda ver la estructura 2D.

4. **Data Augmentation** → durante el entrenamiento aplicamos giros y zoom aleatorios. Así la red ve "más variedad" y no memoriza. (Solo en entrenamiento, no en test.)

5. **Separar en 3 grupos** estratificados:
   - **Entrenamiento (70%)** → aprender.
   - **Validación (15%)** → vigilar que no haga trampa.
   - **Test (15%)** → escondido, para la nota final.

> **Resumen de una línea:** *"Redimensionamos a 128×128, normalizamos a [0,1], NO aplanamos, aumentamos los datos con giros, y partimos en aprender / vigilar / examinar."*

---

## 0. El "elevator pitch" (lo que dices al empezar)

> *"Construimos una **red neuronal convolucional (CNN)** con TensorFlow/Keras que aprende a **reconocer especies de flores en fotos reales** (margarita, rosa, girasol...). A diferencia de un MLP, esta **no aplana la imagen**: usa **filtros** que se deslizan detectando bordes y texturas. Y demostramos que **le gana a un MLP** usando menos parámetros."*

---

## 1. La gran idea: ¿qué hace diferente una CNN? (analogías)

### El problema del MLP con imágenes
Un MLP aplana la foto → pierde la noción de "qué píxel está al lado de cuál". Es como leer un libro al que le cortaste todas las palabras y las pusiste en una bolsa: pierdes el orden.

### La solución: la convolución
Imagina una **lupa pequeña (filtro 3×3)** que vas deslizando por toda la foto buscando un patrón concreto (un borde, un pétalo, un color).

- Cada **filtro** busca **una cosa** distinta y produce un **mapa** de "dónde aparece ese patrón".
- Usas la **misma lupa en toda la imagen** → por eso una CNN tiene **muchos menos parámetros** y detecta la flor **esté donde esté** en la foto.

### Las 3 piezas (dilas así)
1. **Convolución (`Conv2D`)** → *"las lupas que detectan patrones locales: bordes, texturas, colores."*
2. **Pooling (`MaxPooling`)** → *"encoge la imagen quedándose con lo importante; da resumen y tolerancia a que la flor se mueva un poco."*
3. **Jerarquía** → *"las primeras capas ven bordes; las profundas combinan esos bordes en pétalos y flores completas."* (Justo la diapositiva: *"detectar características locales y combinarlas en una representación global"*.)

> **Frase clave:** *"La CNN no mira píxeles sueltos: mira vecindarios de píxeles con filtros, y eso es justo lo que necesitan las imágenes."*

---

## 2. Cómo aprende (igual que la anterior)

Lo mismo que la FFNN: **Backpropagation + Gradiente Descendente (Adam)**. Analogía de **bajar la montaña con niebla**:

- La montaña = el error de la red. El valle = el mínimo error.
- El **gradiente** = la pendiente que sientes bajo los pies.
- El **learning rate** = el tamaño del paso. **Adam** lo ajusta solo.

> Lo único nuevo: ahora lo que se aprende son **los valores de los filtros** (qué patrón busca cada lupa), además de los pesos de las capas finales.

---

## 3. Recorrido del notebook (qué decir en cada celda)

### Puntos 1–6 — La teoría (pasa rápido, 1-2 min)
Menciona que ahí está documentado según la rúbrica: qué es una CNN, referencias (Fukushima 1980 el "abuelo", LeCun 1998 LeNet, Krizhevsky 2012 AlexNet que ganó ImageNet y disparó el deep learning), tipo de modelo, supuestos y las **reglas de cómo se prueba** (train/val/test).

### 7.1 — Imports y configuración
> *"Cargamos TensorFlow/Keras, fijamos la semilla, y definimos que todas las fotos se redimensionarán a 128×128."*

### 7.2 — Descargar y mirar el dataset
> *"Bajamos las fotos con una línea; cada carpeta es una especie. Las convertimos a arrays de NumPy."*
- **Gráfica 1 (las 5 flores):** *"Así se ve un ejemplo de cada especie. Son fotos reales: fondos, ángulos, luz distinta."*
- **Gráfica 2 (barras):** *"Cuántas fotos hay por especie; está más o menos balanceado."*

### 7.3 — Preprocesamiento (¡PUNTO CLAVE!)
> *"Redimensionamos a 128×128, NO aplanamos (la CNN necesita la imagen en 2D), normalizamos dentro del modelo, y separamos en train (70%) / validación (15%) / test (15%)."*

Recalca la diferencia con el MLP: *"el MLP aplanaba, la CNN NO."*

### 7.4 — Construir la CNN
> *"En el propio modelo metemos: data augmentation (giros/zoom), normalización, y bloques Conv2D + MaxPooling, y al final capas densas."*
- Señala en `model.summary()` cómo el tamaño espacial **se reduce** (128→64→32→16) mientras la **profundidad aumenta** (32→64→128 filtros).
- *"Recién al final hacemos Flatten, cuando ya extrajimos las características."*

### 7.5 — Compilar
> *"Adam como optimizador y entropía cruzada como error."*

### 7.6 — Entrenar (las *training iterations*)
> *"Entrena por épocas. En CPU tarda por las convoluciones. El Early Stopping para cuando deja de mejorar y restaura los mejores pesos."*

### 7.7 — Curvas de aprendizaje 📈
> *"Train y validación: si la validación se queda plana o empeora mientras el train sigue subiendo, hay overfitting. Por eso usamos dropout y data augmentation."*

### 7.8 — Evaluación en test
> *"Con fotos nuevas mide la accuracy real: **~72%**. Es un problema difícil (fotos reales, 5 clases parecidas); el azar sería 20%."*

### 7.9 — Matriz de confusión
> *"La diagonal son aciertos. Vean que confunde flores parecidas: rosa con tulipán, margarita con diente de león."*

### 7.10 — Ver predicciones
> *"Verde acierto, rojo error. Muchos errores son entre flores que hasta a nosotros nos costaría distinguir."*

### 7.11 — ¿Qué VE la CNN? (la joya 🔍)
> *"Esto es lo más bonito: visualizamos la salida de la primera capa de filtros. Cada recuadro es lo que detecta un filtro distinto: unos resaltan bordes, otros colores, otros texturas. Así es literalmente como la red 've' la flor por dentro."*

### 7.12 — CNN vs MLP (el cierre que vende 🎯)
> *"Para demostrar por qué usamos convolución: entrenamos un MLP normal sobre las mismas fotos. La CNN gana claramente (**~72% vs ~25%**) y encima con MENOS parámetros (**4.3M vs 12.6M**). Conclusión: para imágenes, la convolución es la herramienta correcta."*

### 8 — Conclusión
> *"CNN > MLP para imágenes. Si quisiéramos más precisión: transfer learning desde un modelo pre-entrenado (MobileNet), que pasaría del 90%."*

---

## 4. El guion de 10 minutos (resumido)

| Min | Qué muestras |
|---|---|
| 0–1 | Pitch + por qué el MLP falla con imágenes |
| 1–2.5 | La idea de la convolución (lupa/filtros) |
| 2.5–3.5 | El dataset de flores y las fotos (7.2) |
| 3.5–4.5 | Preprocesado: redimensionar, NO aplanar, train/val/test (7.3) ← *recalca* |
| 4.5–5.5 | La arquitectura Conv+Pool + data augmentation (7.4) |
| 5.5–7 | Entrenamiento + curvas (7.6, 7.7) |
| 7–8 | Resultado + matriz de confusión (7.8, 7.9) |
| 8–9 | Qué ve la CNN (7.11) + CNN vs MLP (7.12) ← *el broche* |
| 9–10 | Conclusión + preguntas |

---

## 5. Preguntas que te pueden hacer (y cómo responder)

**¿Qué es una convolución?**
> Un filtro pequeño que se desliza por la imagen detectando un patrón local (un borde, un color). Produce un mapa de dónde aparece ese patrón.

**¿Por qué no aplanamos como en el MLP?**
> Porque aplanar destruye la estructura 2D. La CNN necesita saber qué píxel está al lado de cuál para detectar formas.

**¿Por qué redimensionan todas a 128×128?**
> Las fotos reales tienen tamaños distintos y la red necesita una entrada de tamaño fijo.

**¿Qué es data augmentation y para qué sirve?**
> Aplicar giros/zoom aleatorios a las fotos durante el entrenamiento para que la red vea más variedad y no memorice (regularización).

**¿Qué hace el MaxPooling?**
> Encoge la imagen quedándose con el valor máximo de cada región → menos cómputo y tolerancia a que la flor se desplace.

**¿Por qué la CNN tiene menos parámetros que el MLP?**
> Porque comparte el mismo filtro por toda la imagen, en vez de un peso por cada píxel.

**¿Cómo se mejora la precisión?**
> Transfer learning (usar un modelo ya entrenado en millones de imágenes), redes más profundas, más data augmentation y entrenar en GPU.

**¿En qué se usa esto en la vida real?**
> Reconocimiento facial, autos autónomos, diagnóstico médico por imágenes, identificación de plantas/plagas en agricultura.

---

## 6. Errores que NO debes cometer al presentar

- ❌ No digas que la CNN "entiende" la foto → di que **detecta patrones locales y los combina**.
- ❌ No olvides recalcar que **NO se aplana** (es LA diferencia con el notebook de FFNN).
- ❌ No te saltes el preprocesado: el **redimensionado** y el **train/val/test** son lo que más valora la rúbrica.
- ✅ Muestra la sección 7.11 (feature maps) y la 7.12 (CNN vs MLP): son las que impresionan.
- ✅ Conecta con la diapositiva pág. 12: *"detectar características locales y combinarlas en una representación global"*.

---

**TL;DR para ti:** *CNN = red para imágenes que usa filtros (lupas) en vez de aplanar; detecta bordes→pétalos→flores, comparte pesos (menos parámetros). Bajamos fotos reales de flores, redimensionamos, NO aplanamos, aumentamos datos, separamos para no hacer trampa, entrenamos por épocas, vemos qué detectan los filtros, y demostramos que le gana al MLP. Fin.* 🚀
