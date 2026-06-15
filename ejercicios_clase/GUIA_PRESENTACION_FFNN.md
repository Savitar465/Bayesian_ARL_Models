# 🎤 Guía para presentar el notebook FFNN a la clase

> Cómo explicar [feed_forward_nn_mnist.ipynb](feed_forward_nn_mnist.ipynb) en ~10 min, en cristiano y sin morir en el intento. xd

---

## 📦 El dataset: de dónde sale, qué es y cómo lo procesamos

### ¿De dónde viene? (origen)
Usamos **MNIST** (*Modified National Institute of Standards and Technology*). Es **el dataset más famoso** para empezar en machine learning, el "Hola Mundo" del reconocimiento de imágenes.

- Lo armaron **Yann LeCun, Corinna Cortes y Christopher Burges** en los años 90.
- Las imágenes originales venían de dos fuentes del **NIST** (la agencia de estándares de EE.UU.): formularios escritos por **empleados del censo** y por **estudiantes de secundaria**.
- LeCun lo usó en su paper de 1998 (*Gradient-based learning…*, una de nuestras referencias) y desde entonces es el estándar para comparar modelos.
- **No lo descargamos a mano**: viene incluido en Keras. La línea `keras.datasets.mnist.load_data()` lo baja sola la primera vez.

> *"Es un set público de dígitos escritos a mano por personas reales, creado por LeCun en los 90; lo trae Keras integrado."*

### ¿De qué trata? (contenido)
Son **imágenes de números escritos a mano del 0 al 9**. El objetivo: que la máquina **lea el número** igual que lo haría una persona.

| Característica | Valor |
|---|---|
| Total de imágenes | **70 000** (60 000 entrenamiento + 10 000 test) |
| Tamaño de cada imagen | **28 × 28 píxeles** (784 en total) |
| Color | **Escala de grises** (1 canal) |
| Valor de cada píxel | **0 a 255** (0 = negro, 255 = blanco) |
| Etiqueta (la respuesta) | Un número del **0 al 9** |
| Clases | **10** (una por dígito), y está **balanceado** |

> *"Cada dato es una foto de 28×28 de un dígito, en grises, y su etiqueta diciendo qué número es."*

### ¿Cómo lo procesamos? (los 3 pasos antes de entrenar)
La red **no** puede comer las imágenes tal cual; las preparamos:

1. **Normalizar** → dividimos cada píxel entre 255 para que pase de `0–255` a **`0–1`**.
   - *Por qué:* las redes aprenden mejor y más rápido con números pequeños (y lo pide la diapositiva: *"Entrada: numérica de rango [0-1]"*).

2. **Aplanar (flatten)** → convertimos cada imagen de **28×28 (un cuadrado)** en **una fila de 784 números (un vector)**.
   - *Por qué:* un MLP recibe un vector de entrada, no una matriz 2D. (Es como desarmar la cuadrícula y ponerla en una sola línea.)

3. **Separar en 3 grupos** → del total partimos:
   - **Entrenamiento (54 000)** → para aprender.
   - **Validación (6 000)** → para vigilar que no haga trampa mientras aprende.
   - **Test (10 000)** → escondido, solo para la nota final con imágenes nuevas.
   - *Lo hacemos de forma **estratificada*** = manteniendo la misma proporción de cada dígito en cada grupo.

> **Resumen de una línea:** *"Bajamos píxeles a [0,1], estiramos la imagen a 784 números, y partimos los datos en aprender / vigilar / examinar."*

---

## 0. El "elevator pitch" (lo que dices al empezar)

> *"Construimos una **red neuronal feed-forward** con TensorFlow/Keras que aprende a **reconocer números escritos a mano** (del 0 al 9). Le mostramos 60 mil imágenes, aprende sola ajustando sus pesos, y al final acierta el **97.9 %** de imágenes que nunca había visto."*

Con eso ya saben de qué va todo. El resto es explicar **cómo** lo logra.

---

## 1. La analogía base (úsala para que todos entiendan)

Imagina que una **neurona** es una persona que toma una mini-decisión mirando unos números de entrada. Cada entrada le importa más o menos: eso es el **peso** ($w$). La neurona suma todo, y si pasa cierto umbral, "se activa" y pasa el mensaje a la siguiente.

- Muchas neuronas en fila = una **capa**.
- Varias capas una tras otra = una **red profunda** (Deep FFNN).
- **"Feed-forward"** = la info va **solo hacia adelante**: entrada → capas ocultas → salida. No hay vueltas atrás.

> **Frase clave:** *"La red no sabe nada al principio: los pesos son aleatorios. Aprender = ir corrigiendo esos pesos hasta que acierta."*

La fórmula de la diapositiva (pág. 30) en cristiano:

$$z_{k,i} = \underbrace{w_{0,i}^{(k)}}_{\text{sesgo}} + \sum_j \underbrace{g(z_{k-1,j})}_{\text{lo que dijo la capa anterior}} \cdot \underbrace{w_{j,i}^{(k)}}_{\text{cuánto me importa}}$$

"Cada neurona escucha a las de la capa anterior, le da importancia (peso) a cada una, suma, y aplica una función para decidir su salida."

---

## 2. Cómo aprende: Backpropagation + Gradiente Descendente

Esto es lo único "matemático" que conviene explicar. Analogía de **bajar una montaña con niebla**:

- Estás en una montaña (la **función de costo** = qué tan equivocada está la red) y quieres llegar al valle (el **mínimo error**).
- Hay niebla, no ves nada. Solo sientes la **pendiente** bajo tus pies (eso es el **gradiente**).
- Das un paso hacia abajo. El tamaño del paso es el **learning rate** ($\eta$).
- Repites miles de veces hasta llegar abajo.

> $W \leftarrow W - \eta \cdot \dfrac{\partial J(W)}{\partial W}$ = *"corrige cada peso un poquito en la dirección que reduce el error."*

- **Backpropagation** = el método para calcular esa pendiente capa por capa, de atrás hacia adelante.
- **Adam** = un caminante listo que ajusta solo el tamaño del paso (la "estrategia adaptativa" de la diapositiva pág. 33).
- **Época (epoch)** = una pasada completa por todas las imágenes. Entrenamos varias épocas.

---

## 3. Recorrido del notebook (qué decir en cada celda)

> El notebook tiene **2 partes**: documentación (puntos 1–6, lo teórico de la rúbrica) y **código** (punto 7, lo práctico). En la presentación enfócate en el código, que es lo vistoso.

### Puntos 1–6 — La teoría (pasa rápido, 1-2 min)
Solo menciona que ahí está documentado:
- **Qué es** la FFNN y para qué sirve.
- Las **referencias** (los papers: Rosenblatt inventó el perceptrón en 1958, Rumelhart popularizó backpropagation en 1986).
- **Tipo de modelo**: supervisado, paramétrico, offline, resultado = una red.
- El **algoritmo de entrenamiento** y los **supuestos** (los datos deben ir escalados a [0,1], etc.).

> *"Toda la teoría según la rúbrica está aquí arriba; vamos a la demostración en código."*

### 7.1 — Imports y configuración
> *"Cargamos TensorFlow/Keras y fijamos una semilla aleatoria para que los resultados sean **reproducibles** (que a todos nos dé lo mismo)."*

### 7.2 — Cargar MNIST y mirarlo
> *"MNIST son 70 mil imágenes de 28×28 píxeles de dígitos escritos a mano."*
- **Gráfica 1 (los 10 dígitos):** *"Así se ve un ejemplo de cada número."*
- **Gráfica 2 (barras):** *"Comprobamos que el dataset está balanceado: hay parecida cantidad de cada dígito, así que la red no se sesga hacia ninguno."*

### 7.3 — Preprocesamiento y separación (¡PUNTO CLAVE de la rúbrica!)
Aquí explica las **2 cosas importantes**:
1. **Normalizar**: los píxeles van de 0 a 255; los dividimos entre 255 para dejarlos en **[0,1]** (lo pide la diapositiva, las redes aprenden mejor con números pequeños).
2. **Aplanar**: cada imagen 28×28 se convierte en una **fila de 784 números** (el MLP come vectores, no cuadrados).

> **Lo más importante de decir:** *"Separamos los datos en **3 grupos**: entrenamiento (54 mil) para aprender, **validación** (6 mil) para vigilar que no haga trampa, y **test** (10 mil) que **escondemos** y solo usamos al final para la nota real."*

Analogía: **entrenamiento** = ejercicios de práctica; **validación** = simulacro de examen; **test** = el examen final con preguntas nuevas. Si solo midieras con los ejercicios de práctica, harías trampa (memorizar ≠ aprender).

### 7.4 — Construir el modelo
> *"Apilamos las capas: entrada de 784 → 128 neuronas → 64 neuronas → 10 salidas (una por dígito)."*
- **ReLU**: la función que da la no-linealidad (sin ella, la red sería una simple recta).
- **Softmax** (capa final): convierte la salida en **probabilidades** que suman 100% ("85% de que es un 7").
- **Dropout**: en cada paso apaga al azar el 30% de las neuronas → obliga a la red a no depender de una sola → **evita que memorice** (diapositiva pág. 37).
- `model.summary()`: *"Aquí se ve cuántos **parámetros** (pesos) tiene que aprender: ~109 mil."*

### 7.5 — Compilar
> *"Le decimos **con qué** aprender: optimizador **Adam**, y **qué minimizar**: la entropía cruzada (el error de clasificación)."*

### 7.6 — Entrenar (¡el corazón! las *training iterations*)
> *"Aquí entrena. Cada línea es una **época**. Fíjense cómo el **loss baja** y el **accuracy sube** época tras época: la red está aprendiendo en vivo."*
- **Early Stopping**: *"Si deja de mejorar, paramos solos para no sobreentrenar (diapositiva pág. 38). Por eso paró en ~15 épocas en vez de 30."*

### 7.7 — Curvas de aprendizaje (la gráfica estrella 📈)
> *"Estas dos curvas son la radiografía del entrenamiento."*
- Línea de **train** y línea de **validación**, juntas y bajando = **bien, generaliza**.
- Si la de validación **subiera** mientras la de train baja = **overfitting** (memorizó).

### 7.8 — Evaluación en test
> *"El momento de la verdad: con imágenes que NUNCA vio, acierta el **97.88 %**."*
- El **classification report** muestra precisión/recall por cada dígito.

### 7.9 — Matriz de confusión
> *"La diagonal son los aciertos. Fuera de la diagonal, los errores: vean que confunde dígitos parecidos, como el **4 con el 9** o el **3 con el 5**. Igual que nos pasaría a nosotros con letra fea."*

### 7.10 — Ver predicciones
> *"En verde los aciertos, en rojo los errores. Y aquí los casos que falló: muchos son números que ni un humano leería claro."*

### 7.11 — Demo de Overfitting vs Regularización (el broche de oro 🎯)
> *"Para demostrar la teoría: entrenamos OTRA red **sin** Dropout ni Early Stopping. Miren la diferencia: la red sin protección **memoriza** (el hueco entre train y validación es enorme: 0.147 vs 0.022). Eso es exactamente el **overfitting** de la diapositiva."*

### 8 — Conclusión
> *"Logramos 97.9% con un modelo sencillo. Si quisiéramos más, usaríamos una **CNN** (red convolucional), que entiende imágenes mejor que un MLP."*

---

## 4. El guion de 10 minutos (resumido)

| Min | Qué muestras |
|---|---|
| 0–1 | Pitch + analogía de la neurona |
| 1–2 | Cómo aprende (montaña/gradiente) |
| 2–3 | MNIST y las imágenes (7.2) |
| 3–4 | Los 3 grupos train/val/test (7.3) ← *recalca esto* |
| 4–5 | La arquitectura del modelo (7.4) |
| 5–7 | Entrenamiento en vivo + curvas (7.6, 7.7) ← *lo central* |
| 7–8 | Resultado: 97.9% + matriz de confusión (7.8, 7.9) |
| 8–9 | Demo overfitting (7.11) |
| 9–10 | Conclusión + preguntas |

---

## 5. Preguntas que te pueden hacer (y cómo responder)

**¿Por qué se separa en train/test?**
> Para medir si de verdad **aprendió** o solo **memorizó**. El test son preguntas nuevas.

**¿Qué es una época?**
> Una pasada completa por todas las imágenes de entrenamiento. Entrenamos varias.

**¿Qué hace el Dropout?**
> Apaga neuronas al azar durante el entrenamiento para que la red no dependa de unas pocas → evita el sobreajuste.

**¿Por qué Adam y no SGD normal?**
> Adam ajusta solo el tamaño del paso (learning rate adaptativo) → converge más rápido y estable.

**¿Por qué falla algunos?**
> Confunde dígitos visualmente parecidos (4/9, 3/5) o números escritos muy feo, igual que un humano.

**¿Por qué softmax al final?**
> Convierte los números crudos en probabilidades que suman 100%, una por cada dígito.

**¿Es deep learning?**
> Sí: al tener varias capas ocultas es una *Deep* Feed Forward NN.

---

## 6. Errores que NO debes cometer al presentar

- ❌ No digas "el modelo piensa/entiende" → di "**ajusta pesos para minimizar el error**".
- ❌ No te enredes con la fórmula de backprop → usa la analogía de la montaña.
- ❌ No leas el código línea por línea → explica **qué hace cada bloque** y muestra la **gráfica**.
- ✅ Apóyate en las gráficas: venden solas.
- ✅ Conecta cada parte con las **diapositivas** (lo valoran: épocas, overfitting, dropout, early stop, Adam).

---

**TL;DR para ti:** *Red que aprende a leer números corrigiendo pesos (backprop), separamos datos para no hacer trampa, entrenamos por épocas, miramos las curvas, sacamos 97.9%, y demostramos el overfitting. Fin.* 🚀
