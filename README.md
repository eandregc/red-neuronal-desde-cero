# Red Neuronal desde Cero
Una red neuronal artificial es un modelo formado por capas de neuronas conectadas entre sí, donde cada conexión tiene un peso que se ajusta durante el entrenamiento. En este proyecto se programa **desde cero**, usando únicamente operaciones de matrices de `numpy` (sin TensorFlow, PyTorch ni scikit-learn), un **perceptrón multicapa** que aprende la compuerta lógica **XOR**. Los pesos generados se exportan a un Excel que vuelve a calcular la red con fórmulas, para comprobar que los números son correctos.

XOR es el problema clásico para una red con capa oculta: no es linealmente separable (ninguna recta separa los casos con salida 1 de los casos con salida 0), por lo que un perceptrón de una sola capa no puede resolverlo.

| x1 | x2 | XOR |
|:--:|:--:|:---:|
| 0  | 0  |  0  |
| 0  | 1  |  1  |
| 1  | 0  |  1  |
| 1  | 1  |  0  |

**Características clave:**

* **Arquitectura:** 2 entradas → 4 neuronas ocultas → 1 salida (red 2-4-1). Las capas están totalmente conectadas y cada neurona tiene su propio sesgo.
* **Función de activación:** Sigmoide `σ(z) = 1 / (1 + e^(−z))` en la capa oculta y en la salida. Su derivada se escribe con la propia salida: `σ'(z) = a·(1 − a)`.
* **Función de error:** Error cuadrático medio `E = (1/n) Σ (ŷ − y)²`.
* **Retropropagación:** El gradiente del error se calcula con la regla de la cadena, desde la salida hacia la entrada, para cada peso y sesgo.
* **Descenso de gradiente:** Cada parámetro se actualiza con `w ← w − η · ∂E/∂w`, por lotes (las 4 combinaciones en cada época), con tasa de aprendizaje `η = 1.0` durante 10 000 épocas.
* **Inicialización:** Pesos y sesgos aleatorios uniformes en `[−1, 1]` con semilla fija (`42`), para que el entrenamiento sea reproducible.
* **Clasificación:** La salida `ŷ` es un valor entre 0 y 1; la clase predicha es 1 si `ŷ ≥ 0.5` y 0 en otro caso.
* **Comprobación en Excel:** El archivo `pesos.xlsx` contiene los pesos y además rehace con fórmulas la propagación hacia adelante y la primera época de retropropagación.

## Ecuaciones

```
Propagación hacia adelante
    z1 = X·W1 + b1          a1 = σ(z1)          (capa oculta)
    z2 = a1·W2 + b2         ŷ  = σ(z2)          (capa de salida)

Error
    E = (1/n) Σ (ŷ − y)²

Retropropagación (regla de la cadena)
    δ2 = ∂E/∂z2 = 2(ŷ − y)/n · ŷ(1 − ŷ)
    δ1 = ∂E/∂z1 = (δ2 · W2ᵀ) · a1(1 − a1)

    ∂E/∂W2 = a1ᵀ · δ2       ∂E/∂b2 = Σ δ2
    ∂E/∂W1 = Xᵀ  · δ1       ∂E/∂b1 = Σ δ1

Descenso de gradiente
    W ← W − η · ∂E/∂W       b ← b − η · ∂E/∂b
```

## Resultados (XOR, semilla 42)

El error baja de `0.250725` con los pesos iniciales a `0.000200` después de 10 000 épocas, y la red clasifica correctamente las cuatro combinaciones:

| x1 | x2 | y | ŷ        | Clase |
|:--:|:--:|:-:|:--------:|:-----:|
| 0  | 0  | 0 | 0.015350 |   0   |
| 0  | 1  | 1 | 0.987000 |   1   |
| 1  | 0  | 1 | 0.984623 |   1   |
| 1  | 1  | 0 | 0.012553 |   0   |

Con 40 puntos de prueba a los que se les agrega ruido gaussiano (σ = 0.15) alrededor de cada esquina, la red acierta 40/40.

```
Pesos finales
W1 = [[ 6.339719  -4.214788   2.593728   3.432181]
      [-6.325662   4.332709  -1.895390  -3.363834]]
b1 = [[-3.342436  -2.578712   1.173648   2.077485]]
W2 = [[10.565537]  [6.464626]  [-2.275276]  [-4.168569]]
b2 = [[ 0.464649]]
```

> Con solo 2 neuronas ocultas (la red mínima para XOR) el entrenamiento a veces se queda atorado en un mínimo local: en una prueba con 20 semillas convergió en 16. Con 3 o 4 neuronas ocultas convergió en las 20.

## Excel de comprobación (`pesos.xlsx`)

| Hoja | Contenido |
|---|---|
| **Pesos** | Configuración del entrenamiento y pesos finales `W1`, `b1`, `W2`, `b2`. |
| **Comprobación** | Propagación hacia adelante de las 4 entradas calculada con fórmulas de Excel a partir de la hoja Pesos. Compara `ŷ (Excel)` con `ŷ (Python)`: la diferencia máxima es `2.2E-16` (error de redondeo) y la exactitud es 100 %. |
| **Retropropagación** | Parte de los pesos iniciales y calcula con fórmulas el error, las deltas `δ`, los gradientes `∂E/∂w` y el paso de descenso de gradiente. El resultado coincide con los pesos de Python después de una época (diferencia máxima `6.9E-17`). |
| **Pesos iniciales** | Pesos aleatorios con los que comenzó el entrenamiento. |
| **Entrenamiento** | Error cuadrático medio cada 100 épocas y su gráfica. |

En todas las hojas los valores en **azul** vienen de Python, los de color **negro** son fórmulas y los de color **verde** son referencias a otra hoja. Si se modifica un peso en la hoja Pesos, la hoja Comprobación se recalcula sola. `test.py` lee los pesos directamente de este Excel, así que el archivo es la fuente de los pesos que usa la red.

## Distribución del Proyecto
red_neuronal_desde_cero/
├── src/
│   ├── data.py       # Tablas de verdad de las compuertas lógicas (XOR por defecto) y puntos de prueba con ruido.
│   ├── model.py      # Clase RedNeuronal: sigmoide, propagación hacia adelante, retropropagación y descenso de gradiente.
│   ├── excel.py      # Exporta los pesos y las hojas de comprobación a 'pesos.xlsx' y los vuelve a leer.
│   ├── train.py      # Script que entrena la red y genera 'pesos.xlsx'.
│   └── test.py       # Script que carga los pesos desde el Excel y evalúa la red.
├── pesos.xlsx        # Excel con los pesos generados y las hojas de comprobación.
├── .gitignore        # Excluye la caché de Python y los archivos temporales de Excel.
└── requirements.txt  # Dependencias del entorno virtual (numpy, openpyxl).

## Ejecución

```bash
pip install -r requirements.txt
cd src
python train.py   # Entrena la red XOR 2-4-1 y genera pesos.xlsx
python test.py    # Carga los pesos desde pesos.xlsx y evalúa la red
```

Opciones adicionales:

```bash
python train.py --ocultas 2 --semilla 3                  # cambiar neuronas ocultas o semilla
python train.py --epocas 20000 --tasa 0.5                # cambiar épocas o tasa de aprendizaje
python train.py --compuerta and --salida and.xlsx        # otras compuertas: xor, xnor, and, or, nand
python test.py --pesos and.xlsx --ruido 0.2              # evaluar otro archivo de pesos
```

> `numpy` se usa solo para la aritmética de matrices y `openpyxl` solo para escribir y leer el Excel. Toda la lógica de la red (activación, propagación, gradientes y actualización de pesos) está escrita a mano en `model.py`.
