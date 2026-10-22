import numpy as np

# Tablas de verdad de compuertas lógicas de dos entradas: cada fila (x1, x2) -> y.
# XOR es el caso clásico: no es linealmente separable, así que un perceptrón de
# una sola capa no puede resolverlo y se necesita al menos una capa oculta.

ENTRADAS = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

COMPUERTAS = {
    "xor":  [0, 1, 1, 0],
    "xnor": [1, 0, 0, 1],
    "and":  [0, 0, 0, 1],
    "or":   [0, 1, 1, 1],
    "nand": [1, 1, 1, 0],
}

def get_training_data(compuerta="xor"):
    """Regresa X (4 x 2) y y (4 x 1) con la tabla de verdad de la compuerta."""
    X = ENTRADAS.copy()
    y = np.array(COMPUERTAS[compuerta], dtype=float).reshape(-1, 1)
    return X, y

def get_test_data(compuerta="xor", n=40, ruido=0.15, seed=42):
    """Puntos alrededor de las cuatro esquinas con ruido gaussiano.
    Cada punto conserva la etiqueta de la esquina de la que proviene."""
    rng = np.random.default_rng(seed)
    X, y = get_training_data(compuerta)
    indices = rng.integers(0, len(X), n)
    X_ruido = X[indices] + rng.normal(0, ruido, (n, X.shape[1]))
    return X_ruido, y[indices]
