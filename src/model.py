"""
Perceptrón multicapa con una capa oculta, programado solo con numpy:

    z1 = X·W1 + b1      a1 = σ(z1)        (capa oculta)
    z2 = a1·W2 + b2     ŷ  = σ(z2)        (capa de salida)

Se entrena con retropropagación y descenso de gradiente por lotes,
minimizando el error cuadrático medio E = (1/n) Σ (ŷ − y)².
"""

import numpy as np


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def derivada_sigmoide(a):
    """Derivada de la sigmoide escrita con su propia salida: σ'(z) = a·(1 − a)."""
    return a * (1 - a)


class RedNeuronal:
    def __init__(self, n_entradas, n_ocultas, n_salidas=1, seed=None):
        rng = np.random.default_rng(seed)
        # Pesos y sesgos iniciales aleatorios en [-1, 1]
        self.W1 = rng.uniform(-1, 1, (n_entradas, n_ocultas))
        self.b1 = rng.uniform(-1, 1, (1, n_ocultas))
        self.W2 = rng.uniform(-1, 1, (n_ocultas, n_salidas))
        self.b2 = rng.uniform(-1, 1, (1, n_salidas))
        self.historial = []

    def forward(self, X):
        """Propagación hacia adelante. Guarda las activaciones para la retropropagación."""
        self.a1 = sigmoide(X @ self.W1 + self.b1)
        self.a2 = sigmoide(self.a1 @ self.W2 + self.b2)
        return self.a2

    def error(self, y, y_pred):
        """Error cuadrático medio: E = (1/n) Σ (ŷ − y)²."""
        return float(np.mean((y_pred - y) ** 2))

    def backward(self, X, y, tasa):
        """
        Regla de la cadena desde la salida hacia la entrada y un paso de descenso de gradiente.
            delta2 = ∂E/∂z2 = 2(ŷ − y)/n · ŷ(1 − ŷ)
            delta1 = ∂E/∂z1 = (delta2 · W2ᵀ) · a1(1 − a1)
        """
        n = len(X)
        delta2 = 2 * (self.a2 - y) / n * derivada_sigmoide(self.a2)
        delta1 = (delta2 @ self.W2.T) * derivada_sigmoide(self.a1)  # usa W2 antes de actualizarla

        self.W2 -= tasa * self.a1.T @ delta2
        self.b2 -= tasa * delta2.sum(axis=0, keepdims=True)
        self.W1 -= tasa * X.T @ delta1
        self.b1 -= tasa * delta1.sum(axis=0, keepdims=True)

    def train(self, X, y, epocas=10000, tasa=1.0):
        """historial[k] es el error después de k épocas (historial[0] = error inicial)."""
        for _ in range(epocas):
            self.historial.append(self.error(y, self.forward(X)))
            self.backward(X, y, tasa)
        self.historial.append(self.error(y, self.forward(X)))

    def predict(self, X):
        """Clase predicha: 1 si ŷ ≥ 0.5, 0 en otro caso."""
        return (self.forward(X) >= 0.5).astype(int)

    def get_pesos(self):
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    def set_pesos(self, pesos):
        self.W1, self.b1 = pesos["W1"].copy(), pesos["b1"].copy()
        self.W2, self.b2 = pesos["W2"].copy(), pesos["b2"].copy()
