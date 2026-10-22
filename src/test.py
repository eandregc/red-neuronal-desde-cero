"""
Carga los pesos desde el Excel y evalúa la red con la tabla de verdad
y con puntos ruidosos alrededor de cada esquina.
"""

import argparse
import sys

import numpy as np

from data import get_test_data, get_training_data
from excel import ARCHIVO_PESOS, leer_pesos
from model import RedNeuronal


def main():
    parser = argparse.ArgumentParser(description="Evalúa la red con los pesos del Excel")
    parser.add_argument("--pesos", default=str(ARCHIVO_PESOS), help="archivo de Excel con los pesos")
    parser.add_argument("--ruido", type=float, default=0.15, help="desviación estándar del ruido")
    parser.add_argument("--n", type=int, default=40, help="número de puntos ruidosos")
    args = parser.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    config, pesos = leer_pesos(args.pesos)
    compuerta = config["Compuerta"].lower()
    red = RedNeuronal(*pesos["W1"].shape)
    red.set_pesos(pesos)
    print(f"Pesos cargados de '{args.pesos}'")
    print(f"Red {config['Arquitectura']} entrenada para la compuerta {compuerta.upper()}")

    print(f"\n{'=' * 40}\nTabla de verdad\n")
    X, y = get_training_data(compuerta)
    salidas = red.forward(X)
    print(" x1  x2 | y |    ŷ     | clase")
    for x, esperado, salida in zip(X, y, salidas):
        print(f"  {x[0]:.0f}   {x[1]:.0f} | {esperado[0]:.0f} | {salida[0]:.6f} |   {int(salida[0] >= 0.5)}")
    aciertos = int(np.sum(red.predict(X) == y))
    print(f"\nCasos correctos: {aciertos}/{len(X)}")

    print(f"\n{'=' * 40}\nPuntos con ruido gaussiano (σ = {args.ruido})\n")
    X_ruido, y_ruido = get_test_data(compuerta, n=args.n, ruido=args.ruido)
    predichas = red.predict(X_ruido)
    for x, esperado, clase in zip(X_ruido, y_ruido, predichas):
        marca = "" if clase[0] == esperado[0] else "   <-- error"
        print(f"  ({x[0]:6.3f}, {x[1]:6.3f})  y = {esperado[0]:.0f}  clase = {clase[0]}{marca}")
    aciertos = int(np.sum(predichas == y_ruido))
    print(f"\nExactitud con ruido: {aciertos}/{len(X_ruido)} ({aciertos / len(X_ruido):.0%})")


if __name__ == "__main__":
    main()
