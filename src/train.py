"""
Entrena la red neuronal y exporta los pesos a Excel.

Uso:
    python train.py                               # XOR, 4 neuronas ocultas, 10000 épocas
    python train.py --compuerta and --ocultas 2
    python train.py --epocas 20000 --tasa 0.5 --semilla 7
"""

import argparse
import sys

from data import COMPUERTAS, get_training_data
from excel import ARCHIVO_PESOS, exportar
from model import RedNeuronal


def main():
    parser = argparse.ArgumentParser(description="Red neuronal desde cero (perceptrón multicapa)")
    parser.add_argument("--compuerta", choices=list(COMPUERTAS), default="xor")
    parser.add_argument("--ocultas", type=int, default=4, help="neuronas en la capa oculta")
    parser.add_argument("--epocas", type=int, default=10000)
    parser.add_argument("--tasa", type=float, default=1.0, help="tasa de aprendizaje")
    parser.add_argument("--semilla", type=int, default=42, help="semilla de los pesos iniciales")
    parser.add_argument("--salida", default=str(ARCHIVO_PESOS), help="archivo de Excel de salida")
    args = parser.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    X, y = get_training_data(args.compuerta)
    arquitectura = f"{X.shape[1]}-{args.ocultas}-1"
    red = RedNeuronal(X.shape[1], args.ocultas, 1, seed=args.semilla)
    iniciales = red.get_pesos()

    # Con la misma semilla se repiten los pesos iniciales; se entrena una sola época
    # para que el Excel compruebe paso a paso la retropropagación.
    red_1 = RedNeuronal(X.shape[1], args.ocultas, 1, seed=args.semilla)
    red_1.train(X, y, epocas=1, tasa=args.tasa)

    print(f"Entrenando red {arquitectura} para la compuerta {args.compuerta.upper()} "
          f"({args.epocas} épocas, η = {args.tasa}, semilla {args.semilla})...")
    red.train(X, y, epocas=args.epocas, tasa=args.tasa)
    for e in range(0, args.epocas + 1, max(1, args.epocas // 10)):
        print(f"  época {e:>6}   ECM = {red.historial[e]:.6f}")

    y_pred = red.forward(X)
    print("\n x1  x2 | y |    ŷ     | clase")
    for x, esperado, salida in zip(X, y, y_pred):
        print(f"  {x[0]:.0f}   {x[1]:.0f} | {esperado[0]:.0f} | {salida[0]:.6f} |   {int(salida[0] >= 0.5)}")

    print("\nPesos finales:")
    for nombre, matriz in red.get_pesos().items():
        print(f"{nombre} =\n{matriz.round(6)}")

    config = {
        "compuerta": args.compuerta,
        "arquitectura": arquitectura,
        "ocultas": args.ocultas,
        "epocas": args.epocas,
        "tasa": args.tasa,
        "semilla": args.semilla,
    }
    exportar(args.salida, config, X, y, iniciales, red_1.get_pesos(), red.get_pesos(),
             red.historial, y_pred)
    print(f"\nEntrenamiento finalizado. Pesos exportados a '{args.salida}'.")


if __name__ == "__main__":
    main()
