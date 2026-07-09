"""
h7_shared.py — Constantes y utilidades compartidas del framework H7.

Fuente única de verdad para:
  - π, φ (razón áurea, precisión completa float64)
  - O_n = cos(π·n) = (-1)^n, el operador de paridad H7

Se centraliza aquí para que h7_benchmark.py, h7_quantum_oracle.py y
h7_streamlit_app.py (y cualquier módulo H7 futuro) usen exactamente
los mismos valores. Antes de esta consolidación existían dos
definiciones de φ divergentes en el framework:

    (1+√5)/2                → 1.618033988749895   (precisión completa)
    1.6180339887 (literal)  → 1.6180339887         (truncado a 10 dígitos)

Diferencia: ≈4.99e-11. Insignificante para n pequeño, pero es una
fuente de verdad duplicada e innecesaria — se elimina usando esta
única definición en todo el framework.

De igual forma, O_n = cos(πn) es una identidad de paridad exacta
para n entero: NO se calcula vía cos() (que arrastra error de punto
flotante a n grande), sino como (-1)^n directo.

Autoría: Jacobo Tlacaelel Mina Rodriguez
Marco: Mandato Metripléxico
"""

import math

# ─────────────────────────────────────────────
#  CONSTANTES — fuente única
# ─────────────────────────────────────────────
PI = math.pi
PHI = (1 + math.sqrt(5)) / 2  # razón áurea, precisión completa float64


# ─────────────────────────────────────────────
#  OPERADOR DE PARIDAD H7
# ─────────────────────────────────────────────
def h7_parity(n: int) -> float:
    """
    O_n = cos(π·n), para n entero, es exactamente (-1)^n.

    Se implementa como chequeo de paridad entero en vez de invocar
    cos(π·n) para eliminar el arrastre de error de punto flotante
    que aparece a n grande (cos(π·n) puede desviarse de ±1 exacto
    por acumulación de error en la evaluación de π·n y en la propia
    función coseno).

    Devuelve +1.0 si n es par, -1.0 si n es impar.
    """
    return 1.0 if n % 2 == 0 else -1.0


def h7_quasiperiod(n: int, phi: float = PHI, pi: float = PI) -> float:
    """
    I_n = cos(π·φ·n), el operador cuasiperiódico H7.

    A diferencia de O_n, esta cantidad NO tiene una forma cerrada
    exacta (φ es irracional), así que sí se evalúa vía cos(). Se
    expone aquí solo para que todos los módulos usen el mismo φ.
    """
    return math.cos(pi * phi * n)
