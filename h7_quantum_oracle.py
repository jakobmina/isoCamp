"""
h7_quantum_oracle.py — Metriplex Oracle + H7 Conservation + Quantum Simulation

Implementa el algoritmo de Simon cuántico (MetriplexOracle) y la conservación
H7 en el espacio de Hilbert de 3 qubits. Integra Qiskit para validación cuántica.

Nota de consistencia (verificada empíricamente, no asumida): la cadena
oculta que descubre MetriplexOracle vía Simon es s = 7 (binario 111) —
no s = 3 como decía una versión anterior de este docstring — y coincide
exactamente con H7Conservation.CONSERVATION_CONSTANT = 7. Ambas clases
describen la misma estructura de emparejamiento x ↔ (7 ⊕ x) desde dos
ángulos distintos: el oráculo la *descubre* (algoritmo de Simon sobre
una función 2-a-1), la conservación la *impone* (regla de paridad en
el espacio de Hilbert de 3 qubits). Verificable reproduciendo
MetriplexOracle().get_oracle_info()['symmetry_string'] == 7.

Autoria: Jacobo Tlacaelel Mina Rodriguez
Marco: Mandato Metripléxico — Componente Simpléctica (H) + Métrica (S)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from typing import Tuple, Dict, List, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
import warnings

from h7_shared import PI as pi, PHI  # fuente única, compartida con
                                      # h7_benchmark.py / h7_streamlit_app.py.
                                      # PHI aquí es (1+√5)/2 con precisión
                                      # completa float64 — antes este
                                      # archivo usaba el literal truncado
                                      # 1.6180339887 (Δ≈4.99e-11 vs. el
                                      # valor correcto).


class EnergyProfile(Enum):
    """Perfiles de normalización de energía."""
    LINEAR    = "linear"
    QUADRATIC = "quadratic"
    METRIPLEX = "metriplex"   # Ajustado para ciclo topológico H7 (s=7)
    CUSTOM    = "custom"


@dataclass
class MetriplexConfig:
    """Configuración del oráculo metriplex."""
    momentum_range: Tuple[int, int] = (0, 7)
    energy_profile: EnergyProfile = EnergyProfile.METRIPLEX
    normalization_target: float = 0.45
    collision_groups: Optional[Dict[str, List[int]]] = None

    def __post_init__(self):
        if self.collision_groups is None:
            self.collision_groups = {
                '1-6': [1, 6],
                '2-5': [2, 5],
                '3-4': [3, 4],
                'TRUNCATED': [0, 7]
            }


class MetriplexOracle:
    """
    Oráculo de momentum metriplex para segunda cuantización.

    Función 2-a-1 oculta: pares de momentum p₁, p₂ con p₁ ⊕ p₂ = 7
    colisionan (grupos '1-6', '2-5', '3-4', 'TRUNCATED'=[0,7] — los
    cuatro pares complementarios del cubo Z₂³). El algoritmo de Simon
    descubre la estructura: s = 7 (binario 111), verificable con
    _compute_symmetry_string() / get_oracle_info()['symmetry_string'].

    d_symp: Fase cuántica e^{i·E(p)·2π}  → movimiento conservativo
    d_metr: Corrección de colisiones      → relajación disipativa
    """

    def __init__(self, config: MetriplexConfig = None):
        if config is None:
            config = MetriplexConfig()
        self.config = config
        self.p_min, self.p_max = config.momentum_range
        self._build_energy_map()
        self._build_collision_map()

    def compute_lagrangian(self, p: int) -> Tuple[float, float]:
        """
        Retorna (L_symp, L_metr) para el estado de momentum p.

        L_symp = E(p) · 2π  (fase conservativa)
        L_metr = -E(p)²      (disipación de colisión)
        """
        energy = self.energy_map[p]
        L_symp = energy * 2 * np.pi
        L_metr = -(energy ** 2)
        return L_symp, L_metr

    def _build_energy_map(self):
        self.energy_map = {}
        for p in range(self.p_min, self.p_max + 1):
            if self.config.energy_profile == EnergyProfile.LINEAR:
                raw_energy = p / self.p_max
            elif self.config.energy_profile == EnergyProfile.QUADRATIC:
                raw_energy = (p / self.p_max) ** 2
            elif self.config.energy_profile == EnergyProfile.METRIPLEX:
                raw_energy = self.config.normalization_target * np.sin(np.pi * p / 7.0)
            else:
                raise ValueError(f"Unknown energy profile: {self.config.energy_profile}")
            self.energy_map[p] = raw_energy

        mean_energy = np.mean(list(self.energy_map.values()))
        if abs(mean_energy - self.config.normalization_target) > 0.01:
            warnings.warn(f"Mean energy {mean_energy:.4f} deviates from target {self.config.normalization_target:.4f}")

    def _build_collision_map(self):
        self.collision_map = {}
        self.output_groups = {}
        for group_name, momenta in self.config.collision_groups.items():
            for p in momenta:
                self.collision_map[p] = group_name
            group_index = list(self.config.collision_groups.keys()).index(group_name)
            n_groups = len(self.config.collision_groups)
            output_vec = np.zeros(n_groups)
            output_vec[group_index] = 1.0
            self.output_groups[group_name] = output_vec

    def _compute_symmetry_string(self) -> int:
        xor_accumulator = 0
        for group in self.config.collision_groups.values():
            for i, p1 in enumerate(group):
                for p2 in group[i+1:]:
                    xor_accumulator |= (p1 ^ p2)
        return xor_accumulator

    def forward(self, momentum: int) -> Tuple[str, np.ndarray, float]:
        if momentum < self.p_min or momentum > self.p_max:
            raise ValueError(f"Momentum {momentum} out of range [{self.p_min}, {self.p_max}]")
        group      = self.collision_map[momentum]
        output_vec = self.output_groups[group]
        energy     = self.energy_map[momentum]
        return group, output_vec, energy

    def collide_pair(self, p1: int, p2: int) -> bool:
        return self.collision_map[p1] == self.collision_map[p2]

    def get_collision_partners(self, p: int) -> List[int]:
        group = self.collision_map[p]
        return self.config.collision_groups[group]

    def symmetry_string(self) -> int:
        return self._compute_symmetry_string()

    def get_oracle_info(self) -> Dict:
        return {
            'momentum_range':     self.config.momentum_range,
            'n_groups':           len(self.config.collision_groups),
            'collision_groups':   dict(self.config.collision_groups),
            'symmetry_string':    self.symmetry_string(),
            'energy_profile':     self.config.energy_profile.value,
            'normalization_target': self.config.normalization_target,
            'energy_map':         dict(self.energy_map),
            'collision_structure': {p: self.collision_map[p] for p in range(self.p_min, self.p_max + 1)},
        }


class H7Conservation:
    """
    Mecanismo I: Conservación de Entrelazamiento H7.

    En el espacio de Hilbert de 3 qubits (8 estados: 0-7),
    los estados se emparejan por la regla: x ↔ (7 ⊕ x).

    Conservativo: |ψ⟩ + |7⊕ψ⟩  → reversible (d_symp)
    Disipativo: colapso a uno de los pares → irreversible (d_metr)
    """
    CONSERVATION_CONSTANT = 7

    @staticmethod
    def partner_state(state: int) -> int:
        if not (0 <= state <= 7):
            raise ValueError("State must be in [0, 7]")
        return H7Conservation.CONSERVATION_CONSTANT ^ state

    @staticmethod
    def verify_pairing(state_a: int, state_b: int) -> bool:
        return state_b == H7Conservation.partner_state(state_a)

    @staticmethod
    def pairing_table() -> Dict[int, int]:
        return {i: H7Conservation.partner_state(i) for i in range(8)}

    @staticmethod
    def verify_conservation_invariant(state_vector: np.ndarray, threshold: float = 1e-6) -> bool:
        if len(state_vector) != 8:
            raise ValueError("State vector must be 8-dimensional (3-qubit Hilbert space)")
        for i in range(8):
            if abs(state_vector[i]) > threshold:
                partner = H7Conservation.partner_state(i)
                if abs(state_vector[partner]) < threshold:
                    return False
        return True


def run_quantum_simulation():
    """
    Simulación cuántica: estabilidad de circuito sobre fases φ.
    d_symp: rotación RZ/RX (unitaria)  → conservativa
    d_metr: decoherencia de medición   → disipativa
    """
    print("\n===== Quantum Stability Analysis =====")

    # Métricas de referencia
    stability_data  = [0.9935, 0.9905, 0.9875, 0.985, 0.993, 0.989, 0.9827, 0.9997]
    stability_model = [0.990,  0.990,  0.988,  0.990, 0.9925, 0.993, 0.995, 0.9997]
    r2   = r2_score(stability_data, stability_model)
    rmse = np.sqrt(mean_squared_error(stability_data, stability_model))
    print(f"R²   = {r2:.4f}")
    print(f"RMSE = {rmse:.5f}")

    # Circuito base — 1 qubit, fase φ
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.rz(PHI, 0)
    state = Statevector(qc)

    print(f"\nStatevector: {state}")
    print(f"Probabilities: {state.probabilities_dict()}")

    sim = AerSimulator()
    qc.measure(0, 0)
    counts = sim.run(qc, shots=1024).result().get_counts()
    print(f"Counts (1024 shots): {counts}")

    # Barrido de fases
    phases = [pi/4, pi/3, pi/2, PHI, 2*pi/3, 3*pi/4]
    fidelities = []
    for phase in phases:
        qc_p = QuantumCircuit(1, 1)
        qc_p.h(0)
        qc_p.rz(phase, 0)
        qc_p.rx(pi, 0)
        qc_p.measure(0, 0)
        res = sim.run(qc_p, shots=10000).result().get_counts()
        p0 = res.get('0', 0) / 10000
        stability = 1 - abs(p0 - 0.5) * 2
        fidelities.append(stability)
        print(f"  Phase {phase:.4f} → Stability = {stability:.4f}")

    return phases, fidelities, stability_data


def plot_sigmoid_euler(phases, fidelities, stability_data):
    """
    Descomposición Sigmoid-Euler: visualiza la competencia d_symp vs d_metr.
    """
    GOLDEN_ANGLE = math.cos(np.pi * PHI * 1)
    theta = np.linspace(-2, 2, 1000)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    plt.style.use('dark_background')

    # Panel 1: Transiciones Sigmoid (componente Euler — d_metr)
    sigmoid1 = 1 / (1 + np.exp(-5 * (theta - PHI)))
    sigmoid2 = 1 / (1 + np.exp(-5 * (theta - GOLDEN_ANGLE)))
    axes[0].plot(theta, sigmoid1, label=f'Sigmoid @ φ={PHI:.3f}', linewidth=2, color='#00f2ff')
    axes[0].plot(theta, sigmoid2, label=f'Sigmoid @ golden_angle={GOLDEN_ANGLE:.3f}', linewidth=2, color='#f39c12')
    axes[0].axvline(PHI, color='gold', linestyle='--', alpha=0.5)
    axes[0].axvline(GOLDEN_ANGLE, color='orange', linestyle='--', alpha=0.5)
    axes[0].set_title('Component 1: Sigmoid Transitions — d_metr (Dissipative)', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Panel 2: Interferencia oscilatoria (d_symp)
    oscillation = np.cos(3 * theta) * 0.5 + 0.5
    axes[1].plot(theta, oscillation, color='#00ff41', linewidth=2, label='cos(ωθ) — Quantum interference')
    axes[1].axvline(PHI, color='gold', linestyle='--', alpha=0.5)
    axes[1].set_title('Component 2: Interference Pattern — d_symp (Conservative)', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Panel 3: Efecto combinado
    combined = sigmoid1 * oscillation + sigmoid2 * (1 - oscillation)
    rango = max(stability_data) - min(stability_data)
    combined = (combined - combined.min()) / (combined.max() - combined.min()) * rango + min(stability_data)
    phases_data = np.array([-2.0, -1.5, -1.2, 1.0, 1.3, 1.618, 1.8, 2.0])
    axes[2].plot(theta, combined, linewidth=3, color='red', alpha=0.7, label='Sigmoid-Euler Model')
    axes[2].plot(phases_data, stability_data, 'o', markersize=12, color='steelblue',
                 label='Measured', zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    axes[2].axvline(PHI, color='gold', linestyle='--', linewidth=2, label='φ (golden)')
    axes[2].axvline(GOLDEN_ANGLE, color='orange', linestyle=':', linewidth=2, label='Golden angle')
    axes[2].set_xlabel('Phase θ (radians)', fontsize=13)
    axes[2].set_title('Combined: Sigmoid-Euler Interference — d_symp ↔ d_metr', fontweight='bold')
    axes[2].legend(fontsize=10, loc='lower left'); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sigmoid_euler_decomposition.png', dpi=300, bbox_inches='tight')
    print("\n📊 Plot saved: sigmoid_euler_decomposition.png")
    plt.show()


if __name__ == "__main__":
    # 1. Oracle Info
    config = MetriplexConfig()
    oracle = MetriplexOracle(config)
    info   = oracle.get_oracle_info()

    print("=" * 50)
    print("  METRIPLEX ORACLE — H7 Quantum Module")
    print("=" * 50)
    print(f"Momentum range : {info['momentum_range']}")
    print(f"Collision groups: {info['n_groups']}")
    print(f"Hidden symmetry : s = {info['symmetry_string']}")

    print("\nEnergy Map (normalized):")
    for p in range(1, 7):
        group, output, energy = oracle.forward(p)
        L_symp, L_metr = oracle.compute_lagrangian(p)
        print(f"  p={p}: group={group}, E={energy:.4f}, L_symp={L_symp:.4f}, L_metr={L_metr:.4f}")

    print("\nH7 Conservation Pairing:")
    for state, partner in H7Conservation.pairing_table().items():
        print(f"  |{state}⟩ ↔ |{partner}⟩")

    # 2. Quantum Simulation
    phases, fidelities, stability_data = run_quantum_simulation()

    # 3. Sigmoid-Euler Plot
    plot_sigmoid_euler(phases, fidelities, stability_data)
