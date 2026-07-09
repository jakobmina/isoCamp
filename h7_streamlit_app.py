"""
H7 · DIT · Metriplectic QNN — Z₇ Particle Classifier
Streamlit UI (adaptado del CLI h7_main.py)

Requiere en el mismo entorno:
    pip install streamlit qiskit qiskit-aer plotly --break-system-packages
    y el módulo local `qnn.py` con `string_to_qnn_seed`.

Ejecutar con:
    streamlit run h7_streamlit_app.py
"""

import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorSampler, StatevectorEstimator

from h7_shared import PI as π, PHI as φ, h7_parity, h7_quasiperiod

try:
    from qnn import string_to_qnn_seed
except ImportError:
    string_to_qnn_seed = None

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
# π, φ ahora vienen de h7_shared — fuente única, compartida con
# h7_benchmark.py y h7_quantum_oracle.py.
cos = np.cos

OPERATOR = SparsePauliOp.from_list(
    [("XXY", 1), ("XYX", 1), ("YXX", 1), ("YYY", -1)]
)

QUARK_MAP = {
    0: "ddd", 1: "ddu", 2: "dud", 3: "duu",
    4: "udd", 5: "udu", 6: "uud", 7: "uuu",
}

# ─────────────────────────────────────────────
#  PAGE CONFIG + THEME
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="H7 · Z₇ Particle Classifier",
    page_icon="⬡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0d1117; }
    h1, h2, h3, h4, h5, h6, p, span, div, label { color: #e6edf3; }
    .h7-badge {
        display:inline-block; padding:2px 10px; border-radius:6px;
        font-family: monospace; font-size:0.8rem; margin-right:6px;
    }
    .h7-cyan   { background:#0b3d47; color:#67e8f9; }
    .h7-green  { background:#0b3d1f; color:#86efac; }
    .h7-mag    { background:#3d0b3d; color:#f0abfc; }
    .h7-yellow { background:#3d360b; color:#fde68a; }
    .h7-gray   { background:#161b22; color:#8b949e; }
    div[data-testid="stMetricValue"] { font-family: monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⬡ H7 · DIT · Metriplectic QNN")
st.caption("Z₇ Particle Classifier — smokApp Quantum & AI Lab · Streamlit build")

# ─────────────────────────────────────────────
#  CACHED HEAVY RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource
def get_simulator():
    return AerSimulator()


# ─────────────────────────────────────────────
#  INPUT PARSING  (texto o número → n entero)
# ─────────────────────────────────────────────
def parse_input(raw: str):
    """
    Acepta un entero directo o texto libre. Si es texto, lo pasa por
    el cifrado QNN (string_to_qnn_seed) para derivar un seed entero.
    Devuelve (n, info) — info es None si fue entero directo.
    """
    raw = raw.strip()
    try:
        return int(raw), None
    except ValueError:
        pass

    if not raw:
        raise ValueError("empty input")

    if string_to_qnn_seed is None:
        raise RuntimeError(
            "No se encontró el módulo `qnn.py` (string_to_qnn_seed). "
            "Colócalo junto a esta app o ingresa un entero directamente."
        )

    seed = string_to_qnn_seed(raw)
    info = dict(source_text=raw, seed=seed, seed_bin=format(seed, "b"))
    return seed, info


# ─────────────────────────────────────────────
#  PHYSICS  (idéntico al CLI original)
# ─────────────────────────────────────────────
def classify(n: int) -> dict:
    n_z7 = n % 7
    if n_z7 == 0 and n != 0:
        n_z7 = 7

    is_vacuum = (math.gcd(n, 7) == 7)
    o_n = h7_parity(n)        # antes: cos(π·n) — ahora identidad exacta (-1)^n
    i_n = h7_quasiperiod(n)   # cos(π·φ·n), φ con precisión completa desde h7_shared
    r = o_n * i_n
    pcv = r + i_n
    pcc = r - i_n
    ptype = "fermionic" if pcv == 0.0 else "bosonic"

    observable_value = i_n
    contra_val = 7 - n_z7
    contra_bin = f"{contra_val:03b}"
    estado_qubit = f"{n_z7:03b}"
    o_s = n + contra_val - n_z7
    om = n_z7 + contra_val

    return dict(
        n=n, n_z7=n_z7, is_vacuum=is_vacuum,
        o_n=o_n, i_n=i_n, chiral=pcv, color=pcc, o_s=o_s, om=om,
        ptype=ptype, observable_value=observable_value,
        contra_val=contra_val, contra_bin=contra_bin,
        estado_qubit=estado_qubit,
    )


def run_circuit(n_z7, observable_value):
    def z7_to_quaternion(k, angle_moment):
        theta_rot = angle_moment * (π / 4)
        axis = np.array(
            [math.sin(k), cos(k), math.sin(k * φ)], dtype=float
        )
        norm = np.linalg.norm(axis)
        axis = axis / norm if norm > 0 else np.array([0, 0, 1], dtype=float)
        w = math.cos(theta_rot / 2)
        x, y, z = math.sin(theta_rot / 2) * axis
        return np.array([w, x, y, z])

    def quat_to_su2(q):
        w, x, y, z = q
        return np.array(
            [[w + 1j * z, y + 1j * x], [-y + 1j * x, w - 1j * z]],
            dtype=complex,
        )

    simulator = get_simulator()
    q_vec = z7_to_quaternion(n_z7, observable_value)
    su2 = quat_to_su2(q_vec)
    assert np.allclose(su2.conj().T @ su2, np.eye(2), atol=1e-12)

    def run_sv(matrix):
        qc = QuantumCircuit(3)
        qc.unitary(Operator(matrix), [0, 1, 2])
        qc.save_statevector()
        return simulator.run(qc).result().get_statevector(qc)

    sv_orig = run_sv(su2)
    sv_comp = run_sv(quat_to_su2(-q_vec))
    return sv_orig, sv_comp


def build_hardware_circuit(d: dict) -> dict:
    """
    Construye el circuito completo (H + rotaciones + entrelazado H7),
    lo transpila a gate-set nativo, corre counts y calcula el
    expectation value sobre el operador XXY+XYX+YXX-YYY.
    """
    qc = QuantumCircuit(3)
    qc.h(0); qc.h(1); qc.h(2)
    qc.rz(d["n_z7"], 0)
    qc.ry(d["contra_val"], 0)
    qc.rx(d["i_n"], 0)
    qc.ccx(1, 0, 2)
    qc.cswap(0, 2, 1)
    qc.ccx(2, 1, 0)

    estimator = StatevectorEstimator()
    job_est = estimator.run([(qc, OPERATOR)])
    expval = job_est.result()[0].data.evs

    qc_transpiled = transpile(
        qc,
        basis_gates=["cz", "sx", "rz"],
        coupling_map=[[2, 1], [0, 1], [0, 2]],
        optimization_level=3,
    )

    qc_measured = qc.measure_all(inplace=False)
    simulator = get_simulator()
    counts = simulator.run(
        transpile(qc_measured, simulator), shots=1024
    ).result().get_counts()

    sampler = StatevectorSampler()
    job = sampler.run([qc_measured], shots=1024)
    result = job.result()
    sampler_counts = result[0].data["meas"].get_counts()

    estimator2 = StatevectorEstimator()
    job2 = estimator2.run([(qc, OPERATOR)], precision=1e-3)
    expval_precise = job2.result()[0].data.evs

    return dict(
        qc=qc,
        qc_transpiled=qc_transpiled,
        counts=counts,
        sampler_counts=sampler_counts,
        expval=expval,
        expval_precise=expval_precise,
    )


# ─────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────
def badge(text, cls):
    return f'<span class="h7-badge {cls}">{text}</span>'


def show_classification(d: dict):
    st.subheader("H7 Particle Classification")

    quarks = QUARK_MAP.get(d["n_z7"], d["contra_bin"])
    quarksb = QUARK_MAP.get(d["contra_val"], d["contra_bin"])
    pcolor_cls = "h7-mag" if d["ptype"] == "fermionic" else "h7-green"
    flow = (
        "→ Pauli exclusion (topological · neutron)"
        if d["ptype"] == "fermionic"
        else "→ Continuous flow (JIT queue · proton)"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n (original)", d["n"])
    c2.metric("n mod 7 (Z₇)", d["n_z7"])
    c3.metric("Hidden n (index)", d["o_s"])
    c4.metric("hn mod 7 (Z₇)", d["contra_val"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Observable mod", d["n_z7"] + d["contra_val"])
    c2.metric("Parity cos(π·n₇)", f"{d['o_n']:+.1f}")
    c3.metric("Quasiperiod cos(πφn₇)", f"{d['i_n']:+.6f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Chiral (q+p)", f"{d['chiral']:+.4f}")
    c2.metric("Chiral negative (q-p)", f"{d['color']:+.4f}")
    c3.metric("Vacuum / boundary", "sí" if d["is_vacuum"] else "no")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Complement (7−n₇)**  `{d['n_z7']} → {d['estado_qubit']}`")
        st.markdown(f"**Complement (7−in₇)**  `{d['contra_val']} → {d['contra_bin']}`")
        st.markdown(f"**Isotope (H)**  `{d['om']} → {d['om']:03b}`")
    with c2:
        st.markdown(f"**H7 state**  `{d['estado_qubit']}, {d['contra_bin']}`")
        st.markdown(f"**Quark composition**  `{(quarksb, quarks)}`")
        st.markdown(
            f"**Quaternionic axis**  `Z₇[{d['estado_qubit']}] × Q₈[{d['contra_bin']}]`"
        )

    st.divider()
    st.markdown(
        f"### Particle type: {badge(d['ptype'].upper(), pcolor_cls)}",
        unsafe_allow_html=True,
    )
    st.caption(flow)


def _statevector_bar(sv, title, color):
    amps = np.round(sv.data, 4)
    labels = [f"|{i}⟩" for i in range(len(amps))]
    mags = np.abs(amps)
    hover = [f"{a}" for a in amps]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels, y=mags, marker_color=color,
                text=hover, textposition="outside",
                hovertemplate="%{x}: %{text}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis_title="|amplitude|",
    )
    return fig


def show_statevectors(d: dict):
    st.subheader("Statevectors")
    sv_orig, sv_comp = run_circuit(d["n_z7"], d["o_n"])

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _statevector_bar(sv_orig, "ψ original (SU(2) · q_vec)", "#67e8f9"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            _statevector_bar(sv_comp, "ψ̄ complementary (SU(2) · −q_vec)", "#f0abfc"),
            use_container_width=True,
        )


def show_qasm(d: dict):
    st.subheader("OpenQASM 2.0 — 3-Qubit Equivalent Mapping")
    lines = [
        "// Initialization",
        "h q[0]; h q[1]; h q[2];",
        f"// Quasiperiodic phase modulation  rz(cos(π·φ·{d['n_z7']}))",
        f"rz({d['n_z7']:.1f}) q[0];",
        f"ry({d['contra_val']:.1f}) q[0];",
        f"rx({d['i_n']:.3f}) q[0];",
        "// Native entanglement layer (H7 CSWAP + Toffoli ansatz)",
        "ccx   q[1], q[0], q[2];",
        "cswap q[0], q[2], q[1];",
        "ccx   q[2], q[1], q[0];",
        "measure q -> c;",
    ]
    st.code("\n".join(lines), language="c")


def show_hardware(d: dict):
    st.subheader("Hardware Output")
    with st.spinner("Transpilando y ejecutando circuito..."):
        hw = build_hardware_circuit(d)

    c1, c2 = st.columns(2)
    c1.metric("Expectation value (H7 operator)", f"{hw['expval']:.4f}")
    c2.metric("Expectation value (precision=1e-3)", f"{hw['expval_precise']:.4f}")

    st.markdown("**Circuito lógico**")
    st.text(hw["qc"].draw(output="text"))

    st.markdown("**Circuito transpilado** (basis_gates=[cz, sx, rz])")
    st.text(hw["qc_transpiled"].draw(output="text"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Counts (AerSimulator, 1024 shots)**")
        st.bar_chart(hw["counts"])
    with c2:
        st.markdown("**Counts (StatevectorSampler, 1024 shots)**")
        st.bar_chart(hw["sampler_counts"])


# ─────────────────────────────────────────────
#  SIDEBAR — INPUT
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Input")
    raw = st.text_input("Entero o palabra", placeholder="ej. 42  ó  isotopoH")
    run_btn = st.button("Clasificar", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "Si ingresas texto, se convierte a seed vía `string_to_qnn_seed` "
        "(módulo `qnn.py` requerido en el mismo directorio)."
    )

if "h7_data" not in st.session_state:
    st.session_state.h7_data = None
if "h7_seed_info" not in st.session_state:
    st.session_state.h7_seed_info = None

if run_btn:
    if not raw or not raw.strip():
        st.sidebar.error("✗ Entrada vacía.")
    else:
        try:
            n, seed_info = parse_input(raw)
            st.session_state.h7_data = classify(n)
            st.session_state.h7_seed_info = seed_info
        except Exception as e:
            st.sidebar.error(f"✗ {e}")

# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
if st.session_state.h7_seed_info is not None:
    info = st.session_state.h7_seed_info
    with st.expander("Text → QNN Seed → Binary", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Texto de entrada", f"'{info['source_text']}'")
        c2.metric("Seed (Σ cifrado QNN)", info["seed"])
        c3.metric("Seed binario", info["seed_bin"])

if st.session_state.h7_data is None:
    st.info(
        "Ingresa un entero o una palabra en la barra lateral y presiona "
        "**Clasificar** para comenzar."
    )
else:
    d = st.session_state.h7_data
    tabs = st.tabs(
        ["Classification", "Statevectors", "OpenQASM", "Hardware output"]
    )
    with tabs[0]:
        show_classification(d)
    with tabs[1]:
        show_statevectors(d)
    with tabs[2]:
        show_qasm(d)
    with tabs[3]:
        show_hardware(d)
