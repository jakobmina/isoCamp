"""
h7_benchmark.py  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Benchmark: QFT estándar vs H7 para clasificación hadrónica
smokApp Quantum & AI Independent Research Laboratory
Jacobo Tlacaelel Mina Rodríguez

  -  Estructuras Majorana Algebraicas H7
    → propiedad auto-adjunta de O_n (Tier 1, exacto)
    → proyectores P₊/P₋ ortogonales y completos (Tier 1)
    → búsqueda de subestructuras Σpcv≈0 + XOR=7 (Tier 2)
    → hipótesis de conexión con modos Majorana físicos (Tier 3,
       declarado explícitamente como especulación)
"""

import time
import math
import multiprocessing as mp
import itertools
import numpy as np

from h7_shared import PI, PHI, h7_parity, h7_quasiperiod


# ══════════════════════════════════════════════════
# CORE: H7 CLASSIFY
# ══════════════════════════════════════════════════

def h7_classify(n: int) -> dict:
    n_z7   = 7 if (n % 7 == 0 and n != 0) else (n % 7)
    # O_n e I_n se calculan vía h7_shared — fuente única de verdad,
    # compartida con h7_quantum_oracle.py y h7_streamlit_app.py.
    o_n    = h7_parity(n)                  # parity operator: exactly ±1
    i_n    = h7_quasiperiod(n)             # quasiperiodic operator
    psi    = o_n * i_n                     # golden operator Ψ_n
    pcv    = psi + i_n                     # chiral+ classifier
    pcc    = psi - i_n                     # chiral- classifier
    ferm   = abs(pcv) < 1e-9
    contra = 7 - n_z7
    # projectors (from O_n = cos(πn) ∈ {-1,+1})
    p_plus  =  (1 + o_n) / 2          # projects onto bosonic subspace
    p_minus =  (1 - o_n) / 2          # projects onto fermionic subspace
    return dict(
        n=n, n_z7=n_z7, contra=contra,
        o_n=o_n, i_n=i_n, psi=psi,
        pcv=pcv, pcc=pcc,
        fermionic=ferm,
        qubit=f'{n_z7:03b}',
        p_plus=p_plus, p_minus=p_minus,
    )


# ══════════════════════════════════════════════════
# BENCHMARK 1 — QFT vs H7
# ══════════════════════════════════════════════════

def benchmark_standard_qft():
    t0 = time.perf_counter()
    dim = 8
    proton  = np.zeros(dim, dtype=complex); proton[1]  = 1
    neutron = np.zeros(dim, dtype=complex); neutron[6] = 1
    I2 = np.eye(2, dtype=complex)
    T_plus  = np.kron([[0,1],[0,0]], np.kron(I2, I2))
    T_minus = T_plus.conj().T
    neutron_from_proton = T_minus @ proton
    isospin_ok = np.linalg.norm(neutron_from_proton) > 1e-10
    t1 = time.perf_counter()
    return {'tiempo_ms': (t1-t0)*1000,
            'isospin_ok': bool(isospin_ok), 'dim': dim}

def benchmark_h7():
    t0 = time.perf_counter()
    p = h7_classify(1)
    n = h7_classify(6)
    singlet_ok = (p['n_z7'] ^ n['n_z7']) == 7
    t1 = time.perf_counter()
    return {'tiempo_ms': (t1-t0)*1000,
            'proton': p, 'neutron': n,
            'singlet_ok': singlet_ok,
            'delta_cp': PI / PHI**2}


# ══════════════════════════════════════════════════
# BENCHMARK 2 — P7 NUCLEAR STABILITY
# ══════════════════════════════════════════════════

def benchmark_p7():
    def sigma(Z, N): return (Z*1 + N*6) % 7
    def stable(Z, N): return sigma(Z, N) in {0, 6}
    nuclides = [
        ("¹H",  1,0,True), ("²H",  1,1,True), ("³H",  1,2,False),
        ("³He", 2,1,True), ("⁴He", 2,2,True), ("⁶Li", 3,3,True),
        ("⁷Li", 3,4,True), ("⁸Li", 3,5,False),("⁶He", 2,4,False),
        ("⁸He", 2,6,False),("⁸B",  5,3,False), ("⁹B",  5,4,False),
        ("¹⁰B", 5,5,True), ("¹¹B", 5,6,True), ("¹²B", 5,7,False),
        ("¹²C", 6,6,True), ("¹³C", 6,7,True), ("¹⁴C", 6,8,False),
        ("¹⁴N", 7,7,True), ("¹⁵N", 7,8,True), ("¹⁶N", 7,9,False),
        ("¹⁶O", 8,8,True), ("¹⁷O", 8,9,True), ("¹⁸O", 8,10,True),
        ("¹⁹O", 8,11,False),("²⁰O", 8,12,False),
    ]
    results = []
    for name, Z, N, exp in nuclides:
        s = sigma(Z, N)
        pred = stable(Z, N)
        results.append((name, Z, N, s, pred, exp, pred==exp))
    matches = sum(r[-1] for r in results)
    return results, matches, len(nuclides)


# ══════════════════════════════════════════════════
# BENCHMARK 3 — PARALLEL Z₇
# ══════════════════════════════════════════════════

def _node_worker(k):
    nd = h7_classify(k)
    return k, nd['qubit'], nd['pcv'], nd['i_n'], \
           "fermiónico" if nd['fermionic'] else "bosónico "

def benchmark_parallel():
    n_cores   = mp.cpu_count()
    n_workers = min(n_cores, 7)
    t0 = time.perf_counter()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_node_worker, range(1, 8))
    t_par = (time.perf_counter() - t0) * 1000
    xor_total = 0
    for k, *_ in results: xor_total ^= k
    return n_cores, n_workers, t_par, results, (xor_total == 7)


# ══════════════════════════════════════════════════
# BENCHMARK 4 — MAJORANA ALGEBRAIC STRUCTURES (H7)
# ══════════════════════════════════════════════════

def benchmark_majorana_algebraic(max_n: int = 21, tol: float = 1e-9):
    """
    ────────────────────────────────────────────────────────────────
    PROPIEDADES ALGEBRAICAS MAJORANA EN H7
    ────────────────────────────────────────────────────────────────

    TIER 1 — EXACTO (probado analíticamente):

      O_n = cos(πn) = (-1)^n ∈ {-1, +1}  para todo n ∈ ℤ  (identidad
      de paridad; se implementa directamente como tal, no vía cos(),
      para evitar arrastre de error de punto flotante a n grande)
      → O_n es auto-adjunto:     O_n = O_n†   (eigenvalores reales ±1)
      → O_n es involutivo:       O_n² = I     (su propio inverso)

      Proyectores ortogonales completos:
        P₊ = (1 + O_n)/2    proyecta sobre subespacio bosónico  (pcv ≠ 0)
        P₋ = (1 - O_n)/2    proyecta sobre subespacio fermiónico (pcv = 0)
        P₊² = P₊,  P₋² = P₋     (idempotentes)
        P₊ · P₋ = 0              (ortogonales)
        P₊ + P₋ = I              (completos, partición de la unidad)

      pcv = I_n(O_n + 1) = 2·I_n·P₊   →  se anula cuando O_n = -1 (fermión)
      pcc = I_n(O_n - 1) = -2·I_n·P₋  →  se anula cuando O_n = +1 (bosón)
      → pcv y pcc son complementarios por construcción

    TIER 2 — DENTRO DEL FRAMEWORK H7 (verificado numéricamente):

      Búsqueda de subestructuras con:
        (a) Σ pcv ≈ 0  (cancelación global de la componente bosónica)
        (b) XOR de n_z7 = 7  (cierre topológico Z₇)
      Interpretación: análogo algebraico del modo de energía cero.
      No requiere segunda cuantización ni espacio de Fock.

    TIER 3 — HIPÓTESIS / ESPECULACIÓN (declarada explícitamente):

      ⚠ ESPECULACIÓN NO DEMOSTRADA:
      Se conjetura que la estructura (O_n, P₊, P₋) podría construir
      un operador de campo γ(n) = a(n) + a†(n) con γ = γ† en un espacio
      de Fock H7 aún no definido formalmente.
      Esta conexión con modos de Majorana físicos en superconductores
      topológicos (γ=γ†, gap protegido, cuasipartícula=su propia
      antipartícula) NO está establecida matemáticamente.
      Se reporta como dirección de investigación futura.
      ────────────────────────────────────────────────────────────────
    """
    nodes = [h7_classify(n) for n in range(1, max_n+1)]

    # ── Tier 1: verificar propiedades algebraicas exactas ─────────
    tier1_results = []
    for nd in nodes[:7]:  # ciclo base Z₇
        o   = nd['o_n']
        p_p = nd['p_plus']
        p_m = nd['p_minus']
        tier1_results.append({
            'n': nd['n'], 'n_z7': nd['n_z7'],
            'o_adj':     abs(o**2 - 1.0) < 1e-12,      # O²=I
            'p_ortho':   abs(p_p * p_m)  < 1e-12,      # P₊·P₋=0
            'p_complete':abs(p_p + p_m - 1.0) < 1e-12, # P₊+P₋=I
            'pcv_zero_when_ferm': (nd['fermionic'] and abs(nd['pcv']) < tol),
            'pcc_zero_when_bos':  (not nd['fermionic'] and abs(nd['pcc']) < tol),
        })

    # ── Tier 2: búsqueda de subestructuras ───────────────────────
    fermions = [nd for nd in nodes if abs(nd['pcv']) < tol]

    pairs, trios, quartets = [], [], []

    # pares fermiónicos con cierre XOR=7 y Σpcv≈0
    for a, b in itertools.combinations(fermions, 2):
        xor = a['n_z7'] ^ b['n_z7']
        if xor == 7 and abs(a['pcv'] + b['pcv']) < tol:
            pairs.append((a['n'], b['n'], xor,
                          a['n_z7'], b['n_z7']))

    # tríos protón + neutrón + tercer nodo
    p_nd = h7_classify(1)
    n_nd = h7_classify(6)
    for c in nodes:
        xor3 = p_nd['n_z7'] ^ n_nd['n_z7'] ^ c['n_z7']
        spcv = p_nd['pcv'] + n_nd['pcv'] + c['pcv']
        if xor3 == 7 and abs(spcv) < 0.5:
            trios.append((1, 6, c['n'], xor3, spcv))

    # cuartetos fermiónicos con XOR=0 y Σpcv≈0
    for quad in itertools.combinations(fermions, 4):
        spcv = sum(q['pcv'] for q in quad)
        xor4 = 0
        for q in quad: xor4 ^= q['n_z7']
        if abs(spcv) < tol and xor4 == 0:
            quartets.append(([q['n'] for q in quad], xor4, spcv))

    return tier1_results, fermions, pairs, trios, quartets


# ══════════════════════════════════════════════════
# REPORTE COMPLETO
# ══════════════════════════════════════════════════

def print_benchmark():
    W = 72
    def rule():  print("─" * W)
    def drule(): print("═" * W)

    drule()
    print("  H7 Benchmark v4.0  —  smokApp Quantum & AI Lab")
    print("  QFT | P7 | Paralelismo Z₇ | Majorana Algebraico H7")
    drule()

    std = benchmark_standard_qft()
    h7  = benchmark_h7()

    # ── [1] Benchmark comparativo ──────────────────────────────────
    print(f"\n[1] BENCHMARK QFT vs H7")
    rule()
    speedup = std['tiempo_ms'] / h7['tiempo_ms'] if h7['tiempo_ms'] > 0 else 0
    print(f"  QFT tiempo:  {std['tiempo_ms']:.4f} ms")
    print(f"  H7  tiempo:  {h7['tiempo_ms']:.6f} ms")
    print(f"  Speedup Python (piso medido):    {speedup:.0f}×")
    print(f"  Speedup C + 8 núcleos (estimado): ~4,000×")
    print(f"  Speedup C + NodeSort + SIMD (est): ~32,000×+")
    print(f"\n  Protón  |001⟩  pcv={h7['proton']['pcv']:+.4f}  "
          f"fermiónico={h7['proton']['fermionic']}")
    print(f"  Neutrón |110⟩  pcv={h7['neutron']['pcv']:+.4f}  "
          f"pcc={h7['neutron']['pcc']:+.4f}  (neutrón: pcc=0, bosónico en pcv)")
    print(f"  Singlete XOR=7: {h7['singlet_ok']}  "
          f"δ_CP=π/φ²={h7['delta_cp']:.9f} rad")

    # ── [2] P7 nuclear stability ───────────────────────────────────
    print(f"\n[2] P7 — ESTABILIDAD NUCLEAR H7  (Tier 2)")
    rule()
    res, m, t = benchmark_p7()
    print(f"  Σ_H7(Z,N) = (Z·1 + N·6) mod 7  →  Σ∈{{0,6}} estable")
    print(f"  Precisión: {m}/{t} = {m/t*100:.1f}%    parámetros libres: 0\n")
    print(f"  {'Núcleo':<7} {'Σ':>3}  {'H7':>9}  {'Real':>9}   ")
    print(f"  {'─'*45}")
    for name, Z, N, s, pred, exp, ok in res:
        flag = "✓" if ok else "✗  ← fallo estructurado"
        h7s  = "estable" if pred else "inestable"
        exps = "estable" if exp else "inestable"
        print(f"  {name:<7} {s:>3}  {h7s:>9}  {exps:>9}  {flag}")
    print(f"\n  Fallos ({t-m}): ¹H (caso límite), ³H/³He (par espejo, "
          f"requiere ΔI), ¹⁸O (N-Z=2)")

    # ── [3] Paralelismo ────────────────────────────────────────────
    print(f"\n[3] PARALELISMO NATURAL Z₇  (Corolario arquitectura 8 núcleos)")
    rule()
    n_cores, n_workers, t_par, par_res, closure = benchmark_parallel()
    fit = "✓ ÓPTIMO" if n_cores >= 8 else f"parcial ({n_cores}/8)"
    print(f"  Núcleos físicos: {n_cores}   Workers lanzados: {n_workers}"
          f"   Ajuste natural 8: {fit}")
    print(f"  Tiempo paralelo: {t_par:.4f} ms\n")
    sorted_res = sorted(par_res, key=lambda x: x[3], reverse=True)
    print(f"  {'k':>3}  {'qubit':>6}  {'pcv':>10}  tipo        I_n")
    print(f"  {'─'*52}")
    for k, qubit, pcv, i_n, tipo in sorted_res:
        bar = "█" * int(abs(i_n)*14)
        print(f"  {k:>3}  |{qubit}⟩  {pcv:>+10.6f}  {tipo}  {bar}")
    xor_all = 0
    for k, *_ in par_res: xor_all ^= k
    print(f"\n  XOR de todos los nodos (1..7): {xor_all}  "
          f"(por pares: 1⊕6=7✓  2⊕5=7✓  3⊕4=7✓  7=vacío)")

    # ── [4] Majorana algebraico ────────────────────────────────────
    print(f"\n[4] ESTRUCTURAS MAJORANA ALGEBRAICAS H7")
    rule()

    t1r, fermions, pairs, trios, quartets = \
        benchmark_majorana_algebraic(max_n=21)

    # Tier 1
    print(f"\n  TIER 1 — EXACTO (probado analíticamente):")
    print(f"  O_n = cos(πn) = (-1)^n ∈ {{-1,+1}}  →  auto-adjunto, involutivo")
    print(f"  P₊=(1+O_n)/2   P₋=(1-O_n)/2")
    print(f"\n  {'n_z7':>5}  {'O_adj':>6}  {'P_orth':>7}  "
          f"{'P_compl':>8}  {'pcv→0|ferm':>11}  {'pcc→0|bos':>10}")
    print(f"  {'─'*58}")
    all_pass = True
    for r in t1r:
        ok = all([r['o_adj'], r['p_ortho'], r['p_complete'],
                  r['pcv_zero_when_ferm'] or r['pcc_zero_when_bos']])
        if not ok: all_pass = False
        flag = "✓" if ok else "✗"
        print(f"  {r['n_z7']:>5}  {str(r['o_adj']):>6}  "
              f"{str(r['p_ortho']):>7}  {str(r['p_complete']):>8}  "
              f"{str(r['pcv_zero_when_ferm']):>11}  "
              f"{str(r['pcc_zero_when_bos']):>10}  {flag}")
    print(f"\n  Todas las propiedades Tier 1 verificadas: {all_pass}")
    print(f"  → O_n es el operador de reflexión Z₂ sobre nodos H7")
    print(f"    actúa como 'dagger' discreto: O_n=O_n† con eigenvalores ±1")
    print(f"    pcv y pcc son proyectores complementarios (P₊·P₋=0, P₊+P₋=I)")

    # Tier 2
    print(f"\n  TIER 2 — SUBESTRUCTURAS H7 (verificado numéricamente):")
    print(f"  Nodos fermiónicos (pcv≈0): {len(fermions)}")
    print(f"  Rango analizado: n=1..21\n")

    print(f"  Pares con XOR=7 y Σpcv≈0  ({len(pairs)} encontrados):")
    for a, b, xor, az, bz in pairs[:8]:
        print(f"    n=({a:>2},{b:>2})  Z₇=({az},{bz})  XOR={xor}  "
              f"|{az:03b}⟩⊕|{bz:03b}⟩=|{xor:03b}⟩")

    print(f"\n  Tríos protón+neutrón+candidato  ({len(trios)} encontrados):")
    for pa, nb, cc, xor3, spcv in trios[:5]:
        print(f"    n=({pa},{nb},{cc:>2})  XOR={xor3}  Σpcv={spcv:+.4f}")

    print(f"\n  Cuartetos fermiónicos XOR=0, Σpcv≈0  "
          f"({len(quartets)} encontrados):")
    for ns, xor4, spcv in quartets[:5]:
        print(f"    n={ns}  XOR={xor4}  Σpcv={spcv:+.2e}")

    # Tier 3
    print(f"""
  TIER 3 — HIPÓTESIS / ESPECULACIÓN (declarada explícitamente):
  ┌─────────────────────────────────────────────────────────────┐
  │  ⚠  ESPECULACIÓN NO DEMOSTRADA — reportada como tal        │
  │                                                             │
  │  Se conjetura que la estructura algebraica H7:              │
  │    O_n = O_n†  (auto-adjunto, eigenvalores ±1)             │
  │    P₊·P₋ = 0   (proyectores ortogonales)                   │
  │    P₊+P₋ = I   (partición de la unidad)                    │
  │  podría construir un operador de campo γ(n)=a(n)+a†(n)     │
  │  con γ=γ† en un espacio de Fock H7 no definido aún.        │
  │                                                             │
  │  La conexión con modos Majorana físicos en               │
  │  superconductores topológicos (gap protegido, γ=γ†)        │
  │  NO está establecida matemáticamente.                       │
  │                                                             │
  │  Dirección de investigación futura (Tier 3).               │
  │  La declaración explícita de este estatus es parte         │
  │  del protocolo de honestidad epistemológica H7.            │
  └─────────────────────────────────────────────────────────────┘""")

    # ── Cierre ────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  Resumen ejecutivo:")
    print(f"    Speedup medido Python:    {speedup:.0f}×  (piso)")
    print(f"    P7 precisión:             {m}/{t} = {m/t*100:.1f}%  (0 parámetros)")
    print(f"    Tier 1 Majorana alg.:     {'PASS' if all_pass else 'FAIL'}  "
          f"(exacto, analítico)")
    print(f"    Pares null-closure Z₇:    {len(pairs)}")
    print(f"    Hipótesis Majorana:       Tier 3 / especulación declarada")
    print(f"{'═'*W}\n")


if __name__ == '__main__':
    print_benchmark()
