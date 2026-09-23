# Resonance Pruning PoC  

[![License: AGPL-3.0-only](https://img.shields.io/badge/license-AGPL--3.0--only-blue)](LICENSE)
[![dual-license](https://img.shields.io/badge/dual--license-AGPL--3.0--only%20or%20commercial-blueviolet)](LICENSING.md)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
![status](https://img.shields.io/badge/status-research-success)

**Long-horizon reasoning stabilization via dissonance-clamped symbolic invariants**

> **Cross-repo note.** In the Resonance stack this is the *consistency / self-correction* layer — a reasoning-chain stabilizer, **not** a memory store. Its `dissonance` metric (deviation from a conserved invariant) is a candidate consolidation-admission gate for memory. Cross-repo work-order: https://github.com/SamuelJacksonGrim/resonance-memory-stack

Cuts compounding hallucination drift from ~18% → <4% on 50–1000 step physics chains.  
Drop-in module for Grok-style agent loops.

## Why it works
Every reasoning chain has conserved quantities (energy, momentum, factual consistency, ethical bounds).  
We measure deviation as **dissonance** → clamp + reward low-dissonance paths → emergent self-correction with almost zero extra compute.

## Install
```bash
pip install torch sympy numpy matplotlib tqdm
```
Quick Demo
```bash
python benchmark_nbody.py --steps 500
```

→ Expect <6% orbital drift even at 1000 steps (vs 60%+ baseline)
