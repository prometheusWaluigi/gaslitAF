# Unified GASLIT-AF Theory and Simulation Plan

## Purpose
This document evaluates the current core theories in `coreTheories/` and consolidates them into a single, testable framework with a concrete simulation plan.

---

## Evaluation of Core Theories

### 1. SystemsBiologyGASLITAF.md
**Strengths**
- Provides a rigorous, formal dynamical-systems backbone (state space, ODEs/DDEs/SDEs, stability and bifurcation analysis).
- Introduces a coherent parameter vocabulary (γ, Λ, Ω, σ, Θ, Ξ) that maps well to physiological interpretations.

**Gaps to address**
- Mechanistic specificity is under-defined: it describes the math well, but concrete physiological loop definitions are sparse.
- The theory is broad enough to be hard to validate without a minimal, testable sub-model.

### 2. systemsBiologyModel.md
**Strengths**
- Provides a focused neuroimmune/limbic attractor model with clear equations and clinical evidence anchors.
- Strong on longitudinal, post-viral persistence logic and hysteresis.

**Gaps to address**
- Tuned toward post-COVID/PVS; needs generalization to CMSS beyond that context.
- Needs integration with structural and genetic vulnerability themes (e.g., connective tissue, RCCX, polygenic risk).

### 3. gaslitDynamicalRefined.md
**Strengths**
- Proposes a minimal sub-model (autonomic, immune, fatigue) that is immediately testable.
- Offers a concrete validation strategy (retrospective fitting + prospective studies).

**Gaps to address**
- Minimal model is not yet explicitly tied to structural and metabolic loops.
- Needs explicit mapping between parameters and measurable biomarkers for reproducibility.

### 4. gaslitZebraUnified.md
**Strengths**
- Integrates the GASLIT-AF attractor model with genetic/structural “zebra” predisposition.
- Provides a roadmap for computational modeling, including machine learning augmentation.

**Gaps to address**
- Heavily conceptual; requires a tighter core mechanism that can be simulated and falsified.
- AI augmentation ideas need a baseline mechanistic model before adding ML complexity.

### 5. integrativeSystemsModelCMSS.md
**Strengths**
- Makes the genetic vulnerability (RCCX and related variants) explicit and maps it to multi-system loops.
- Offers explicit feedback loop descriptions connecting immune, autonomic, metabolic, neural, and structural systems.

**Gaps to address**
- Overly broad in scope; could benefit from clearer prioritization of the dominant loops that drive the attractor.
- Needs a more compact, parameterized core model to avoid being descriptive-only.

### 6. criticalReview.md and critiqueGASLITAFCMSS.md
**Strengths**
- Provide a rigorous critique of overreach, lack of mechanistic specificity, and genetic overemphasis.
- Highlight heterogeneity and the need for subtypes rather than a single monolith.

**Gaps to address**
- The critique does not specify a constructive, minimal model or testing path.
- Needs to be translated into actionable constraints on the unified theory.

---

## Consolidated Theory: The Multi-Loop Attractor (MLA) Model

### Core Claim
Chronic Multi-System Syndromes (CMSS) emerge when a susceptible system crosses a critical threshold into a self-sustaining attractor state. This attractor is maintained by *three dominant coupled loops* that integrate the broader GASLIT-AF framework while remaining testable:

1. **Neuroimmune Loop (I ↔ N):** Immune activation drives neuroinflammation and autonomic dysregulation, which in turn sustains immune activation.
2. **Stress–Endocrine Loop (H ↔ I/N):** HPA axis dysfunction and limbic stress signaling amplify immune activation and autonomic instability.
3. **Structural–Vascular Loop (S ↔ A):** Connective tissue fragility and vascular instability increase autonomic load, which worsens structural stress and inflammatory signaling.

These loops are *modulated* by:
- **Genetic predisposition (γ):** Polygenic vulnerability (e.g., RCCX, HLA, ECM-related genes).
- **Allostatic load (Λ):** Cumulative stress and immune triggers.
- **Buffering capacity (Ω):** Endocannabinoid tone, mitochondrial resilience, vagal tone.
- **Noise (σ):** Physiological and environmental stochasticity.

This yields a model that is both mechanistic (explicit loops) and generalizable (parameters can be tuned to subtypes).

---

## Minimal Testable Model (Core Dynamical System)

We define a 4-state minimal model that captures the dominant loops while remaining measurable:

- **I(t)** = immune activation (cytokine index, CRP/IL-6 composite)
- **A(t)** = autonomic stability (HRV index / orthostatic tolerance)
- **H(t)** = stress/HPA axis drive (cortisol dynamics + limbic stress score)
- **S(t)** = structural–vascular fragility (connective tissue/vascular compliance proxy)

### Prototype Equations (conceptual form)

```
I' = -αI * I + βIN * f(A) + βIH * g(H) + uI(t) + σI * ξI
A' = -αA * A + βAI * h(I) + βAS * k(S) + σA * ξA
H' = -αH * H + βHA * m(A) + external_stress(t)
S' = -αS * S + βSI * n(I) + βSA * p(A)
```

Where f,g,h,k,m,n,p are saturating (Hill-type) functions with thresholds (Θ). The model expresses:
- **I ↔ A**: immune activation destabilizes autonomic tone; dysautonomia feeds back into inflammation.
- **H ↔ I/A**: stress dysregulates immune/autonomic feedback (and vice versa).
- **S ↔ A/I**: structural fragility elevates autonomic strain and inflammation; chronic inflammation worsens structural integrity.

This minimal system is small enough for bifurcation analysis yet rich enough to encode the core dynamics.

---

## Unified Theory Predictions (Falsifiable)

1. **Bistability:** Under certain parameter regimes, the system exhibits two stable equilibria (healthy vs. pathological attractor). Transition thresholds are observable.
2. **Hysteresis:** Removing the initial trigger (uI or external_stress) does not restore health without multi-loop intervention.
3. **Subtype Patterns:** Different parameter regimes yield immune-dominant, autonomic-dominant, or structural-dominant phenotypes.
4. **Early-Warning Signals:** Rising variance, delayed recovery, or critical slowing down in HRV/cytokine dynamics precede collapse.
5. **Synergistic Therapy Effects:** Multi-target interventions outperform single-loop interventions in destabilizing the pathological attractor.

---

## Simulation Plan

### Phase 1 — Baseline Model and Stability Mapping
**Goal:** Prove bistability and attractor transitions.
- Implement the 4-state model in Python (SciPy/NumPy).
- Conduct bifurcation analysis by varying Λ and γ proxies (e.g., scaling β terms).
- Identify threshold values and hysteresis loops.

**Outputs:**
- Bifurcation diagrams (Λ vs. state variables).
- Attractor basin maps with healthy vs. pathological states.

---

### Phase 2 — Parameterization With Real-World Proxies
**Goal:** Map parameters to measurable biomarkers.
- I(t): CRP/IL-6 composite, cytokine panels.
- A(t): HRV indices, orthostatic vitals.
- H(t): cortisol diurnal slope + stress inventories.
- S(t): Beighton score + vascular compliance proxies.

**Outputs:**
- Parameter priors and scaling rules.
- Data dictionary for mapping clinical measures → model states.

---

### Phase 3 — Retrospective Fitting
**Goal:** Validate model fit on existing datasets.
- Fit to ME/CFS, POTS, Long COVID cohorts with time-series data.
- Use Bayesian inference or particle filtering to infer β and α parameters.

**Outputs:**
- Posterior distributions for parameters.
- Fit-quality metrics and subtype clustering.

---

### Phase 4 — Prospective Simulation Trials
**Goal:** Simulate interventions and predict outcomes.
- Single-loop intervention: immune-only (reduce βIN or uI).
- Multi-loop intervention: immune + autonomic + stress modulation.
- Compare predicted recovery rates and relapse probability.

**Outputs:**
- Intervention response curves.
- Predicted optimal multi-target combos.

---

### Phase 5 — Early-Warning Signal Detection
**Goal:** Identify pre-collapse signals.
- Measure variance, autocorrelation, and recovery time in A(t)/I(t).
- Determine thresholds for “approaching attractor shift.”

**Outputs:**
- Quantitative early-warning markers.
- Clinical monitoring recommendations.

---

## Integration With Existing Theories
This unified model preserves the strengths of the existing corpus while meeting the critique requirements:
- **Mathematical rigor** from SystemsBiologyGASLITAF.
- **Empirical grounding** from systemsBiologyModel and integrativeSystemsModelCMSS.
- **Minimal testability** from gaslitDynamicalRefined.
- **Genetic susceptibility integration** from gaslitZebraUnified.
- **Critical guardrails** from criticalReview and critiqueGASLITAFCMSS.

---

## Next Actions (Implementation Checklist)
1. Implement the 4-state model and run initial bifurcation scans.
2. Build a parameter mapping sheet for biomarkers → model variables.
3. Select 1–2 retrospective datasets for fitting.
4. Define a prospective cohort protocol aligned with Phase 4.

