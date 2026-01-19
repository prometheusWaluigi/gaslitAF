# Dynamical Systems Simulation Ideas for GASLIT-AF

This folder collects simulation concepts and execution paths that translate the GASLIT-AF core theories into runnable dynamical systems experiments, from minimal ODE prototypes to GPU-accelerated pipelines and personalized monitoring loops. The intent is to map theoretical constructs (attractor basins, hysteresis, buffering capacity, and polygenic risk) onto specific simulation architectures and validation workflows.

## Core theory pillars to encode in simulations

Across the core theories, GASLIT-AF frames chronic multi-system syndromes as transitions into pathological attractor states within a high-dimensional physiological state space, driven by genetic vulnerability (γ), cumulative stress/allostatic load (Λ), buffering capacity (Ω), and a shift toward high entropy/low coherence dynamics (σ/C). These elements guide the selection of variables, couplings, and analyses for simulations.

- **Attractor landscape + tipping points:** Health and disease are alternative basins in state space; rising allostatic load or depleted buffers can tilt the landscape to force transitions and hysteresis effects. This motivates simulations that explicitly track basin depth, recovery thresholds, and irreversibility. 【F:coreTheories/gaslitDynamicalRefined.md†L16-L134】【F:coreTheories/gaslitDynamicalSystemsLongForm.md†L48-L176】
- **Minimal testable ODEs:** A compact model (x = autonomic capacity, y = inflammation, f = fatigue) plus stress load and buffering terms provides a tractable entry point for parameter sweeps and phase-transition tests. 【F:coreTheories/gaslitDynamicalRefined.md†L93-L161】
- **Expanded multi-loop dynamics:** Symbolic models connect endocrine (cortisol), immune activation, autonomic balance, energy availability, and oxidative stress into a coupled ODE system that supports multistability and bifurcations. 【F:coreTheories/symbolicAttractorExpansion.md†L18-L133】【F:coreTheories/symbolicAttractorExpansion.md†L248-L335】
- **Systems biology framing:** Nonlinear neuro-immune loops, stochasticity, time delays, and hysteresis (including persistent immune triggers) are central to the long-form theory and should appear in simulation design. 【F:coreTheories/systemsBiologyModel.md†L36-L155】
- **Genetic + systems integration:** RCCX/polygenic vulnerabilities and interlocked subsystems (structural, immune, autonomic, metabolic) shape model parameters and coupling strengths. 【F:coreTheories/integrativeSystemsModelCMSS.md†L1-L143】【F:coreTheories/integrativeSystemsModelCMSS.md†L294-L372】

## Simulation approaches and ideas

### 1) Minimal ODE prototyping

Start with the minimal 3–4 variable system plus Λ/Ω, mirroring the refined GASLIT-AF draft. Use this to validate basic attractor formation, phase transitions, and hysteresis with simple stress-input curves.

- **Variables:** Autonomic capacity (x), immune activation (y), fatigue (f), stress load Λ(t), and optionally Ω(t).
- **Targets:** Recover a stable healthy attractor, a collapsed attractor, and threshold behavior as Λ or γ increase.
- **Outputs:** Phase portraits, time-series trajectories, basin boundary maps, and sensitivity to γ and Ω. 【F:coreTheories/gaslitDynamicalRefined.md†L93-L176】

### 2) Multi-loop symbolic system

Use the symbolic attractor expansion to simulate a richer physiological network (cortisol, inflammation, autonomic activation, energy, oxidative stress). This captures the endocrine–immune–metabolic–autonomic loops and yields bistability and bifurcation conditions.

- **Variables:** X (cortisol), I (inflammation), A (autonomic), E (energy), O (oxidative stress).
- **Features:** Hill-function nonlinearities, feedback loops, and multi-attractor basins.
- **Analyses:** Fixed-point analysis, Jacobian stability, saddle-node bifurcation detection, and basin mapping. 【F:coreTheories/symbolicAttractorExpansion.md†L18-L133】【F:coreTheories/symbolicAttractorExpansion.md†L660-L741】

### 3) Delays, noise, and hysteresis

Introduce delayed feedback and stochastic terms to reflect the recursive and noisy nature of the neuro-immune loop. The systems biology model emphasizes time delays and noise sources that can generate oscillations, instability, and path dependence.

- **Add-ons:** Stochastic terms (σ ξ(t)), delayed coupling (Θ), persistent immune drive (u(t)).
- **Goal:** Produce realistic flare dynamics, delayed crashes, and irreversible transitions. 【F:coreTheories/systemsBiologyModel.md†L36-L155】

### 4) Parameter sweeps + bifurcation analysis

Systematically explore parameter space for γ, Λ, Ω, and coupling coefficients to map stability regimes. Start with brute-force sweeps; later move to continuation methods.

- **MVP path:** Grid sweeps and simple bifurcation diagrams, as outlined in the MVP plan. 【F:simulationIdeas/gaslitHPCMVP.md†L10-L48】
- **Advanced path:** Pseudo-arclength continuation and Lyapunov exponent calculations. 【F:simulationIdeas/gaslitHPCMVP.md†L83-L110】【F:simulationIdeas/gaslitHPC.md†L269-L338】

### 5) Personalized attractor tracking

Model individual trajectories by ingesting wearable signals and symptom journals as state-space coordinates. Fit model parameters per individual and identify threshold surfaces that predict collapse.

- **Inputs:** HRV, temperature, sleep metrics, symptom logs.
- **Outputs:** Personalized phase portraits, early-warning thresholds, and intervention simulations. 【F:simulationIdeas/personalizedProtocol.md†L5-L69】

### 6) HPC + GPU-accelerated pipelines

A full-scale validation pipeline integrates simulation, bifurcation analysis, and ML-based surrogate modeling on GPU clusters. This allows large parameter sweeps, high-dimensional data integration, and fast inference.

- **Architecture:** Simulation engine + ML analysis + API services on Kubernetes, GPU-accelerated solvers, and multi-omics ingestion. 【F:simulationIdeas/gaslitHPC.md†L6-L85】【F:simulationIdeas/gaslitHPC.md†L157-L239】
- **Surrogates:** Train neural or GP models to approximate simulation outputs for fast parameter exploration. 【F:simulationIdeas/gaslitHPC.md†L339-L366】
- **MVP roadmap:** Stand up minimal ODE sims, run GPU-parallel sweeps, and add notebooks. 【F:simulationIdeas/gaslitHPCMVP.md†L1-L48】

## Suggested next steps (sequenced)

1. **Implement the minimal ODE system** (x, y, f, Λ, Ω) and validate attractor transitions with simple stress signals. 【F:coreTheories/gaslitDynamicalRefined.md†L93-L176】
2. **Expand to the symbolic multi-loop system** and run stability/bifurcation analyses on key parameter sets. 【F:coreTheories/symbolicAttractorExpansion.md†L18-L133】【F:coreTheories/symbolicAttractorExpansion.md†L660-L741】
3. **Add noise and delay terms** to capture relapse dynamics and hysteresis. 【F:coreTheories/systemsBiologyModel.md†L36-L155】
4. **Scale parameter sweeps** using GPU or distributed compute, and train surrogate models for faster iteration. 【F:simulationIdeas/gaslitHPC.md†L269-L366】
5. **Fit to personalized data** once real-world time-series are available, using the personalized attractor protocol as a template. 【F:simulationIdeas/personalizedProtocol.md†L5-L69】

## Related documents

- **Core theories:** `coreTheories/` (refined dynamical model, long-form dynamical systems paper, symbolic attractor equations, integrative RCCX + systems analysis).
- **Simulation ideas:** `simulationIdeas/gaslitHPCMVP.md`, `simulationIdeas/gaslitHPC.md`, `simulationIdeas/personalizedProtocol.md`.

