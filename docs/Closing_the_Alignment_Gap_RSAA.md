# Closing the Alignment Gap: The Resonant State Alignment Algorithm (RSAA)

Gianne P. Bacay [ARCANE Research. Email: giannebacay2004@gmail.com]

> **ARCANE 5.0 revision.** This Markdown document is the maintained System 1 specification. The bundled PDF is the historical paper and predates ARC 1's formal RSAA integration.

## Abstract

The **Resonant State Alignment Algorithm (RSAA)** is ARCANE's formulation for aligning hierarchical state before a System 1 decision is committed. Higher levels project per-example targets downward, lower states move toward those targets, and the current input is evaluated again. The core inference path keeps learned weights fixed; a separate objective-driven adapter provides bounded weight updates, rollback, and reset semantics. The result is a controlled interface for measuring state divergence, adaptation cost, and task quality independently.

## What ARCANE implements

ARCANE 5.0 provides two concrete RSAA paths. General ARCANE models use per-example resonance state and differentiable feedback projections through `predict_with_resonance`. ARC 1 now performs state-only RSAA inside `Arc1Model.perceive`: equal-width perception blocks use identity feedback projections, bounded cycles, per-example convergence, and current-input re-evaluation. The legacy training callback stages a previous-batch mean prototype and serves a different role.

### Relationship to ARC 1

Earlier ARC 1 releases were **resonance-inspired, not an implementation of this algorithm**. `FieldResonance` mixed token fields and `ResonantBinding` iteratively settled schema probes, but neither projected a higher perception-layer state to the layer below, measured global hierarchy divergence, or stopped per example at an RSAA threshold.

ARCANE 5.0 adds those missing operations to ARC 1. New `arc1-tiny` and `arc1` presets expose up to two RSAA cycles, while `active_rsaa_cycles` selects the default. The current checkpoint keeps zero as its production default because the same-checkpoint RSAA ablation did not pass the replacement gate. Configurations saved before 5.0 omit `rsaa_cycles`; they load with zero cycles so their tensor layout and inference behavior remain compatible. RSAA changes activations only and adds no trainable parameters.

---

## 1. Introduction

In a hierarchical computational system $\mathcal{H}$ with $L$ layers, each layer $i \in \{1, \dots, L\}$ maintains an internal state representation $S_i \in \mathbb{R}^{d_i}$. A single-pass inference path maps inputs to outputs without revisiting lower state after higher context becomes available. RSAA addresses this **Alignment Gap** by making top-down constraints part of an explicit, measurable inference loop.

The textbook approach to neural learning, backpropagation, combined with advanced optimizers such as Adam or SGD with momentum, solves the credit assignment problem by propagating a global error signal backward through the network. However, this approach is:

1. **Retrospective**: It identifies errors *after* an output is generated.
2. **Global**: It requires a complete forward pass before any correction can occur.
3. **Biologically Implausible**: The brain does not utilize such global error signals.

For biological neural systems, the neocortex is organized into a hierarchy where every feed-forward connection is matched by a feedback connection. Predictive Coding theory suggests that the brain constantly generates top-down predictions about what lower-level inputs *should* be, and learning occurs when these predictions fail to match reality.

### 1.1 Main Results

This note presents RSAA as a formal state-smoothing algorithm and documents how the restricted update relates to ARCANE's implementation.

**Theorem 1.1.** *There exists a deterministic algorithm that takes $O(N \cdot L)$ time to achieve Prospective Configuration in a hierarchy $\mathcal{H}$ of $L$ layers, where $N$ is the number of resonance cycles. The algorithm operates solely on internal state activations $\{S_1, \dots, S_L\}$ without modifying synaptic weights $W$.*

The theorem counts projection and harmonization steps abstractly. It does not establish applicability or usefulness in domains beyond the implemented neural-network hierarchy.

### 1.2 Technical Overview

Broadly speaking, there are two traditional paradigms for neural computation:

- **Feed-Forward Inference**: Information flows from input to output in a single pass.
- **Backpropagation Learning**: A global error signal propagates backward to adjust weights.

My approach merges these two paradigms through a **Resonance Loop** technique. At any point during the execution of RSAA, each layer maintains a "state" $S_i$ that represents its current interpretation of the input. A higher layer $i$ can "project" its expectation $P_{i \to i-1}$ to the layer below, and the lower layer $i-1$ can "harmonize" its state to reduce the divergence $\Delta_{i-1} = S_{i-1} - P_{i \to i-1}$.

Iterating this projection-harmonization cycle reduces divergence when projections are fixed. In the full model, projections and recurrent states can change after each forward evaluation, so reaching an equilibrium is an empirical property rather than a theorem.

The central engineering distinction is between **state alignment** and **weight modification**. RSAA refines states with fixed learned weights. ARCANE's separate test-time adapter can optionally modify selected variables against an explicit objective. Keeping the paths separate makes state divergence, objective change, weight drift, and forgetting measurable independently; it does not by itself prevent gradient pathologies or establish cognitive or biological equivalence.

---

## 2. Preliminaries

**Definition 2.1 (Resonant Hierarchy).** A Resonant Hierarchy $\mathcal{H}$ is an ordered set of $L$ layers, where each layer $i$ maintains:
- An internal state $S_i \in \mathbb{R}^{d_i}$.
- A projection function $f_{proj}^{(i)}: \mathbb{R}^{d_i} \to \mathbb{R}^{d_{i-1}}$.
- A harmonization rate $\gamma_i \in (0, 1]$.

**Definition 2.2 (Feedback Projection).** For a layer $i$, the Feedback Projection to layer $i-1$ is:

$$ P_{i \to i-1} = f_{proj}^{(i)}(S_i; W_{i, proj}) $$

In the ARCANE implementation, hidden-space feedback uses a trainable projection kernel; an additional feedback matrix can reconstruct the cell input space:

$$ P_{i \to i-1} = S_i \cdot W_{proj} $$

ARC 1 is a useful special case because every perception block has width $D$. It therefore uses the parameter-free identity projection

$$ f_{proj}^{(i)}(S_i) = S_i $$

which preserves old checkpoint tensor layouts. This is an architectural choice, not a claim that identity feedback is optimal.

**Definition 2.3 (Prediction Divergence).** The Prediction Divergence at layer $i-1$ is the signed difference between its current state and the expectation projected from above:

$$ \Delta_{i-1} = S_{i-1} - P_{i \to i-1} $$

**Definition 2.4 (Global Divergence).** The Global Divergence $\mathcal{D}$ of a hierarchy $\mathcal{H}$ is the sum of squared local divergences:

$$ \mathcal{D} = \sum_{i=1}^{L-1} \| \Delta_i \|^2 $$

**Definition 2.5 (Resonance).** A hierarchy $\mathcal{H}$ is said to be in Resonance when $\mathcal{D} < \epsilon$ for some threshold $\epsilon > 0$.

---

## 3. The RSAA Algorithm

### 3.1 State Harmonization

The core operation of RSAA is the State Harmonization update rule.

**Lemma 3.1 (Harmonization Update).** *Given a layer $i-1$ with state $S_{i-1}^{(t)}$ and an incoming projection $P_{i \to i-1}$, the harmonized state at cycle $t+1$ is:*

$$ S_{i-1}^{(t+1)} = S_{i-1}^{(t)} - \gamma \cdot \Delta_{i-1}^{(t)} $$

*where $\gamma \in (0, 1]$ is the Resonance Factor.*

*Proof.* The update directly minimizes the local divergence $\| \Delta_{i-1} \|^2$ via gradient descent on the state variable $S_{i-1}$. Since $\Delta_{i-1} = S_{i-1} - P_{i \to i-1}$, the gradient with respect to $S_{i-1}$ is $\nabla_{S_{i-1}} \| \Delta_{i-1} \|^2 = 2\Delta_{i-1}$. A gradient descent step with learning rate $\gamma/2$ yields the update rule. $\square$

### 3.2 Convergence

**Theorem 3.2 (Convergence of RSAA).** *For a Resonant Hierarchy $\mathcal{H}$ with fixed projections $\{P_{i \to i-1}\}$, the RSAA update rule converges to $\mathcal{D} = 0$ as $N \to \infty$, provided $\gamma \in (0, 1]$.*

*Proof.* At each cycle, the local divergence $\| \Delta_{i-1}^{(t+1)} \|^2 = \| (1-\gamma) \Delta_{i-1}^{(t)} \|^2 = (1-\gamma)^2 \| \Delta_{i-1}^{(t)} \|^2$. Since $(1-\gamma)^2 < 1$ for $\gamma \in (0, 1]$, the divergence decreases geometrically. Summing over all layers, $\mathcal{D}^{(t+1)} \leq (1-\gamma)^2 \mathcal{D}^{(t)}$, which converges to $0$. $\square$

### 3.3 Algorithmic Flow

**Algorithm 1: Resonant State Alignment**

```
Input: Hierarchy H, Input x, Cycles N, Threshold epsilon
Output: Aligned states {S_1, ..., S_L}

1.  [Forward Initialization]
    Perform feed-forward pass to populate {S_1, ..., S_L}.

2.  [Resonance Loop]
    for t = 1 to N do:
        // Step A: Project (Top-Down)
        for i = L down to 2 do:
            P_{i -> i-1} = f_proj(S_i; W_i)
        end for
        
        // Step B: Harmonize (Bottom-Up)
        for i = 1 to L-1 do:
            Delta_i = S_i - P_{i+1 -> i}
            S_i = S_i - gamma * Delta_i
        end for
        
        // Step C: Check Convergence
        D = sum of ||Delta_i||^2
        if D < epsilon then break
    end for

3.  [Final Inference]
    Return output Y = f_output(S_L)
```

### 3.4 ARC 1 execution

The paper's state update alone changes lower states but leaves an already-computed top state unchanged. ARC 1 therefore re-evaluates the current input after each harmonization step, feeding aligned lower state into the next perception block. This makes the final field and its grounded readouts depend on alignment while keeping the learned weights fixed.

ARC 1 exposes three per-example measurements: `rsaa_initial_divergence`, `rsaa_divergence`, and `rsaa_steps`. Its operational divergence is the sum of adjacent-layer mean squared errors over real (non-padding) token coordinates, a length- and width-normalized version of $\mathcal{D}$. `rsaa_epsilon` stops updates independently for examples that have converged; `rsaa_factor` is $\gamma$; and `rsaa_cycles` is the hard latency bound. Since re-evaluation changes the states used as projections, Theorem 3.2's fixed-projection guarantee does not apply to the full ARC 1 loop. Its convergence and task benefit must be measured empirically.

### 3.5 Existing-checkpoint ablation

The ARCANE 5.0 rollout evaluated the same published `arc1-tiny` weights at zero, one, and two RSAA cycles. Each profile received its own held-out temperature calibration. The macro quality score averages held-out and unseen tool selection, exact calls, extraction F1, and held-out and unseen classification.

| RSAA cycles | Macro quality | Macro median latency | Result |
|---:|---:|---:|---|
| 0 | 72.33% | 5.36 ms | Production default |
| 1 | 71.03% | 6.86 ms | Rejected |
| 2 | 71.13% | 8.92 ms | Rejected |

The aligned profiles slightly improved held-out exact calls and extraction F1, but lost 3.33 percentage points on both unseen-tool selection and unseen exact calls. Neither passed the predeclared replacement gate, so ARCANE exposes RSAA for experimentation without replacing the validated zero-cycle checkpoint. The complete machine-readable result is in [`Models/arc1_rsaa_comparison.json`](../Models/arc1_rsaa_comparison.json).

---

## 4. Substrate Agnosticism

The update can be written for any compatible vector state and projection. Transfer to another computational substrate would require a separate implementation and evaluation.

**Corollary 4.1.** *RSAA can be instantiated in any system $\mathcal{S}$ that satisfies:*
1. *$\mathcal{S}$ maintains adjustable internal states $\{S_i\}$.*
2. *$\mathcal{S}$ supports a projection operation between state levels.*
3. *$\mathcal{S}$ supports an additive update operation on states.*

Possible research directions include:
- **Quantum Computing**: Variational state stabilization in VQE/QAOA circuits.
- **Cybernetics**: Maintaining equilibrium in feedback-rich physical systems.
- **Multi-Agent Systems**: Achieving collective agreement through decentralized resonance.
- **Symbolic AI**: Enforcing semantic consistency between logical representations.

---

## 5. Conclusion

RSAA gives ARCANE System 1 models a concrete state-alignment rule and a measurable divergence signal. Per-example targets prevent co-batched requests from sharing alignment state; feedback maps carry top-down context; and same-checkpoint ablations expose the accuracy and latency tradeoff for each task. ARC 1's identity-projection path is state-only, bounded, and checkpoint-compatible; usefulness is an empirical question rather than a consequence of the fixed-projection theorem.

---

## References

- [ART87] Carpenter, G. A., & Grossberg, S. (1987). A massively parallel architecture for a self-organizing neural pattern recognition machine. *Computer Vision, Graphics, and Image Processing*, 37(1), 54-115.
- [PC99] Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79-87.
- [FEP10] Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
- [PC21] Millidge, B., Tschantz, A., & Buckley, C. L. (2021). Predictive Coding Approximates Backprop along Arbitrary Computation Graphs. *Neural Computation*, 34(6), 1329-1368.
- [HEB49] Hebb, D. O. (1949). *The Organization of Behavior*. Wiley.
- [BP86] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
