# Closing the Alignment Gap: The Resonant State Alignment Algorithm (RSAA)

Gianne P. Bacay [ARCANE Research. Email: giannebacay2004@gmail.com]

## Abstract

The **Resonant State Alignment Algorithm (RSAA)** is ARCANE's formulation for aligning hierarchical state before a System 1 decision is committed. Higher levels project per-example targets downward, lower states move toward those targets, and the current input is evaluated again. The core inference path keeps learned weights fixed; a separate objective-driven adapter provides bounded weight updates, rollback, and reset semantics. The result is a controlled interface for measuring state divergence, adaptation cost, and task quality independently.

## What ARCANE implements

ARCANE 4.1 provides per-example resonance state, differentiable feedback projections, optional objective-driven test-time adaptation, and a same-checkpoint ablation harness. The explicit current-input path is `predict_with_resonance`; the legacy training callback stages a previous-batch mean prototype and serves a different role.

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

RSAA gives ARCANE System 1 models a concrete state-alignment rule and a measurable divergence signal. Per-example targets prevent co-batched requests from sharing alignment state; trained feedback maps carry top-down context; reversible adaptation keeps changes bounded; and same-checkpoint ablations expose the accuracy and latency tradeoff for each task.

---

## References

- [ART87] Carpenter, G. A., & Grossberg, S. (1987). A massively parallel architecture for a self-organizing neural pattern recognition machine. *Computer Vision, Graphics, and Image Processing*, 37(1), 54-115.
- [PC99] Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79-87.
- [FEP10] Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
- [PC21] Millidge, B., Tschantz, A., & Buckley, C. L. (2021). Predictive Coding Approximates Backprop along Arbitrary Computation Graphs. *Neural Computation*, 34(6), 1329-1368.
- [HEB49] Hebb, D. O. (1949). *The Organization of Behavior*. Wiley.
- [BP86] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
