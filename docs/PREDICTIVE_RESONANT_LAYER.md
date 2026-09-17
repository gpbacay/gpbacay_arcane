# PredictiveResonantLayer in ARCANE

## Overview

The `PredictiveResonantLayer` is a specialized RNN layer within the ARCANE framework that implements **Local Predictive Resonance**, directly inspired by **Predictive Coding theory in Neuroscience**. It mimics the brain's ability to constantly generate internal predictions about future sensory inputs and adjust its neural state to minimize "prediction error" (divergence).

Unlike the hierarchical resonance seen in `ResonantGSER`, this layer models the **local homeostatic alignment** of individual neural clusters, making it fully autonomous in maintaining its own internal world model on a per-sample basis.

## Core Philosophical Difference

| Feature | ResonantGSER | PredictiveResonantLayer |
|:--- |:--- |:--- |
| **Resonance Type** | Hierarchical (Top-Down) | Local (Self-Contained) |
| **Alignment Target** | Provided by higher layer | Generated internally |
| **State Scope** | Global/Batch-wide | Per-Example |
| **Complexity** | High (Requires connectivity) | Moderate (Drop-in replacement) |

## How It Works

The `PredictiveResonantLayer` operates through a four-stage process during each forward step:

1. Base Dynamics: A standard gated recurrent update (LSTM-based) produces a raw candidate state (analogous to **Feedforward Sensory Input**).
2. Harmonization: a **closed-form** $N$-step EMA moves the candidate toward the current alignment vector (equivalent to the old iterative loop).
3. Spiking & Modulation: STE spike + subtractive reset.
4. Predictive Update: a linear head updates the alignment vector with a slow EMA ($\alpha = 0.1$).

## Process Flow

```mermaid
graph TD
    classDef default fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    classDef highlight fill:#27272a,stroke:#C785F2,color:#ffffff,stroke-width:2px
    classDef subtle fill:#09090b,stroke:#27272a,color:#a1a1aa
    
    subgraph predictive_resonance [Predictive Resonant Layer]
        direction TB
        
        Input((Input X_t)):::subtle
        State_Old["Previous State h_{t-1}"]:::subtle
        Align_Old["Alignment Memory A_{t-1}"]:::highlight
        
        %% Base Dynamics
        subgraph base_dynamics [1. Base Dynamics]
            LSTM[LSTM Cell]
            RecurrentUpdate("h_raw = LSTM(Input, State)")
        end
        
        Input --> LSTM
        State_Old --> LSTM
        LSTM --> RecurrentUpdate
        
        %% Harmonization
        subgraph harmonization [2. Harmonization Loop]
            Optimize{Minimize Divergence}:::highlight
            ResonatedState("Resonated State h_res")
        end
        
        RecurrentUpdate --> Optimize
        Align_Old -.-> Optimize
        Optimize -->|Iterate| ResonatedState
        
        %% Spiking & Modulation
        subgraph modulation [3. Spiking & Modulation]
            SpikeFunction{Spike Check}
            FinalState["Final State h_final"]:::highlight
        end
        
        ResonatedState --> SpikeFunction
        SpikeFunction -->|Threshold > 0.5| FinalState
        
        %% Predictive Update
        subgraph prediction [4. Predictive Update]
            Predictor[Predictive Head]
            NewAlign["New Alignment A_t"]:::highlight
        end
        
        FinalState --> Predictor
        Predictor -->|Project Future| NewAlign
        Align_Old -.->|Moving Average| NewAlign
        NewAlign -->|Update| Align_Old
        
        FinalState --> Output(("Output h_t")):::subtle
    end
    
    style predictive_resonance fill:#00000000,stroke:#27272a,stroke-width:1px,color:#fff
    style base_dynamics fill:#18181b,stroke:#3f3f46,color:#a1a1aa
    style harmonization fill:#18181b,stroke:#C785F2,stroke-width:1px,stroke-dasharray: 5 5,color:#fff
    style modulation fill:#18181b,stroke:#3f3f46,color:#a1a1aa
    style prediction fill:#27272a,stroke:#3f3f46,color:#e4e4e7
```

### The Alignment Vector

The Alignment Vector acts as the layer's internal expectation. It is a slow-moving target that represents where the layer "believes" its latent state should be based on perceived temporal patterns.

## Implementation Details

### Configuration

```python
from gpbacay_arcane.layers import PredictiveResonantLayer

layer = PredictiveResonantLayer(
    units=128,
    resonance_cycles=3,        # Number of harmonization iterations
    resonance_step_size=0.2,   # Alpha for state adjustment
    persist_alignment=True,    # Carry alignment across sequences
    return_sequences=True
)
```

### Parameters

- `units`: Dimensionality of the latent space.
- `resonance_cycles`: How many times the state is refined toward the alignment target per step.
- `resonance_step_size`: The magnitude of adjustment in each resonance cycle.
- `persist_alignment`: If enabled, the layer stores a "global memory" of its alignment state, allowing it to maintain semantic continuity even across independent calls or batch reshuffles.

## Mathematical Mechanism

### State Harmonization

$N$ steps of $h \leftarrow h - \eta (h - A)$ with $\eta$ clipped to $(0, 0.99)$ equal

$$h_N = (1-\eta)^N h_0 + \bigl(1-(1-\eta)^N\bigr) A$$

The cell evaluates this closed form. Spikes use a straight-through estimator; the hidden state is then $h - s\cdot\theta$.

### Predictive Update
The alignment vector $A$ is updated using an exponential moving average toward a predicted future state $P$:
$$A_{t+1} = (1 - \alpha) \cdot A_t + \alpha \cdot P(h_t)$$
Where $P$ is a learned linear transformation and $\alpha$ is a fixed slow-update rate (typically 0.1).

## Benefits

1.  **Temporal Coherence**: By harmonizing toward a moving average of its own predictions, the layer resists noise and sudden fluctuations in sequential data.
2.  **Autonomy**: It does not require complex model-wide wiring or top-down feedback loops to benefit from resonance principles.
3.  **Stateful Resonance**: With `persist_alignment`, the layer can develop a long-term "bias" or "worldview" that helps it process new information more efficiently.

## Use Cases

- **Noisy Sequence Processing**: Where temporal stability is more important than immediate reactive changes.
- **Autonomous Agents**: Where a layer needs to maintain internal consistency without external supervision.
- **Streaming Inference**: Where the model must handle continuous data streams with high temporal correlation.

---

*This documentation covers the PredictiveResonantLayer mechanism as implemented in ARCANE v3.1.0.*
