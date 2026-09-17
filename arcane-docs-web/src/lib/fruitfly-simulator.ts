/**
 * Biophysical Drosophila Connectome Simulator integrated with ARCANE's core architectural layers:
 * 1. Resonant Spiking Dynamics (ARCANE `resonant_spike` & `DenseGSER` / `ResonantGSER`)
 * 2. Homeostatic Plasticity & Synaptic Scaling (ARCANE `BioplasticDenseLayer` & `homeostatic_gelu`)
 * 3. Hebbian Co-activation & Activity History Traces (ARCANE `HebbianHomeostaticNeuroplasticity`)
 * 4. Neuromimetic Activations (ARCANE `adaptive_softplus` & `homeostatic_gelu` gain modulation)
 * 5. Prospective Configuration via Resonant State Alignment (RSAA top-down projection)
 *
 * Connectome topology: Adult Drosophila whole-brain FAFB v783 mapped and open-sourced by
 * the Google DeepMind Connectomics team and consortium partners.
 */

export interface BiophysicalNeuron {
  id: string;
  label: string;
  cell_type: string;
  layer: string;
  side: string;
  x: number;
  y: number;
  z: number;
  rfX: number;
  rfY: number;
}

export interface BiophysicalEdge {
  pre: string;
  post: string;
  synapses: number;
  sign: number; // +1 excitatory (acetylcholine), -1 inhibitory (GABA/glutamate)
}

interface PostSynapse {
  postIndex: number;
  baseWeight: number; // Biological synapse count from EM connectome
  plasticWeight: number; // ARCANE Bioplastic online adaptive component
  sign: number;
}

export class FruitflyCircuitSimulator {
  readonly neurons: BiophysicalNeuron[];
  readonly neuronIndex: Map<string, number>;
  readonly postSynapses: PostSynapse[][];

  // Biophysical state vectors
  private v: Float64Array; // Membrane potential (0.0 resting, 1.0 base threshold)
  private gE: Float64Array; // Excitatory synaptic conductance (cholinergic)
  private gI: Float64Array; // Inhibitory synaptic conductance (GABAergic)
  private refractory: Float64Array; // Refractory timer in seconds
  private sensoryDrive: Float64Array; // Depolarizing sensory current from drawing
  public calcium: Float64Array; // Slow calcium / GCaMP activity [0, 1]
  public lastSpikeTime: Float64Array; // Seconds timestamp of most recent spike

  // ARCANE Homeostatic Plasticity state (from ARCANE BioplasticDenseLayer & homeostatic_gelu)
  private activityHistory: Float64Array; // Running BCM-style activity trace (a_bar)
  private homeostaticThreshold: Float64Array; // Self-regulating firing threshold theta_i
  private homeostaticGain: Float64Array; // h-GELU dynamic sensitivity gain factor

  // ARCANE Resonant Modulation (from ARCANE ResonantGSER & RSAA prospective configuration)
  private resonanceModulation: Float64Array; // Top-down resonance alignment factor
  private classDrive: Float64Array; // Descending readout current from class evidence

  // Spikes tracking for realistic firing rate (Hz)
  private spikeTimestamps: number[] = [];
  private totalSpikes = 0;
  private firingRateHz = 0;
  private activeCount = 0;
  private meanActivity = 0;
  private meanGain = 1.0;
  private resonanceDivergence = 0;

  // Biophysical & ARCANE parameters
  private readonly tauM = 0.022; // 22 ms membrane time constant
  private readonly vReset = 0.0;
  private readonly baseThreshold = 1.0;
  private readonly tauRef = 0.003; // 3 ms absolute refractory period
  private readonly tauE = 0.012; // 12 ms nicotinic ACh decay
  private readonly tauI = 0.024; // 24 ms GABAergic decay
  private readonly tauCa = 0.22; // 220 ms calcium / fluorescence decay

  // ARCANE Homeostatic parameters (calibrated to ARCANE BioplasticDenseLayer defaults)
  private readonly targetActivity = 0.12; // Target average firing activity
  private readonly homeostaticRate = 0.0025; // Threshold adaptation rate
  private readonly bcmTau = 1.8; // BCM running average time constant (seconds)
  private readonly hebbianRate = 0.0008; // Hebbian synaptic plasticity rate
  private readonly synapticHomeoRate = 0.0004; // Synaptic weight decay scaling

  constructor(
    rawNeurons: Array<{ id: string; label: string; cell_type: string; layer: string; side: string; x?: number; y?: number; z?: number }>,
    rawEdges: BiophysicalEdge[]
  ) {
    const n = rawNeurons.length;
    this.neuronIndex = new Map();
    this.postSynapses = Array.from({ length: n }, () => []);

    // Filter sensory neurons and establish spatial receptive fields
    const leftSensory: Array<{ neuron: (typeof rawNeurons)[0]; origIdx: number }> = [];
    const rightSensory: Array<{ neuron: (typeof rawNeurons)[0]; origIdx: number }> = [];

    rawNeurons.forEach((nr, idx) => {
      this.neuronIndex.set(nr.id, idx);
      const isSensory = nr.layer === "sensory" || nr.cell_type === "LC4" || nr.cell_type === "LPLC2";
      if (isSensory) {
        if (nr.side === "left") leftSensory.push({ neuron: nr, origIdx: idx });
        else rightSensory.push({ neuron: nr, origIdx: idx });
      }
    });

    // Sort sensory neurons by dorso-ventral / lateral position to map retinotopic grid
    leftSensory.sort((a, b) => (a.neuron.y ?? 0) - (b.neuron.y ?? 0));
    rightSensory.sort((a, b) => (a.neuron.y ?? 0) - (b.neuron.y ?? 0));

    const rfMap = new Map<string, { rfX: number; rfY: number }>();

    // Map left eye to left half of whiteboard (X: 3..12, Y: 3..24)
    leftSensory.forEach(({ neuron }, i) => {
      const frac = leftSensory.length > 1 ? i / (leftSensory.length - 1) : 0.5;
      const rfY = 3.5 + frac * 20.5;
      const rfX = 3.0 + (i % 3) * 4.2;
      rfMap.set(neuron.id, { rfX, rfY });
    });

    // Map right eye to right half of whiteboard (X: 15..24, Y: 3..24)
    rightSensory.forEach(({ neuron }, i) => {
      const frac = rightSensory.length > 1 ? i / (rightSensory.length - 1) : 0.5;
      const rfY = 3.5 + frac * 20.5;
      const rfX = 15.5 + (i % 3) * 4.2;
      rfMap.set(neuron.id, { rfX, rfY });
    });

    this.neurons = rawNeurons.map((nr) => {
      const rf = rfMap.get(nr.id) ?? { rfX: 14, rfY: 14 };
      return {
        id: nr.id,
        label: nr.label,
        cell_type: nr.cell_type,
        layer: nr.layer,
        side: nr.side,
        x: nr.x ?? 0,
        y: nr.y ?? 0,
        z: nr.z ?? 0,
        rfX: rf.rfX,
        rfY: rf.rfY,
      };
    });

    // Build synaptic adjacency graph with actual connectome edge counts & signs
    for (const edge of rawEdges) {
      const preIdx = this.neuronIndex.get(edge.pre);
      const postIdx = this.neuronIndex.get(edge.post);
      if (preIdx !== undefined && postIdx !== undefined) {
        this.postSynapses[preIdx].push({
          postIndex: postIdx,
          baseWeight: Math.abs(edge.synapses),
          plasticWeight: 0.0,
          sign: edge.sign >= 0 ? 1 : -1,
        });
      }
    }

    // Allocate biophysical arrays
    this.v = new Float64Array(n);
    this.gE = new Float64Array(n);
    this.gI = new Float64Array(n);
    this.refractory = new Float64Array(n);
    this.sensoryDrive = new Float64Array(n);
    this.calcium = new Float64Array(n);
    this.lastSpikeTime = new Float64Array(n).fill(-100);

    // Allocate ARCANE homeostatic & resonant vectors
    this.activityHistory = new Float64Array(n);
    this.homeostaticThreshold = new Float64Array(n).fill(this.baseThreshold);
    this.homeostaticGain = new Float64Array(n).fill(1.0);
    this.resonanceModulation = new Float64Array(n);
    this.classDrive = new Float64Array(n);
  }

  /**
   * Translates 28x28 whiteboard pixels into sensory currents on visual projection neurons (LC4, LPLC2).
   * Modulated by ARCANE's homeostatic gain and adaptive softplus rate-coding.
   */
  public updateSensoryInput(grayPixels: number[] | Float32Array, isDrawing: boolean): void {
    let totalInk = 0;
    for (let i = 0; i < 784; i++) {
      const ink = 1.0 - grayPixels[i] / 255.0;
      if (ink > 0.04) totalInk += ink;
    }

    // If whiteboard has no ink, shut off sensory currents completely
    if (totalInk < 5.0) {
      this.sensoryDrive.fill(0);
      return;
    }

    const n = this.neurons.length;
    for (let i = 0; i < n; i++) {
      const nr = this.neurons[i];
      const isSensory = nr.layer === "sensory" || nr.cell_type === "LC4" || nr.cell_type === "LPLC2";
      if (!isSensory) {
        this.sensoryDrive[i] = 0;
        continue;
      }

      // 2D Gaussian receptive field integration over the 28x28 retina
      let drive = 0;
      const minX = Math.max(0, Math.floor(nr.rfX - 5));
      const maxX = Math.min(27, Math.ceil(nr.rfX + 5));
      const minY = Math.max(0, Math.floor(nr.rfY - 5));
      const maxY = Math.min(27, Math.ceil(nr.rfY + 5));

      for (let py = minY; py <= maxY; py++) {
        for (let px = minX; px <= maxX; px++) {
          const ink = 1.0 - grayPixels[py * 28 + px] / 255.0;
          if (ink > 0.05) {
            const dx = px - nr.rfX;
            const dy = py - nr.rfY;
            const w = Math.exp(-(dx * dx + dy * dy) / (2 * 7.5));
            drive += ink * w;
          }
        }
      }

      // ARCANE Neuromimetic Activation: adaptive_softplus rate curve
      // softplus(sharpness * (x - threshold)) / sharpness
      const rawCurrent = 0.8 + drive * 0.075 + (isDrawing ? 0.45 + drive * 0.05 : 0);
      const sharpness = 1.2;
      const softplusDrive = Math.log(1 + Math.exp(sharpness * (rawCurrent - 0.2))) / sharpness;

      // Scaled by ARCANE homeostatic gain (homeostatic_gelu principle)
      this.sensoryDrive[i] = Math.min(3.2, softplusDrive * this.homeostaticGain[i]);
    }
  }

  /**
   * Applies ARCANE RSAA class evidence to descending readout cells.
   * Competing digits drive different descending neurons, so an ambiguous 1/4
   * lights both pathways instead of collapsing to a single hardcoded label.
   */
  public updateResonanceFeedback(probabilities: number[] | null): void {
    this.classDrive.fill(0);
    this.resonanceModulation.fill(0);
    if (!probabilities || probabilities.length < 10) {
      this.resonanceDivergence = 0;
      return;
    }

    const descendingIdx: number[] = [];
    for (let i = 0; i < this.neurons.length; i++) {
      const nr = this.neurons[i];
      if (nr.layer === "descending" || nr.cell_type === "DNp01") descendingIdx.push(i);
    }

    let sumDiv = 0;
    descendingIdx.forEach((i, k) => {
      const digit = k % 10;
      const p = probabilities[digit] ?? 0;
      this.classDrive[i] = p * 1.15;
      this.resonanceModulation[i] = p * 0.35;
      sumDiv += Math.abs(p - this.calcium[i]);
    });

    const peak = Math.max(...probabilities);
    for (let i = 0; i < this.neurons.length; i++) {
      if (this.neurons[i].layer === "sensory") {
        this.resonanceModulation[i] = peak * 0.18;
      }
    }

    this.resonanceDivergence = descendingIdx.length ? sumDiv / descendingIdx.length : 0;
  }

  /**
   * Advances the biophysical differential equations by dt seconds using numerical sub-stepping.
   * Incorporates:
   * 1. ARCANE `resonant_spike` leaky integration + top-down resonance amplification
   * 2. ARCANE `homeostatic_gelu` dynamic gain & threshold self-regulation
   * 3. ARCANE `BioplasticDenseLayer` inference-time Hebbian synaptic plasticity
   */
  public step(dt: number, currentTime: number): void {
    const clampedDt = Math.min(0.05, Math.max(0.001, dt));
    const h = 0.001; // 1 ms numerical integration step
    const subSteps = Math.max(1, Math.round(clampedDt / h));
    const dtSub = clampedDt / subSteps;

    const n = this.neurons.length;
    const decayE = Math.exp(-dtSub / this.tauE);
    const decayI = Math.exp(-dtSub / this.tauI);
    const decayCa = Math.exp(-dtSub / this.tauCa);
    const bcmDecay = Math.exp(-dtSub / this.bcmTau);

    for (let s = 0; s < subSteps; s++) {
      const stepTime = currentTime - clampedDt + s * dtSub;

      for (let i = 0; i < n; i++) {
        // Conductance & calcium decay
        this.gE[i] *= decayE;
        this.gI[i] *= decayI;
        this.calcium[i] *= decayCa;

        // BCM activity history decay
        this.activityHistory[i] = this.activityHistory[i] * bcmDecay + (1 - bcmDecay) * this.calcium[i];

        // 1. ARCANE Homeostatic Plasticity (self-regulating threshold and gain)
        // If neuron is over-active, threshold rises to avoid hyper-excitation. If quiet, sensitizes.
        const deltaTheta = this.homeostaticRate * (this.activityHistory[i] - this.targetActivity);
        this.homeostaticThreshold[i] = Math.max(0.75, Math.min(1.45, this.homeostaticThreshold[i] + deltaTheta * dtSub));

        // Homeostatic gain (homeostatic_gelu formula: gain = 1 + rate * (target - history))
        const rawGain = 1.0 + 0.8 * (this.targetActivity - this.activityHistory[i]);
        this.homeostaticGain[i] = Math.max(0.25, Math.min(1.8, rawGain));

        // Refractory period handling
        if (this.refractory[i] > 0) {
          this.refractory[i] -= dtSub;
          this.v[i] = this.vReset;
          continue;
        }

        // 2. Leaky Integrate-and-Fire with ARCANE Resonant Modulation
        // ARCANE: integrated_potential = x + state * (1 - leak)
        // modulated_potential = integrated_potential * (1 + resonance_factor)
        const synCurrent = this.gE[i] - this.gI[i];
        const totalCurrent = this.sensoryDrive[i] + synCurrent + this.classDrive[i];
        const dV = ((-this.v[i] + totalCurrent) / this.tauM) * dtSub;
        this.v[i] = Math.max(0, this.v[i] + dV);

        // Modulate potential via ARCANE resonance factor
        const modulatedV = this.v[i] * (1.0 + this.resonanceModulation[i]);

        // 3. Spiking Logic: Fire if modulated potential exceeds adaptive homeostatic threshold
        if (modulatedV >= this.homeostaticThreshold[i]) {
          this.v[i] = this.vReset;
          this.refractory[i] = this.tauRef;
          this.calcium[i] = Math.min(1.0, this.calcium[i] + 0.65);
          this.lastSpikeTime[i] = stepTime;
          this.spikeTimestamps.push(stepTime);
          this.totalSpikes++;

          // 4. Synaptic transmission + ARCANE Bioplastic Hebbian Learning
          const targets = this.postSynapses[i];
          for (let k = 0; k < targets.length; k++) {
            const syn = targets[k];
            const postIdx = syn.postIndex;

            // Effective weight = base EM connectome count + bioplastic learned component
            const effectiveWeight = Math.max(1.0, syn.baseWeight + syn.plasticWeight);
            const deltaG = (effectiveWeight / 10.0) * 0.52;

            if (syn.sign > 0) {
              this.gE[postIdx] += deltaG;
            } else {
              this.gI[postIdx] += deltaG;
            }

            // ARCANE Inference-Time Plasticity (Hebbian co-activation + homeostatic decay)
            // dW = learning_rate * (pre * post) - homeostatic_rate * plastic_kernel
            const postAct = this.calcium[postIdx];
            if (postAct > 0.15) {
              const dPlastic = this.hebbianRate * postAct - this.synapticHomeoRate * syn.plasticWeight;
              syn.plasticWeight = Math.max(-syn.baseWeight * 0.5, Math.min(syn.baseWeight * 1.5, syn.plasticWeight + dPlastic));
            }
          }
        }
      }
    }

    // Clean up spike history older than 0.5s to measure current firing rate
    const cutoff = currentTime - 0.5;
    while (this.spikeTimestamps.length > 0 && this.spikeTimestamps[0] < cutoff) {
      this.spikeTimestamps.shift();
    }
    this.firingRateHz = Math.round((this.spikeTimestamps.length / 0.5) * 10) / 10;

    // Calculate active cells, mean activity, and mean homeostatic gain
    let active = 0;
    let sumCa = 0;
    let sumGain = 0;
    for (let i = 0; i < n; i++) {
      if (this.calcium[i] > 0.08 || currentTime - this.lastSpikeTime[i] < 0.25) {
        active++;
      }
      sumCa += this.calcium[i];
      sumGain += this.homeostaticGain[i];
    }
    this.activeCount = active;
    this.meanActivity = sumCa / n;
    this.meanGain = sumGain / n;
  }

  /**
   * Resets all membrane potentials, conductances, and plastic traces to resting baseline.
   */
  public reset(): void {
    this.v.fill(0);
    this.gE.fill(0);
    this.gI.fill(0);
    this.refractory.fill(0);
    this.sensoryDrive.fill(0);
    this.calcium.fill(0);
    this.lastSpikeTime.fill(-100);
    this.activityHistory.fill(0);
    this.homeostaticThreshold.fill(this.baseThreshold);
    this.homeostaticGain.fill(1.0);
    this.resonanceModulation.fill(0);
    this.classDrive.fill(0);
    this.spikeTimestamps = [];
    this.firingRateHz = 0;
    this.activeCount = 0;
    this.meanActivity = 0;
    this.meanGain = 1.0;
    this.resonanceDivergence = 0;

    // Reset plastic component of synapses to zero
    for (let i = 0; i < this.postSynapses.length; i++) {
      const list = this.postSynapses[i];
      for (let k = 0; k < list.length; k++) {
        list[k].plasticWeight = 0.0;
      }
    }
  }

  public getStats() {
    return {
      firingRateHz: this.firingRateHz,
      activeCount: this.activeCount,
      meanActivity: this.meanActivity,
      meanGain: Math.round(this.meanGain * 100) / 100,
      divergence: Math.round(this.resonanceDivergence * 100) / 100,
      totalSpikes: this.totalSpikes,
    };
  }

  /**
   * Packs live state for the given neurons into a vec3-per-neuron array for the
   * shader: (calcium, seconds since last spike, homeostatic gain).
   *
   * This is the fast path used with the real FlyWire tube meshes. The arbors
   * carry ~660k vertices, so the action-potential wavefront is evaluated in the
   * vertex shader from these few dozen values instead of being written per
   * vertex on the CPU every frame.
   */
  public writeNeuronState(target: number[], ids: string[], currentTime: number): void {
    for (let s = 0; s < ids.length; s++) {
      const idx = this.neuronIndex.get(ids[s]);
      const o = s * 3;
      if (idx === undefined) {
        target[o] = 0;
        target[o + 1] = 1e3;
        target[o + 2] = 1;
        continue;
      }
      target[o] = this.calcium[idx];
      target[o + 1] = currentTime - this.lastSpikeTime[idx];
      target[o + 2] = this.homeostaticGain[idx];
    }
  }

  /**
   * Updates WebGL vertex buffer with genuine biophysical action potential waves
   * propagating down each neuron's arbor, modulated by ARCANE homeostatic gain and calcium glow.
   *
   * CPU path, retained for the procedural fallback arbors. The real-geometry
   * path uses `writeNeuronState` plus the shader instead.
   */
  public applyColors(
    target: Float32Array,
    vertexNeuron: string[],
    vertexDistances: Float32Array,
    baseColors: Float32Array,
    currentTime: number
  ): void {
    const len = vertexNeuron.length;
    const waveSpeed = 3.6; // Conduction velocity along neurite arbor

    for (let i = 0; i < len; i++) {
      const nid = vertexNeuron[i];
      const idx = this.neuronIndex.get(nid);
      const o = i * 3;
      const r0 = baseColors[o];
      const g0 = baseColors[o + 1];
      const b0 = baseColors[o + 2];

      if (idx === undefined) {
        target[o] = r0 * 0.12;
        target[o + 1] = g0 * 0.12;
        target[o + 2] = b0 * 0.12;
        continue;
      }

      const ca = this.calcium[idx];
      const dtSpike = currentTime - this.lastSpikeTime[idx];
      const dist = vertexDistances[i];
      const gain = this.homeostaticGain[idx];

      // Physical action potential wavefront travelling away from soma along neurite
      let pulse = 0;
      if (dtSpike >= 0 && dtSpike < 0.35) {
        const pulsePosition = dtSpike * waveSpeed;
        const diff = dist - pulsePosition;
        pulse = Math.exp(-(diff * diff) / (2 * 0.012)) * 3.4;
      }

      // If resting (no action potential and zero calcium), stay at dim resting skeleton
      const glow = 0.11 + (ca * 0.82 + pulse) * gain;

      // When action potential pulse passes, brighten and flash towards peak luminance
      if (pulse > 0.3) {
        const flash = Math.min(1.0, pulse * 0.35 * gain);
        target[o] = Math.min(1.0, r0 * glow + flash * 0.7);
        target[o + 1] = Math.min(1.0, g0 * glow + flash * 0.85);
        target[o + 2] = Math.min(1.0, b0 * glow + flash * 1.0);
      } else {
        target[o] = Math.min(1.0, r0 * glow);
        target[o + 1] = Math.min(1.0, g0 * glow);
        target[o + 2] = Math.min(1.0, b0 * glow);
      }
    }
  }
}
