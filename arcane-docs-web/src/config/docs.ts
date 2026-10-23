export type NavItem = {
  title: string;
  href: string;
  disabled?: boolean;
  keywords?: string[];
};

export type SidebarNavItem = {
  title: string;
  items: NavItem[];
};

export const docsConfig: { sidebarNav: SidebarNavItem[] } = {
  sidebarNav: [
    {
      title: "Start Here",
      items: [
        {
          title: "Overview",
          href: "/docs",
          keywords: ["about", "purpose", "benefits", "architecture", "getting started", "overview"],
        },
        {
          title: "System 1 Architecture",
          href: "/docs/system-1",
          keywords: ["system 1", "fast decisions", "adaptive inference", "grounded", "memory", "architecture"],
        },
        {
          title: "Installation",
          href: "/docs/installation",
          keywords: ["install", "setup", "pip", "dependencies", "requirements", "tensorflow", "keras"],
        },
        {
          title: "Quick Start",
          href: "/docs/quick-start",
          keywords: ["example", "tutorial", "usage", "code", "implementation", "first model"],
        },
      ],
    },
    {
      title: "Build System 1",
      items: [
        {
          title: "ARC 1",
          href: "/docs/arc-1",
          keywords: ["automation", "tools", "tool calling", "extract", "classify", "resonant binding"],
        },
        {
          title: "Chat with ARCANE",
          href: "/docs/chat",
          keywords: ["chat", "slm", "language model", "generate", "local inference"],
        },
        {
          title: "Foundation Model",
          href: "/docs/foundation-model",
          keywords: ["hierarchical", "resonance", "feedback", "adaptation", "advanced"],
        },
        {
          title: "Neuromimetic Model",
          href: "/docs/neuromimetic-model",
          keywords: ["semantic", "sequence", "nlp", "text generation", "compact"],
        },
      ],
    },
    {
      title: "Adaptive Systems",
      items: [
        {
          title: "Neural Resonance",
          href: "/docs/neural-resonance",
          keywords: ["prospective configuration", "feedback", "bidirectional", "alignment", "synchronization"],
        },
        {
          title: "Inference-Time Adaptation",
          href: "/docs/inference-time-learning",
          keywords: ["test time adaptation", "resonance", "fast weights", "entropy", "consistency", "rollback"],
        },
        {
          title: "Homeostatic Plasticity",
          href: "/docs/homeostatic-plasticity",
          keywords: ["homeostasis", "bcm", "adaptive target", "attention guided", "hierarchical controller"],
        },
        {
          title: "Layer Reference",
          href: "/docs/layers",
          keywords: ["bioplastic", "hebbian", "homeostatic", "plasticity", "dense", "synapse", "keras"],
        },
        {
          title: "ResonantGSER",
          href: "/docs/resonant-gser",
          keywords: ["spiking", "reservoir", "dynamics", "spectral radius", "leak rate", "threshold"],
        },
        {
          title: "Predictive Resonance",
          href: "/docs/predictive-resonant-layer",
          keywords: ["predictive resonant layer", "local resonance", "recurrent", "alignment", "sequence"],
        },
        {
          title: "Activations",
          href: "/docs/activations",
          keywords: ["spike", "gelu", "softplus", "neuromimetic", "stateful", "lif"],
        },
      ],
    },
    {
      title: "Memory & Grounding",
      items: [
        {
          title: "Hippocampus",
          href: "/docs/hippocampus",
          keywords: ["episodic memory", "retrieval", "few-shot", "examples", "routing", "tool calling"],
        },
        {
          title: "Occipital",
          href: "/docs/occipital",
          keywords: ["visual word form", "letter recall", "typos", "misspellings", "visual cortex", "qr code", "video", "portable memory"],
        },
        {
          title: "Graph of Thought",
          href: "/docs/graph-of-thought",
          keywords: ["verification", "hallucination", "rag", "retrieval", "knowledge graph", "citations"],
        },
      ],
    },
    {
      title: "Reference",
      items: [
        {
          title: "CLI Commands",
          href: "/docs/cli",
          keywords: ["terminal", "version", "commands", "tools"],
        },
        {
          title: "Connectome Demo",
          href: "/docs/fruitfly",
          keywords: ["fruit fly", "drosophila", "brain", "mnist", "flywire", "connectome"],
        },
        {
          title: "Benchmarks",
          href: "/docs/benchmarks",
          keywords: ["performance", "accuracy", "loss", "comparison", "metrics", "latency"],
        },
        {
          title: "Research Paper",
          href: "/docs/research",
          keywords: ["paper", "rsaa", "academic", "theory", "math", "alignment"],
        },
        {
          title: "Blog",
          href: "/docs/blog",
          keywords: ["updates", "news", "articles", "announcements"],
        },
      ],
    },
  ],
};
