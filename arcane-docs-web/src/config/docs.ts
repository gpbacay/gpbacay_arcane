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
      title: "Getting Started",
      items: [
        { 
          title: "Introduction", 
          href: "/docs",
          keywords: ["about", "mission", "vision", "goals", "faq", "sentient ai", "consciousness", "developer", "philosophy", "semantic engineering", "team", "funding", "intellectual property", "breakthrough", "timeline", "layman", "societal", "environmental", "limitations", "research questions"]
        },
        { 
          title: "Installation", 
          href: "/docs/installation",
          keywords: ["install", "setup", "pip", "dependencies", "requirements", "tensorflow", "keras", "prerequisites"]
        },
        { 
          title: "Quick Start", 
          href: "/docs/quick-start",
          keywords: ["example", "tutorial", "usage", "basic", "code", "implementation", "first model"]
        },
      ],
    },
    {
      title: "Core Concepts",
      items: [
        { 
          title: "Neural Resonance", 
          href: "/docs/neural-resonance",
          keywords: ["prospective configuration", "thinking phase", "feedback", "bidirectional", "alignment", "synchronization"]
        },
        { 
          title: "Biological Layers", 
          href: "/docs/layers",
          keywords: ["bioplastic", "hebbian", "homeostatic", "plasticity", "dense", "synapse", "neuroplasticity", "PredictiveResonantLayer", "predictive resonance", "local resonance"]
        },
        { 
          title: "ResonantGSER", 
          href: "/docs/resonant-gser",
          keywords: ["spiking", "reservoir", "dynamics", "spectral radius", "leak rate", "threshold", "reservoir computing"]
        },
        { 
          title: "PredictiveResonantLayer",
          href: "/docs/predictive-resonant-layer",
          keywords: ["predictive resonance", "local resonance", "recurrent", "alignment", "autonomy"]
        },
        { 
          title: "Activations", 
          href: "/docs/activations",
          keywords: ["spike", "gelu", "softplus", "neuromimetic", "stateful", "lif"]
        },
      ],
    },
    {
      title: "Models",
      items: [
        { 
          title: "ARC 1", 
          href: "/docs/arc-1",
          keywords: ["arc1", "arc 1", "automation", "tools", "tool calling", "extract", "engram", "ladder", "needle"]
        },
        { 
          title: "Chat with ARCANE", 
          href: "/docs/chat",
          keywords: ["chat", "slm", "language model", "generate", "talk", "100m", "tiny", "inference"]
        },
        { 
          title: "Foundation Model", 
          href: "/docs/foundation-model",
          keywords: ["hierarchical", "ollama", "llama", "large language model", "reasoning", "advanced"]
        },
        { 
          title: "Neuromimetic Model", 
          href: "/docs/neuromimetic-model",
          keywords: ["semantic", "standard", "nlp", "text generation", "prototype"]
        },
      ],
    },
    {
      title: "Reference",
      items: [
        { 
          title: "CLI Commands", 
          href: "/docs/cli",
          keywords: ["terminal", "about", "list", "version", "commands", "tools"]
        },
        { 
          title: "MNIST Demo", 
          href: "/docs/mnist-demo",
          keywords: ["mnist", "demo", "draw", "digit", "inference", "test", "predict", "weights"]
        },
        { 
          title: "ARCANE Fruitfly Brain", 
          href: "/docs/fruitfly",
          keywords: ["fruit fly", "drosophila", "brain", "mnist", "digit", "predict", "flywire", "fafbseg", "connectome", "rsaa", "arcane", "optic"]
        },
        { 
          title: "Benchmarks", 
          href: "/docs/benchmarks",
          keywords: ["performance", "accuracy", "loss", "comparison", "metrics", "test", "shakespeare"]
        },
        { 
          title: "Research Paper", 
          href: "/docs/research",
          keywords: ["paper", "rsaa", "academic", "theory", "math", "equations", "alignment gap"]
        },
      ],
    },
    {
      title: "Insights",
      items: [
        { 
          title: "Blog", 
          href: "/docs/blog",
          keywords: ["updates", "news", "articles", "future", "announcements"]
        },
      ],
    },
  ],
};
