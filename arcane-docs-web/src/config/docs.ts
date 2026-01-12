export type NavItem = {
  title: string;
  href: string;
  disabled?: boolean;
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
        { title: "Introduction", href: "/docs" },
        { title: "Installation", href: "/docs/installation" },
        { title: "Quick Start", href: "/docs/quick-start" },
      ],
    },
    {
      title: "Core Concepts",
      items: [
        { title: "Neural Resonance", href: "/docs/neural-resonance" },
        { title: "Biological Layers", href: "/docs/layers" },
        { title: "ResonantGSER", href: "/docs/resonant-gser" },
        { title: "Activations", href: "/docs/activations" },
      ],
    },
    {
      title: "Models",
      items: [
        { title: "Foundation Model", href: "/docs/foundation-model" },
        { title: "Neuromimetic Model", href: "/docs/neuromimetic-model" },
      ],
    },
    {
      title: "Reference",
      items: [
        { title: "CLI Commands", href: "/docs/cli" },
        { title: "Benchmarks", href: "/docs/benchmarks" },
        { title: "Research Paper", href: "/docs/research" },
      ],
    },
    {
      title: "Insights",
      items: [
        { title: "Blog", href: "/docs/blog" },
      ],
    },
  ],
};
