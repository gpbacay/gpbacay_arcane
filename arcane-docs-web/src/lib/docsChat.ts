/**
 * What the docs chat knows, and how a message becomes a route.
 * ARC 1 only classifies. The reply text comes from these pages.
 */

export type DocPage = {
  id: string;
  title: string;
  href: string;
  answer: string;
  /** Phrases a navigation command must contain. Longer aliases win. */
  aliases: string[];
  keys: RegExp[];
  /** Imperative examples that teach ARC 1 this page. Not used as replies. */
  nav: string[];
  /** One sentence per question. The first match on the last clause wins. */
  facts?: { key: RegExp; text: string }[];
};

export type ChatIntent = "docs" | "none";

export type ChatRoute = {
  intent: ChatIntent;
  pageId: string | null;
  /** Page Hippocampus retrieved for this message. Not a navigation target. */
  recallId: string | null;
  /** model, memory, or model+memory. */
  source: string | null;
  confidence: number;
  latencyMs: number;
  calls: string[];
  /** The closest remembered examples, shown in the chat's thought trace. */
  recalled?: Neighbor[];
};

type Neighbor = { text: string; label: string | null; score: number };

/** One node of a reply's thought trace. A `tool` step renders as a tool-call card. */
export type TraceStep = { title: string; detail?: string; tool?: string; children?: TraceStep[] };

export type Classifier = {
  classify(text: string, labels: string[]): Promise<{ distribution: Record<string, number>; source?: string }>;
  neighbors?(text: string): Neighbor[];
};

export const PAGES: DocPage[] = [
  {
    id: "installation",
    title: "Installation",
    href: "/docs/installation",
    answer:
      "Install with pip install gpbacay-arcane. You need Python 3.11+ and TensorFlow 2.12+. From a clone: pip install -e .",
    aliases: ["installation", "install"],
    keys: [/\b(install(ation|ing)?|pip|pypi|tensorflow)\b/],
    nav: [
      "go to the installation page",
      "open the install docs",
      "take me to installation",
      "show the setup page",
    ],
  },
  {
    id: "quick-start",
    title: "Quick Start",
    href: "/docs/quick-start",
    answer:
      "The quick start wraps a predictor in System1 and a DecisionSpace, then calls remember() and decide(). A decision below the confidence or margin threshold abstains.",
    aliases: ["quick start", "quickstart", "getting started", "tutorial"],
    keys: [/\bquick ?start\b/, /\bdecision ?space\b/, /\b(first example|tutorial)\b/],
    nav: ["go to the quick start", "open the tutorial", "take me to getting started", "show the quick start page"],
  },
  {
    id: "arc-1",
    title: "ARC 1",
    href: "/docs/arc-1",
    answer:
      "ARC 1 is ARCANE's System 1 decision model. It returns a typed tool call, copied fields, or a label in one pass, and abstains when nothing fits. Unlike a language model used as a router, it does not generate the decision. Unlike a fixed classifier, the schema arrives with the request. Unlike a large encoder or a hosted decision API, you run the 1.3M-parameter file yourself.",
    aliases: ["arc 1", "arc1", "arc-1", "why arc 1", "decision model"],
    keys: [/\barc\s*-?\s*1\b/, /\btool calls?\b/, /\b(extract|classification|abstain)\b/, /\bdecision models?\b/],
    nav: ["go to the arc 1 page", "open arc 1", "take me to the arc1 docs", "show the grounded decisions page"],
  },
  {
    id: "system-1",
    title: "System 1 Architecture",
    href: "/docs/system-1",
    answer:
      "In ARCANE, System 1 means fast, automatic inference that returns a bounded outcome with measurable uncertainty. It is an engineering profile, not a claim about consciousness. ARC 1 is the System 1 decision model: one pass from a request and schema to a tool call, copied fields, a label, or no call.",
    aliases: ["system 1", "system one", "architecture"],
    keys: [/\bsystem\s*1\b/, /\barchitecture\b/, /\boperational definition\b/],
    nav: ["go to system 1", "open the architecture page", "take me to the system 1 docs", "show the system one page"],
  },
  {
    id: "overview",
    title: "Overview",
    href: "/docs",
    answer:
      "ARCANE (Augmented Reconstruction of Consciousness through Artificial Neural Evolution) is a library for System 1 models: adaptive inference, grounded decisions, neural memory, and biologically inspired components.",
    aliases: ["overview", "docs home", "home", "documentation"],
    keys: [/\b(what is arcane|about arcane|this library|overview)\b/],
    nav: ["go to the overview", "open the docs home", "take me to the start of the docs", "show the main docs page"],
  },
  {
    id: "chat",
    title: "Chat with ARCANE",
    href: "/docs/chat",
    answer:
      "The chat page runs ARCANE's compact language models for open-ended generation. They are next-token models, not instruction-tuned assistants. Locally that server starts with npm run dev:with-slm.",
    aliases: ["chat page", "language model", "slm"],
    keys: [/\b(slm|language model|text generation)\b/, /\bchat with arcane\b/],
    nav: ["go to the chat page", "open the language model chat", "take me to the slm demo", "show the generation chat"],
  },
  {
    id: "hippocampus",
    title: "Hippocampus",
    href: "/docs/hippocampus",
    answer:
      "Hippocampus is a fast memory of decided examples. Similar examples vote for their labels and the vote mixes with the model's probabilities, so a new example applies on the next request without retraining.",
    aliases: ["hippocampus", "memory"],
    keys: [/\bhippocampus\b/, /\b(episodic memory|few-shot|remembered examples)\b/],
    nav: ["go to the hippocampus page", "open hippocampus", "take me to neural memory", "show the memory docs"],
  },
  {
    id: "occipital",
    title: "Occipital",
    href: "/docs/occipital",
    answer:
      "Occipital is the visual pathway of Hippocampus. It recalls remembered examples by their letters, the way the brain's visual word form area reads, so typos and other word forms still find their neighbors, and any model behind the memory, ARC 1 included, decides better. It can also store a memory as QR-code frames of an MP4.",
    aliases: ["occipital", "visual memory", "letter recall", "qr video"],
    keys: [/\boccipital\b/, /\b(visual memory|visual cortex|typos?|misspell\w*|qr[- ]?video|qr codes?)\b/],
    nav: ["go to the occipital page", "open occipital", "take me to visual memory", "show the qr video docs"],
  },
  {
    id: "graph-of-thought",
    title: "Graph of Thought",
    href: "/docs/graph-of-thought",
    answer:
      "Graph of Thought grounds answers in a document graph: keyword scoring plus a walk along links, with no vector database. Each hit says how it was reached.",
    aliases: ["graph of thought", "got"],
    keys: [/\bgraph of thought\b/, /\b(hallucination|citation|knowledge graph)\b/],
    nav: ["go to graph of thought", "open the grounding page", "take me to the knowledge graph docs", "show graph of thought"],
  },
  {
    id: "foundation-model",
    title: "Foundation Model",
    href: "/docs/foundation-model",
    answer:
      "The hierarchical resonance foundation model is a System 1 model with per-example resonance, trained feedback, and bounded adaptation.",
    aliases: ["foundation model", "hierarchical resonance"],
    keys: [/\bfoundation model\b/, /\bhierarchical resonance\b/],
    nav: ["go to the foundation model", "open hierarchical resonance", "take me to the foundation model page", "show the foundation docs"],
  },
  {
    id: "neuromimetic-model",
    title: "Neuromimetic Model",
    href: "/docs/neuromimetic-model",
    answer:
      "The neuromimetic semantic model is a compact System 1 sequence model with resonance, activity regulation, and optional fast-weight plasticity.",
    aliases: ["neuromimetic", "semantic model"],
    keys: [/\bneuromimetic\b/, /\bsemantic model\b/],
    nav: ["go to the neuromimetic model", "open the semantic model", "take me to neuromimetic", "show the sequence model page"],
  },
  {
    id: "neural-resonance",
    title: "Neural Resonance",
    href: "/docs/neural-resonance",
    answer:
      "Neural resonance covers prospective configuration, bidirectional feedback, and alignment.",
    aliases: ["neural resonance", "resonance"],
    keys: [/\bneural resonance\b/, /\bprospective configuration\b/, /\bbidirectional\b/],
    nav: ["go to neural resonance", "open the resonance page", "take me to neural resonance", "show the alignment docs"],
  },
  {
    id: "inference-time",
    title: "Inference-Time Adaptation",
    href: "/docs/inference-time-learning",
    answer:
      "Inference-time adaptation uses resonance and bounded updates so the model can adjust to the input it is processing, without a training loop.",
    aliases: ["inference-time", "inference time", "test time", "adaptation"],
    keys: [/\binference[- ]time\b/, /\btest[- ]time\b/, /\bfast weights?\b/],
    nav: ["go to inference-time adaptation", "open test time learning", "take me to the adaptation page", "show inference time docs"],
  },
  {
    id: "homeostatic",
    title: "Homeostatic Plasticity",
    href: "/docs/homeostatic-plasticity",
    answer: "Homeostatic plasticity keeps activity near a target, including BCM-style and attention-guided control.",
    aliases: ["homeostatic", "homeostasis", "plasticity"],
    keys: [/\bhomeostat\w*\b/, /\bbcm\b/],
    nav: ["go to homeostatic plasticity", "open homeostasis", "take me to the plasticity page", "show homeostatic docs"],
  },
  {
    id: "layers",
    title: "Layer Reference",
    href: "/docs/layers",
    answer:
      "The layer reference lists the biological layers you can compose: bioplastic, Hebbian, homeostatic, and related dense synapse layers.",
    aliases: ["layers", "layer reference", "biological layers"],
    keys: [/\blayer reference\b/, /\bbiological layers\b/, /\b(hebbian|bioplastic)\b/],
    nav: ["go to the layer reference", "open the layers page", "take me to biological layers", "show the layer docs"],
  },
  {
    id: "resonant-gser",
    title: "ResonantGSER",
    href: "/docs/resonant-gser",
    answer: "ResonantGSER is the spiking reservoir layer. The page covers its dynamics, spectral radius, leak, and threshold.",
    aliases: ["resonantgser", "gser", "reservoir"],
    keys: [/\b(resonantgser|gser)\b/, /\b(spiking reservoir|spectral radius)\b/],
    nav: ["go to resonant gser", "open the reservoir page", "take me to gser", "show the spiking reservoir docs"],
  },
  {
    id: "predictive",
    title: "Predictive Resonance",
    href: "/docs/predictive-resonant-layer",
    answer: "The predictive resonant layer is a recurrent layer whose local resonance stays aligned with the sequence it is reading.",
    aliases: ["predictive resonance", "predictive resonant"],
    keys: [/\bpredictive resonant\b/, /\bpredictive resonance\b/],
    nav: ["go to predictive resonance", "open the predictive resonant layer", "take me to predictive resonance", "show that layer page"],
  },
  {
    id: "activations",
    title: "Activations",
    href: "/docs/activations",
    answer:
      "Activations include resonant spiking (leaky state, a threshold, and top-down modulation), plus GELU, softplus, and other neuromimetic gates.",
    aliases: ["activations", "activation", "spiking activation"],
    keys: [/\bactivations?\b/, /\b(gelu|softplus|resonant spik\w*)\b/],
    nav: ["go to activations", "open the activation functions", "take me to the activations page", "show spiking activations"],
  },
  {
    id: "cli",
    title: "CLI Commands",
    href: "/docs/cli",
    answer:
      "The CLI commands are gpbacay-arcane-about, gpbacay-arcane-list-models, gpbacay-arcane-list-layers, and gpbacay-arcane-version.",
    aliases: ["cli", "command line", "terminal commands"],
    keys: [/\bcli\b/, /\bcommand[- ]line\b/, /\blist-models\b/],
    nav: ["go to the cli page", "open the terminal commands", "take me to cli docs", "show the command line page"],
  },
  {
    id: "fruitfly",
    title: "Connectome Demo",
    href: "/docs/fruitfly",
    answer:
      "The fruit fly page is a connectome demo: resonance and sparse dynamics on a fly brain wiring diagram, including a fast classifier.",
    aliases: ["fruit fly", "fruitfly", "connectome", "flywire"],
    keys: [/\b(fruit ?fly|connectome|flywire|drosophila)\b/],
    nav: ["go to the fruit fly demo", "open the connectome", "take me to flywire", "show the fruitfly page"],
  },
  {
    id: "benchmarks",
    title: "Benchmarks",
    href: "/docs/benchmarks",
    answer: "Benchmarks report accuracy, calibration, latency, and adaptation cost for the ARCANE models.",
    aliases: ["benchmarks", "benchmark", "metrics"],
    keys: [/\bbenchmarks?\b/, /\b(latency|calibration|accuracy)\b/],
    nav: ["go to benchmarks", "open the metrics page", "take me to performance", "show the benchmark docs"],
  },
  {
    id: "research",
    title: "Research Paper",
    href: "/docs/research",
    answer: "The research page hosts the RSAA paper, Closing the Alignment Gap.",
    aliases: ["research", "paper", "rsaa"],
    keys: [/\b(research paper|rsaa)\b/, /\balignment gap\b/],
    nav: ["go to the research paper", "open the rsaa paper", "take me to the paper", "show the research page"],
  },
  {
    id: "blog",
    title: "Blog",
    href: "/docs/blog",
    answer:
      "The blog is where project updates and articles are posted. The public ARCANE 4.1 article by Gianne Bacay is dated 3 September 2025.",
    aliases: ["blog", "articles", "news"],
    keys: [/\bblog\b/, /\b(articles|announcements|news)\b/],
    nav: ["go to the blog", "open the articles", "take me to the news page", "show the blog"],
  },
  {
    id: "about",
    title: "About ARCANE",
    href: "/docs",
    answer: "Gianne P. Bacay (giannebacay2004@gmail.com). MIT, copyright 2026, v5.0.0.",
    facts: [
      { key: /\b(when|date|dated|release)\b/i, text: "ARCANE 4.1 article: 3 September 2025. This repo's 5.0 release: 13 October 2026." },
      { key: /\b(version|copyright|license|mit)\b/i, text: "v5.0.0. MIT license, copyright 2026." },
      { key: /\b(github|repo)\b/i, text: "github.com/gpbacay/gpbacay_arcane." },
      { key: /\b(who|creator|author|gianne|bacay|person|email)\b/i, text: "Gianne P. Bacay (giannebacay2004@gmail.com)." },
    ],
    aliases: ["creator", "author", "license", "version", "copyright"],
    keys: [
      /\b(who (made|created|built|wrote)|creator|author|gianne|bacay)\b/,
      /\b(when|release date|copyright|what version|mit license)\b/,
      /\b(github\.com\/gpbacay|giannebacay2004)\b/,
    ],
    nav: ["go to the about section", "open the project credits", "show who made arcane"],
  },
];

const SELF = {
  keys: [
    /\b(are you|who are you|this (chat|bot)|how (do|does) (you|this) work|what can you (do|answer))\b/,
    /\b(chatgpt|openai|gpt-?\d|third[- ]party)\b/,
  ],
  answer: "ARC 1, in your browser. Answers come from these docs.",
};

export const MEMORY: Record<string, string[]> = {
  ...Object.fromEntries(PAGES.map((page) => [`go_${page.id}`, page.nav])),
  answer: [
    "what is arcane",
    "how do I install the library",
    "what does pip install gpbacay-arcane need",
    "how does the quick start work",
    "what is arc 1",
    "how fast is arc 1",
    "what is system 1",
    "what does hippocampus remember",
    "how does graph of thought ground an answer",
    "what is the foundation model",
    "what is the neuromimetic model",
    "what is neural resonance",
    "what is inference-time adaptation",
    "what is homeostatic plasticity",
    "which biological layers exist",
    "what is resonant gser",
    "what is the predictive resonant layer",
    "which activation functions ship",
    "what cli commands are there",
    "what does the fruit fly demo show",
    "where are the benchmarks",
    "where is the research paper",
    "what is on the blog",
    "who made this library",
    "who created arcane",
    "when was arcane made",
    "what version is arcane",
    "what can you tell me about this site",
  ],
  none: [
    "write a poem about the sea",
    "what is the weather today",
    "who won the finals",
    "tell me a joke",
    "translate this to spanish",
    "help me with my homework",
    "what is 15 times 12",
    "recommend a movie",
    "write a python function to reverse a string",
    "hi",
    "thanks",
    "good morning",
  ],
};

export const LABELS = Object.keys(MEMORY);

/** Routing examples, plus every docs page so Hippocampus can retrieve the passage itself. */
export const KNOWLEDGE = PAGES.flatMap((page) => [
  { text: `${page.title}. ${page.answer}`, label: page.id },
  ...(page.facts ?? []).map((fact) => ({ text: fact.text, label: page.id })),
  ...page.aliases.map((alias) => ({ text: `${page.title} ${alias}`, label: page.id })),
]);

export const EXAMPLES = [
  ...Object.entries(MEMORY).flatMap(([label, texts]) => texts.map((text) => ({ text, label }))),
  ...KNOWLEDGE,
];

const FOLLOW =
  /\b(it|its|that|this|those|they|them|he|his|her|one|else|also|more)\b|^(and|but|what about|how about|why|how|when|where|who|which)\b/i;

/** Short or pronoun follow-ups keep the previous user turn. A full question stands alone. */
export function withContext(text: string, history: { role: string; text: string }[]): string {
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (!words.length || (words.length > 8 && !FOLLOW.test(text))) return text;
  const prior = [...history].reverse().find((message) => message.role === "user" && message.text.trim())?.text.trim();
  return prior ? `${prior}. ${text.trim()}` : text.trim();
}

export function pageById(id: string | null): DocPage | null {
  return PAGES.find((page) => page.id === id) ?? null;
}

const NAV_LEAD =
  /^\s*(?:please\s+)?(?:go|navigate|take|send|open|show|jump|bring|head)(?:\s+(?:me|us))?(?:\s+(?:to|over to|back to))?\s+/i;

/** A bare "open the X page" command. Questions stay in the chat. */
export function navigationOnly(text: string): string | null {
  const raw = text.trim();
  if (!raw || /[?]/.test(raw) || /\b(what|why|how|who|when|explain|tell|describe|is|are)\b/i.test(raw)) return null;
  if (!NAV_LEAD.test(raw)) return null;
  const rest = raw
    .replace(NAV_LEAD, "")
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .replace(/\b(the|a|an|page|section|docs|documentation|doc)\b/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!rest) return null;
  let best: { id: string; length: number } | null = null;
  for (const page of PAGES) {
    for (const alias of page.aliases) {
      if (rest.includes(alias) && (best === null || alias.length > best.length)) best = { id: page.id, length: alias.length };
    }
  }
  return best?.id ?? null;
}

const GREETING = /^\s*(hi|hello|hey|yo|good (morning|afternoon|evening))\b/i;
const THANKS = /^\s*(thanks|thank you|thx|ty)\b/i;

/** The sentence that answers this question, not the whole page. */
function brief(page: DocPage, message: string): { text: string; pageId: string } {
  const clause = message.split(/[.?!]\s+/).pop()?.trim() || message;
  const hits = (page.facts ?? []).filter((fact) => fact.key.test(clause));
  return { text: hits.length ? hits.map((fact) => fact.text).join(" ") : page.answer, pageId: page.id };
}

export function answerDocs(message: string, route: ChatRoute | null = null): { text: string; pageId: string | null } {
  const text = message.toLowerCase();
  if (SELF.keys.some((key) => key.test(text))) return { text: SELF.answer, pageId: null };

  let best: DocPage | null = null;
  let score = 0;
  for (const page of PAGES) {
    const next = page.keys.filter((key) => key.test(text)).length;
    if (next > score) [best, score] = [page, next];
  }
  if (best) return brief(best, message);

  if (THANKS.test(message)) return { text: "You're welcome.", pageId: null };
  if (GREETING.test(message)) return { text: "Hi. Ask me about the ARCANE docs.", pageId: null };
  if (route?.intent === "none") return { text: "I only answer questions about ARCANE.", pageId: null };
  const recalled = pageById(route?.recallId ?? null);
  if (recalled) return brief(recalled, message);
  const routed = pageById(route?.pageId ?? null);
  if (routed && route?.intent === "docs") return brief(routed, message);
  return { text: "Not in the docs.", pageId: null };
}

function callName(call: string | null): string {
  if (call === "answer") return "answer";
  if (!call || call === "none") return "no tool";
  return `open ${pageById(call.slice(3))?.title ?? call}`;
}

/** The thought trace for one reply: `label` heads it, `steps` are its nodes, ending with what the chat did. */
export function routeThought(route: ChatRoute | null, outcome: TraceStep): { label: string; steps: TraceStep[] } {
  if (!route) {
    return {
      label: "Docs only",
      steps: [{ title: "ARC 1 is still loading", detail: "This reply comes straight from the docs." }, outcome],
    };
  }
  // The chat's Hippocampus always recalls through Occipital too (arc1Router.ts).
  const via =
    route.source === "memory" ? "Hippocampus + Occipital" : route.source === "model+memory" ? "ARC 1 + Hippocampus + Occipital" : "ARC 1";
  const recalled = route.recalled ?? [];
  const recall: TraceStep = recalled.length
    ? {
        title: `Hippocampus + Occipital recalled ${recalled.length} example${recalled.length === 1 ? "" : "s"}`,
        detail: "Matched by words, and by letters through Occipital, so typos still find them.",
        // A label is a routing call (go_x, answer, none) or, for remembered docs passages, a page id.
        children: recalled.map((hit) => ({
          title: `“${hit.text.length > 60 ? `${hit.text.slice(0, 57)}…` : hit.text}” → ${pageById(hit.label)?.title ?? callName(hit.label)} · ${Math.round(hit.score * 100)}%`,
        })),
      }
    : { title: "Hippocampus + Occipital found no similar example" };
  const names = route.calls.map(callName);
  const decision: TraceStep = {
    title: `${route.source === "memory" ? "Hippocampus" : "ARC 1"} chose ${names.length ? names.join(", ") : "nothing it was sure of"}`,
    detail: `${Math.round(route.confidence * 100)}% confident · ${Math.round(route.latencyMs)} ms`,
  };
  return { label: `${via} · ${Math.round(route.latencyMs)} ms`, steps: [recall, decision, outcome] };
}

// ponytail: one margin for every label; split per label if a real log shows a systematic miss.
const INTENT_MIN_MARGIN = 0.05;
const NAV_MIN_MARGIN = 0.1;
const CLAUSES = /\s+(?:and also|and then|and|then)\s+|[,.;!]\s+(?:also\s+)?/;
const QUESTION = /[?]|\b(what|why|how|who|when|explain|tell|describe)\b/i;

/** `hippo` is ARC 1 mixed with Hippocampus, or Hippocampus alone in tests. */
export async function decide(hippo: Classifier, text: string): Promise<ChatRoute> {
  const started = performance.now();
  const clauses = text.split(CLAUSES).filter((clause) => /\w/.test(clause));
  const decided = [];
  let source: string | null = null;
  for (const clause of clauses.length > 1 ? clauses : [text]) {
    const result = await hippo.classify(clause, LABELS);
    if (!source && result.source) source = result.source;
    const ranked = Object.entries(result.distribution).sort((a, b) => b[1] - a[1]);
    const confidence = ranked[0]?.[1] ?? 0;
    decided.push({
      label: ranked[0]?.[0] ?? "none",
      confidence,
      margin: confidence - (ranked[1]?.[1] ?? 0),
    });
  }
  const recalled = (hippo.neighbors?.(text) ?? []).slice(0, 3);
  const recallId = recalled.find((hit) => pageById(hit.label))?.label ?? null;
  const sure = decided.filter((item) => item.margin >= INTENT_MIN_MARGIN);
  const has = (label: string) => sure.some((item) => item.label === label);
  const nav = sure.find((item) => item.label.startsWith("go_") && item.margin >= NAV_MIN_MARGIN);
  const pageId = nav && pageById(nav.label.slice(3)) ? nav.label.slice(3) : null;
  // A question is answered in the chat. Only a command opens a page.
  const asking = QUESTION.test(text);
  return {
    intent: asking || sure.length < decided.length || has("answer") ? "docs" : "none",
    pageId: asking ? null : pageId,
    recallId,
    source,
    confidence: Math.min(...decided.map((item) => item.confidence)),
    latencyMs: Math.round(performance.now() - started),
    calls: sure.map((item) => item.label),
    recalled,
  };
}
