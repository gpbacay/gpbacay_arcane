// Tool-schema helpers shared by web.mjs and hippocampus.mjs. Port of gpbacay_arcane/tools.py.

export const WORD_NUMBERS = { zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  eleven: 11, twelve: 12, fifteen: 15, twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, hundred: 100, a: 1, an: 1 };
const roundHalfEven = (x) => (Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : Math.round(x));

/** Convert a copied span to the parameter's JSON type: [ok, value]. */
export function coerceValue(type, text) {
  type = (type || "string").toLowerCase();
  if (!["integer", "int", "number", "float"].includes(type)) return [Boolean(text), text];
  const m = text.match(/-?\d+(?:[.,]\d+)*/);
  const word = text.trim().toLowerCase();
  const num = m ? Number(m[0].replaceAll(",", "")) : word in WORD_NUMBERS ? WORD_NUMBERS[word] : NaN;
  if (Number.isNaN(num)) return [false, null];
  return [true, type === "integer" || type === "int" ? roundHalfEven(num) : num];
}

export function normalizeTool(t) {
  return {
    name: t.name, description: t.description || "",
    parameters: (t.parameters || []).map((p) => ({
      name: p.name, type: p.type || "string", description: p.description || "",
      required: p.required ?? true, enum: p.enum || null, enum_descriptions: p.enum_descriptions || null,
    })),
  };
}

/** Drop unknown tools and strip unknown / mistyped arguments. */
export function validateCalls(calls, tools) {
  const byName = new Map(tools.map((t) => [t.name, t]));
  return calls.flatMap((call) => {
    const spec = byName.get(call.name || "");
    if (!spec) return [];
    const allowed = new Map(spec.parameters.map((p) => [p.name, p]));
    const cleaned = {};
    for (const [k, v] of Object.entries(call.arguments || {})) {
      const p = allowed.get(k);
      if (p && !(p.enum && !p.enum.includes(String(v)))) cleaned[k] = v;
    }
    return spec.parameters.some((p) => p.required && !(p.name in cleaned)) ? [] : [{ name: spec.name, arguments: cleaned }];
  });
}
