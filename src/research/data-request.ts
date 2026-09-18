import { DATA_PROVIDERS, DATA_POLICY_REASON, RETIRED_DATA_PROVIDERS } from "./data-policy.js";

/** Provider arguments, not a schema for scientific reasoning. Never extract these from prose. */
export interface DataRequestParameters {
  provider: string;
  symbols?: string[];
  query?: string;
  kind?: string;
  start?: string;
  end?: string;
  cadence?: string;
  feed?: string;
  adjustment?: string;
  expiry_window_days?: number;
  max_contracts?: number;
  strike_min?: number;
  strike_max?: number;
  option_type?: string;
}

export function validateDataRequest(input: DataRequestParameters): DataRequestParameters {
  if (!input || typeof input.provider !== "string") throw new Error("Choose a data provider in the tool arguments; the research note is never parsed.");
  const provider = input.provider.trim().toLowerCase();
  if (RETIRED_DATA_PROVIDERS.has(provider)) throw new Error(DATA_POLICY_REASON);
  if (!DATA_PROVIDERS.has(provider)) throw new Error(`No acquisition adapter for ${provider}. Use public research tools or delegate acquisition work; this is not an approval request.`);
  const symbols = input.symbols?.map(symbol => symbol.trim().toUpperCase()).filter(Boolean);
  if (provider === "gdelt" ? !input.query?.trim() : !symbols?.length) {
    throw new Error(provider === "gdelt" ? "GDELT requires a query argument." : `${provider} requires symbols in the tool arguments.`);
  }
  if (symbols?.some(symbol => ["ALL", "EVERYTHING", "*"].includes(symbol))) throw new Error("Supply the instruments to acquire; no symbol-count limit applies.");
  const kinds = provider === "alpaca" ? ["bars", "option_bars", "option_chain"]
    : provider === "yfinance" ? ["prices", "option_chain"] : ["article_list"];
  const kind = input.kind || kinds[0]!;
  if (!kinds.includes(kind)) throw new Error(`${provider} supports ${kinds.join(", ")}.`);
  for (const key of ["start", "end"] as const) {
    if (input[key] && !Number.isFinite(Date.parse(input[key]!))) throw new Error(`${key} must be an actual date; omit an open bound.`);
  }
  if (input.start && input.end && Date.parse(input.start) >= Date.parse(input.end)) throw new Error("The end date must follow the start date.");
  return { ...input, provider, kind, ...(symbols ? { symbols: [...new Set(symbols)] } : {}) };
}
