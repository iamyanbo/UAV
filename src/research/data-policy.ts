/**
 * Validated market-data acquisition providers. yfinance is keyless; Alpaca market
 * data uses the runtime-held paper key pair, which agents never receive.
 */
export const DATA_PROVIDERS = new Set(["yfinance", "alpaca"]);
// Historical readers remain available. New public text and macro discovery
// goes through source collection, not directly into model feature tables.
export const RETIRED_DATA_PROVIDERS = new Set(["fred", "alfred", "sec", "gdelt"]);
export const DATA_POLICY_REASON = "Use request_discovery for SEC, FRED/ALFRED and GDELT public sources. These are authorized for free discovery; supported free API keys are runtime-held. A separately validated point-in-time feature adapter is required before source content becomes trading input. Existing historical evidence is preserved.";
