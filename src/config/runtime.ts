export type RuntimeProfile = "local" | "cloud";
export type ComputeBackend = "local" | "cloud-run";
export type StateBackend = "sqlite" | "firestore";

export interface RuntimeConfig {
  profile: RuntimeProfile;
  /** Pi provider name from ~/.pi/agent/models.json (or a Pi built-in provider). */
  piProvider: string;
  compute: ComputeBackend;
  store: StateBackend;
  region: string;
  maxCostUsd: number;
}

const valueOf = (argv: string[], name: string): string | undefined => {
  const index = argv.indexOf(`--${name}`);
  return index >= 0 ? argv[index + 1] : undefined;
};

function choice<T extends string>(value: string | undefined, allowed: readonly T[], fallback: T, name: string): T {
  const selected = (value ?? fallback) as T;
  if (!allowed.includes(selected)) throw new Error(`invalid --${name} ${selected}; expected ${allowed.join("|")}`);
  return selected;
}

export function runtimeConfig(argv = process.argv.slice(2), env = process.env): RuntimeConfig {
  const profile = choice(valueOf(argv, "profile") ?? env.AR_PROFILE,
    ["local", "cloud"] as const, "local", "profile");
  const defaults = profile === "cloud"
    ? { compute: "cloud-run" as const, store: "firestore" as const }
    : { compute: "local" as const, store: "sqlite" as const };
  const piProvider = (valueOf(argv, "provider") ?? env.AR_PI_PROVIDER ?? "dgx-spark").trim();
  if (!piProvider) throw new Error("Pi provider must not be empty");
  const compute = choice(valueOf(argv, "compute") ?? env.AR_COMPUTE,
    ["local", "cloud-run"] as const, defaults.compute, "compute");
  const store = choice(valueOf(argv, "store") ?? env.AR_STORE,
    ["sqlite", "firestore"] as const, defaults.store, "store");
  const maxCostUsd = Number(valueOf(argv, "max-cost") ?? env.AR_MAX_COST_USD ?? 0);
  if (!Number.isFinite(maxCostUsd) || maxCostUsd < 0) throw new Error("max cost must be a finite non-negative number");
  return {
    profile,
    piProvider,
    compute,
    store,
    region: valueOf(argv, "region") ?? env.GOOGLE_CLOUD_LOCATION ?? "us-central1",
    maxCostUsd,
  };
}

/** Apply profile defaults before the first worker is invoked. */
export function configureRuntime(argv = process.argv.slice(2)): RuntimeConfig {
  const config = runtimeConfig(argv);
  process.env.AR_PROFILE = config.profile;
  process.env.AR_PI_PROVIDER = config.piProvider;
  process.env.AR_COMPUTE = config.compute;
  process.env.AR_STORE = config.store;
  process.env.GOOGLE_CLOUD_LOCATION = config.region;
  return config;
}
