import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseEnv } from "node:util";

/** The deterministic broker process alone receives trading credentials. */
export function brokerCredentialNames(env: NodeJS.ProcessEnv = process.env): string[] {
  return Object.keys(env).filter(key => /^(APCA_|ALPACA_)/i.test(key));
}

export function withoutBrokerCredentials(env: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const clean = { ...env };
  for (const key of brokerCredentialNames(clean)) delete clean[key];
  return clean;
}

/** Pi's RpcClient merges overrides into process.env, so inherited keys must be blanked. */
export function blankBrokerCredentials(env: NodeJS.ProcessEnv = process.env): Record<string, string> {
  return Object.fromEntries(brokerCredentialNames(env).map(key => [key, ""]));
}

/** Broker settings from the configured env file, overridden by the real environment. */
export function loadBrokerEnvironment(root: string, env: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const configured = env.ALPACA_ENV_FILE?.trim();
  const path = configured ? resolve(configured) : join(root, ".env.alpaca");
  const fileEnv = existsSync(path) ? parseEnv(readFileSync(path, "utf8")) : {};
  const merged = { ...fileEnv, ...env };
  // The reference quant checkout uses ALPACA_BASE_URL; normalize it to the
  // adapter's paper-only name so a live endpoint can never be accepted silently.
  if (!merged.APCA_API_BASE_URL && merged.ALPACA_BASE_URL) merged.APCA_API_BASE_URL = merged.ALPACA_BASE_URL;
  return merged;
}

/**
 * Market-data credentials for the runtime's own acquisition subprocess. Alpaca
 * serves market data with the same paper key pair used for orders. Agents never
 * receive it: model and candidate processes strip every broker variable.
 */
export function marketDataCredentials(root: string, env: NodeJS.ProcessEnv = process.env): Record<string, string> {
  const broker = loadBrokerEnvironment(root, env);
  const key = broker.APCA_API_KEY_ID?.trim() || broker.ALPACA_API_KEY?.trim();
  const secret = broker.APCA_API_SECRET_KEY?.trim() || broker.ALPACA_SECRET_KEY?.trim();
  return key && secret ? { APCA_API_KEY_ID: key, APCA_API_SECRET_KEY: secret } : {};
}
