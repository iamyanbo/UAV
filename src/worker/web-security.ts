import { lookup } from "node:dns/promises";
import { isIP } from "node:net";

function privateIp(address: string): boolean {
  const ip = address.toLowerCase();
  if (isIP(ip) === 4) {
    const [a, b] = ip.split(".").map(Number);
    return a === 0 || a === 10 || a === 127 || (a === 169 && b === 254)
      || (a === 172 && b! >= 16 && b! <= 31) || (a === 192 && b === 168)
      || (a === 100 && b! >= 64 && b! <= 127) || a! >= 224;
  }
  if (isIP(ip) === 6) {
    return ip === "::" || ip === "::1" || ip.startsWith("fc") || ip.startsWith("fd")
      || /^fe[89ab]/.test(ip) || ip.startsWith("::ffff:127.") || ip.startsWith("::ffff:10.")
      || ip.startsWith("::ffff:169.254.") || ip.startsWith("::ffff:192.168.");
  }
  return true;
}

export async function assertPublicUrl(url: URL): Promise<void> {
  if (!/^https?:$/.test(url.protocol)) throw new Error("URL is not HTTP(S)");
  const hostname = url.hostname.toLowerCase().replace(/^\[|\]$/g, "");
  if (hostname === "localhost" || hostname === "metadata.google.internal" || hostname.endsWith(".localhost")) {
    throw new Error("URL host is not public");
  }
  const addresses = isIP(hostname) ? [{ address: hostname }] : await lookup(hostname, { all: true, verbatim: true });
  if (addresses.length === 0 || addresses.some(({ address }) => privateIp(address))) {
    throw new Error("URL resolves to a non-public address");
  }
}
