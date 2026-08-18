/** Read one team's public rows from DACON's server-rendered leaderboard. */

import fs from "node:fs";
import vm from "node:vm";


function parseArgs(argv) {
  const args = {
    competitionId: "236743",
    credentialsFile: null,
    team: null,
    diagnostic: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--competition-id") args.competitionId = argv[++i];
    else if (argv[i] === "--credentials-file") args.credentialsFile = argv[++i];
    else if (argv[i] === "--team") args.team = argv[++i];
    else if (argv[i] === "--diagnostic") args.diagnostic = true;
    else throw new Error(`Unknown argument: ${argv[i]}`);
  }
  if (!args.team && !args.credentialsFile) {
    throw new Error("--team or --credentials-file is required");
  }
  return args;
}


function loadTeam(path) {
  const lines = fs.readFileSync(path, "utf8").split(/\r?\n/);
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    const [key, ...rest] = line.split("=");
    if (["DACON_TEAM", "DACON_TEAM_NAME"].includes(key.trim())) {
      return rest.join("=").trim().replace(/^['"]|['"]$/g, "");
    }
  }
  throw new Error("DACON team key is missing from the credentials file");
}


function inlineNuxtScript(html) {
  const scripts = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/gi)]
    .map((match) => match[1])
    .filter((body) => body.includes("window.__NUXT__="));
  if (scripts.length !== 1) throw new Error(`Expected one Nuxt payload, got ${scripts.length}`);
  return scripts[0];
}


function objectsContaining(root, needle) {
  const found = [];
  const seen = new WeakSet();
  function visit(value, path) {
    if (value === null || typeof value !== "object" || seen.has(value)) return;
    seen.add(value);
    const entries = Object.entries(value);
    if (entries.some(([, child]) => child === needle)) found.push({ path, value });
    for (const [key, child] of entries) visit(child, `${path}.${key}`);
  }
  visit(root, "$.");
  return found;
}


function sanitize(value, team) {
  if (value === team) return "[TEAM]";
  if (Array.isArray(value)) return value.map((item) => sanitize(item, team));
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, sanitize(item, team)]));
  }
  return value;
}


const args = parseArgs(process.argv.slice(2));
const team = args.team ?? loadTeam(args.credentialsFile);
const url = `https://dacon.io/competitions/official/${args.competitionId}/leaderboard`;
const response = await fetch(url, { redirect: "follow" });
if (!response.ok) throw new Error(`Leaderboard request failed: HTTP ${response.status}`);
const html = await response.text();
const sandbox = { window: {} };
vm.runInNewContext(inlineNuxtScript(html), sandbox, { timeout: 5000 });
const matches = objectsContaining(sandbox.window.__NUXT__, team);
const output = {
  competition_id: args.competitionId,
  checked_at: new Date().toISOString(),
  team_found: matches.length > 0,
  matches: matches.map(({ path, value }) => ({ path, row: sanitize(value, team) })),
};
if (args.diagnostic) output.nuxt_keys = Object.keys(sandbox.window.__NUXT__ ?? {});
console.log(JSON.stringify(output, null, 2));
