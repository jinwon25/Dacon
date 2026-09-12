/** Extract competition-relevant public text from DACON's server-rendered pages. */

import vm from "node:vm";


function parseArgs(argv) {
  const args = { competitionId: "236743", maxMatches: 250, paths: null, summaryOnly: false };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--competition-id") args.competitionId = argv[++index];
    else if (argv[index] === "--max-matches") args.maxMatches = Number(argv[++index]);
    else if (argv[index] === "--paths") args.paths = argv[++index].split(",");
    else if (argv[index] === "--summary-only") args.summaryOnly = true;
    else throw new Error(`Unknown argument: ${argv[index]}`);
  }
  return args;
}


function inlineNuxtScript(html) {
  const scripts = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/gi)]
    .map((match) => match[1])
    .filter((body) => body.includes("window.__NUXT__="));
  if (scripts.length !== 1) throw new Error(`Expected one Nuxt payload, got ${scripts.length}`);
  return scripts[0];
}


function normalizeText(value) {
  return value.replace(/\s+/g, " ").trim();
}


function collectMatches(root, pattern, limit) {
  const matches = [];
  const seenObjects = new WeakSet();
  const seenText = new Set();
  function visit(value, path) {
    if (matches.length >= limit) return;
    if (typeof value === "string") {
      const text = normalizeText(value);
      if (text && pattern.test(text) && !seenText.has(text)) {
        seenText.add(text);
        matches.push({ path, text: text.slice(0, 1500) });
      }
      pattern.lastIndex = 0;
      return;
    }
    if (value === null || typeof value !== "object" || seenObjects.has(value)) return;
    seenObjects.add(value);
    for (const [key, child] of Object.entries(value)) visit(child, `${path}.${key}`);
  }
  visit(root, "$");
  return matches;
}


function structureSummary(nuxt) {
  const data = Array.isArray(nuxt?.data)
    ? nuxt.data.map((item, index) => ({
        index,
        type: Array.isArray(item) ? "array" : typeof item,
        keys: item && typeof item === "object" ? Object.keys(item) : [],
      }))
    : [];
  return {
    data,
    fetch_keys: nuxt?.fetch && typeof nuxt.fetch === "object" ? Object.keys(nuxt.fetch) : [],
  };
}


function stripHtml(value) {
  return normalizeText(
    String(value ?? "")
      .replace(/<br\s*\/?\s*>/gi, "\n")
      .replace(/<\/p>/gi, "\n")
      .replace(/<\/li>/gi, "\n")
      .replace(/<[^>]+>/g, " ")
      .replace(/&nbsp;/gi, " ")
      .replace(/&amp;/gi, "&")
  );
}


function primitiveRecord(value) {
  if (!value || typeof value !== "object") return value;
  const output = {};
  for (const [key, child] of Object.entries(value)) {
    if (/token|picture|team_info/i.test(key)) continue;
    if (["string", "number", "boolean"].includes(typeof child) || child === null) {
      output[key] = typeof child === "string" ? normalizeText(child).slice(0, 800) : child;
    }
  }
  return output;
}


function pageSpecificSummary(nuxt, url) {
  const detail = nuxt?.fetch?.["CompetitionsName:2"]?.competitionDetail;
  const page = Array.isArray(nuxt?.data) ? nuxt.data[2] : null;
  const summary = {};
  if (detail && url.endsWith("/overview/description")) {
    summary.competition = {
      name: detail.name,
      schedule: stripHtml(detail.schedule),
      evaluation_rule: stripHtml(detail.evaluation_rule),
      rule: stripHtml(detail.rule),
      data_explain: stripHtml(detail.data_explain),
    };
  }
  if (url.endsWith("/codeshare") && page) {
    summary.codeshare = {
      total_list: page.total_list,
      top_list: (page.top_list ?? []).map(primitiveRecord),
      share_list: (page.share_list ?? []).map(primitiveRecord),
    };
  }
  if (url.endsWith("/talkboard") && page) {
    summary.talkboard = {
      total_list: page.total_list,
      top_list: (page.top_list ?? []).map(primitiveRecord),
      talk_list: (page.talk_list ?? []).map(primitiveRecord),
    };
  }
  if (/\/talkboard\/\d+$/.test(url) && page) {
    summary.talk = {
      post: primitiveRecord(page.talkboard_data),
      content: stripHtml(page.talkboard_data?.html_content),
      replies: (page.reply ?? []).map(primitiveRecord),
    };
  }
  if (url.endsWith("/leaderboard") && page) {
    summary.leaderboard = {
      top_15: (page.result ?? []).slice(0, 15).map(primitiveRecord),
    };
  }
  return summary;
}


async function auditPage(url, maxMatches) {
  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const html = await response.text();
  const title = normalizeText(html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1] ?? "");
  const sandbox = { window: {} };
  vm.runInNewContext(inlineNuxtScript(html), sandbox, { timeout: 5000 });
  const pattern = /(Brier|Skill|평가|산식|외부|데이터|누수|TrackMan|트랙맨|test|테스트|추론|행 독립|제출|코드|API|규칙|팀|일일|확률)/gi;
  return {
    url,
    status: response.status,
    title,
    nuxt_keys: Object.keys(sandbox.window.__NUXT__ ?? {}),
    structure: structureSummary(sandbox.window.__NUXT__),
    summary: pageSpecificSummary(sandbox.window.__NUXT__, url),
    matches: collectMatches(sandbox.window.__NUXT__, pattern, maxMatches),
  };
}


const args = parseArgs(process.argv.slice(2));
const base = `https://dacon.io/competitions/official/${args.competitionId}`;
const paths = args.paths ?? [
  "/overview/description",
  "/overview/evaluation",
  "/overview/rules",
  "/data",
  "/codeshare",
  "/talkboard",
  "/leaderboard",
];
const pages = [];
for (const path of paths) {
  const page = await auditPage(`${base}${path}`, args.maxMatches);
  pages.push(args.summaryOnly ? { url: page.url, status: page.status, title: page.title, summary: page.summary } : page);
}
console.log(JSON.stringify({ competition_id: args.competitionId, checked_at: new Date().toISOString(), pages }, null, 2));
