/**
 * Cloudflare Pages advanced-mode Worker: Markdown content negotiation and
 * agent-friendly Markdown 404 bodies. Static assets still come from ASSETS.
 *
 * Protocol: https://acceptmarkdown.com/ (Accept parsing, Vary: Accept, 406).
 */

const PRODUCES = ["text/html", "text/markdown"];

const STATIC_EXT =
  /\.(?:css|js|mjs|map|png|jpe?g|webp|gif|svg|avif|ico|woff2?|ttf|otf|eot|xml|txt|json|pdf|mp4|webm|mp3|wav|ogg|zip)$/i;

/** Clean URL paths that ship a sibling .md representation from the site build. */
const MARKDOWN_PAGES = new Set(["/", "/week3", "/week4", "/about", "/contact", "/privacy"]);

const NOT_FOUND_MARKDOWN = `# Page not found

The requested path is not part of UAV Ground School.

Recovery links for agents and tools:

- [llms.txt](https://uav-ground-school.pages.dev/llms.txt) agent index and when-to-use guidance
- [Sitemap](https://uav-ground-school.pages.dev/sitemap.xml) every public HTML page
- [Home](https://uav-ground-school.pages.dev/) computer vision showcase and week index
`;

function parseAccept(header) {
  return header
    .split(",")
    .map((raw) => {
      const parts = raw
        .trim()
        .split(";")
        .map((s) => s.trim());
      const type = (parts[0] || "").toLowerCase();
      if (!type) return null;
      let q = 1;
      for (const param of parts.slice(1)) {
        const eq = param.indexOf("=");
        if (eq === -1) continue;
        const name = param.slice(0, eq).trim().toLowerCase();
        const value = param.slice(eq + 1).trim();
        if (name === "q") {
          const parsed = Number(value);
          if (!Number.isNaN(parsed)) q = Math.max(0, Math.min(1, parsed));
        }
      }
      const specificity = type === "*/*" ? 0 : type.endsWith("/*") ? 1 : 2;
      return { type, q, specificity };
    })
    .filter((e) => e !== null);
}

function matches(entry, candidate) {
  if (entry.type === "*/*") return true;
  if (entry.type.endsWith("/*")) return candidate.startsWith(entry.type.slice(0, -1));
  return entry.type === candidate;
}

/**
 * Pick the representation to serve, or null when every produced type is rejected.
 * Missing / empty Accept defaults to text/html (first of PRODUCES).
 */
function preferredType(header, produces) {
  if (!header || !header.trim()) return produces[0] ?? null;
  const entries = parseAccept(header);
  if (entries.length === 0) return produces[0] ?? null;

  let bestType = null;
  let bestQ = -1;
  let bestPosition = Infinity;

  for (const candidate of produces) {
    let matched = null;
    let matchedPosition = Infinity;
    for (let idx = 0; idx < entries.length; idx++) {
      const e = entries[idx];
      if (!matches(e, candidate)) continue;
      if (
        matched === null ||
        e.specificity > matched.specificity ||
        (e.specificity === matched.specificity && idx < matchedPosition)
      ) {
        matched = e;
        matchedPosition = idx;
      }
    }
    if (matched === null || matched.q <= 0) continue;
    if (matched.q > bestQ || (matched.q === bestQ && matchedPosition < bestPosition)) {
      bestQ = matched.q;
      bestPosition = matchedPosition;
      bestType = candidate;
    }
  }
  return bestType;
}

function normalizePath(pathname) {
  if (!pathname || pathname === "/") return "/";
  const clean = pathname.replace(/\/+$/, "");
  return clean || "/";
}

function markdownPath(pathname) {
  const clean = normalizePath(pathname);
  if (clean === "/") return "/index.md";
  return `${clean}.md`;
}

function appendVaryAccept(headers) {
  const existing = headers.get("vary");
  if (!existing) {
    headers.set("Vary", "Accept");
    return;
  }
  const tokens = existing.split(",").map((s) => s.trim().toLowerCase());
  if (!tokens.includes("accept")) {
    headers.set("Vary", `${existing}, Accept`);
  }
}

function markdownNotFound(request) {
  return new Response(request.method === "HEAD" ? null : NOT_FOUND_MARKDOWN, {
    status: 404,
    headers: {
      "Content-Type": "text/markdown; charset=utf-8",
      Vary: "Accept",
      "Cache-Control": "no-store",
    },
  });
}

function notAcceptable() {
  const body = "Not Acceptable\n\nAvailable: text/html, text/markdown\n";
  return new Response(body, {
    status: 406,
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      Vary: "Accept",
    },
  });
}

export { preferredType, markdownPath, normalizePath, parseAccept, MARKDOWN_PAGES, NOT_FOUND_MARKDOWN };

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Machine-readable and binary assets: no negotiation.
    if (STATIC_EXT.test(url.pathname)) {
      return env.ASSETS.fetch(request);
    }

    // Explicit .md URLs stay Markdown even without Accept.
    if (url.pathname.endsWith(".md")) {
      const asset = await env.ASSETS.fetch(request);
      if (asset.status === 404) return markdownNotFound(request);
      if (asset.status >= 300 && asset.status < 400) return asset;
      const res = new Response(asset.body, asset);
      res.headers.set("Content-Type", "text/markdown; charset=utf-8");
      appendVaryAccept(res.headers);
      return res;
    }

    const accept = request.headers.get("Accept");
    const chosen = preferredType(accept, PRODUCES);

    if (chosen === null && accept && accept.trim()) {
      return notAcceptable();
    }

    if (chosen === "text/markdown") {
      const mdUrl = new URL(url);
      mdUrl.pathname = markdownPath(url.pathname);
      const mdRes = await env.ASSETS.fetch(new Request(mdUrl.toString(), request));
      if (mdRes.status === 200 || mdRes.status === 304) {
        const res = new Response(mdRes.body, mdRes);
        res.headers.set("Content-Type", "text/markdown; charset=utf-8");
        appendVaryAccept(res.headers);
        return res;
      }
      if (mdRes.status >= 300 && mdRes.status < 400) return mdRes;
      // Unknown path (or missing sibling): always a Markdown 404 for markdown clients.
      return markdownNotFound(request);
    }

    const htmlRes = await env.ASSETS.fetch(request);
    const res = new Response(htmlRes.body, htmlRes);
    appendVaryAccept(res.headers);

    const path = normalizePath(url.pathname);
    if (htmlRes.status === 200 && MARKDOWN_PAGES.has(path)) {
      const alternate = `<${markdownPath(path)}>; rel="alternate"; type="text/markdown"`;
      const existing = res.headers.get("Link");
      res.headers.set("Link", existing ? `${existing}, ${alternate}` : alternate);
    }
    return res;
  },
};
