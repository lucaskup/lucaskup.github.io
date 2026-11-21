#!/usr/bin/env node
/* Refreshes data/metrics.json with the citation metrics OpenAlex holds for
   the author, looked up by ORCID. build.js only reads the JSON file, so a
   failed fetch leaves the last saved numbers in place. The deploy workflow
   runs this before every build; the committed JSON is the fallback.
   No dependencies (needs Node 18+ for fetch) — run with
   `node scripts/update-metrics.js`. */

const fs = require('fs');
const path = require('path');

const ORCID = '0000-0003-2580-3996';
const API = `https://api.openalex.org/authors/orcid:${ORCID}` +
  '?select=id,cited_by_count,summary_stats,counts_by_year';
const OUT = path.join(__dirname, '..', 'data', 'metrics.json');

const isCount = n => Number.isInteger(n) && n >= 0;

(async () => {
  const res = await fetch(API, { signal: AbortSignal.timeout(30000) });
  if (!res.ok) throw new Error(`OpenAlex responded with HTTP ${res.status}`);
  const a = await res.json();

  const stats = a.summary_stats || {};
  if (!isCount(a.cited_by_count) || !isCount(stats.h_index) || !isCount(stats.i10_index)
      || !Array.isArray(a.counts_by_year)) {
    throw new Error('Unexpected OpenAlex response shape');
  }

  // Oldest first, starting at the first year with a citation.
  const years = a.counts_by_year
    .filter(y => isCount(y.year) && isCount(y.cited_by_count))
    .map(y => ({ year: y.year, citations: y.cited_by_count }))
    .sort((x, y) => x.year - y.year);
  const first = years.findIndex(y => y.citations > 0);

  const metrics = {
    source: 'OpenAlex',
    profile: a.id,
    updated: new Date().toISOString().slice(0, 10),
    citations: a.cited_by_count,
    h_index: stats.h_index,
    i10_index: stats.i10_index,
    by_year: first === -1 ? [] : years.slice(first)
  };

  // A sharp drop means OpenAlex mis-merged or split the author profile;
  // keep the saved numbers until someone looks at it.
  let saved = null;
  try { saved = JSON.parse(fs.readFileSync(OUT, 'utf8')); } catch (e) { /* first run */ }
  if (saved && metrics.citations < 0.8 * saved.citations) {
    throw new Error(`citations fell from ${saved.citations} to ${metrics.citations}; keeping saved file`);
  }

  fs.writeFileSync(OUT, JSON.stringify(metrics, null, 2) + '\n');
  console.log(`metrics updated: ${metrics.citations} citations, h ${metrics.h_index}, i10 ${metrics.i10_index}`);
})().catch(e => {
  console.error('metrics not updated: ' + e.message);
  process.exit(1);
});
