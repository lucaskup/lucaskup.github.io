#!/usr/bin/env node
/* Prerenders the site into _site/ so crawlers, link previews, and no-JS
   visitors see full content. The JSON files in data/ stay the source of
   truth; this script bakes them into the HTML that GitHub Pages serves.
   No dependencies — run with `node build.js`. */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const ROOT = __dirname;
const OUT = path.join(ROOT, '_site');

// ── Data ──
const read = f => JSON.parse(fs.readFileSync(path.join(ROOT, 'data', f), 'utf8'));
const publications = read('publications.json');
const news = read('news.json');
const projects = read('projects.json');
const advisees = read('advisees.json');
const related = read('related.json');

// ── Shared templates (mirrors of the former client-side renderers) ──
const bil = (en, pt) => `<span class="lang lang-en">${en}</span><span class="lang lang-pt" lang="pt-BR">${pt}</span>`;

const coadviseeTag = bil(' · Co-advisee', ' · Coorientação');

const degreeLabel = {
  "Ph.D.":         m => bil('Ph.D. Candidate', 'Doutorando(a)')  + (m.coadvisee ? coadviseeTag : ''),
  "Master's":      m => bil('M.Sc. Student', 'Mestrando(a)')     + (m.coadvisee ? coadviseeTag : ''),
  "Undergraduate": m => bil('Undergraduate Researcher', 'Pesquisador(a) de Graduação')
};

const levelLabel = {
  "Ph.D.":         bil('Ph.D.', 'Doutorado'),
  "Master's":      bil("Master's", 'Mestrado'),
  "Undergraduate": bil('Undergraduate', 'Graduação')
};

function personLinks(m) {
  const links = [];
  if (m.github)  links.push(`<a href="${m.github}"  target="_blank" rel="noopener">GitHub</a>`);
  if (m.website) links.push(`<a href="${m.website}" target="_blank" rel="noopener">${bil('Profile', 'Perfil')}</a>`);
  return links.length ? `<p class="person-links">${links.join(' · ')}</p>` : '';
}

function personCard(m, level) {
  const topic = m.topic && m.topic !== 'A definir' ? `<p class="person-topic">${m.topic}</p>` : '';
  return `<div class="person-card">
    <p class="person-name">${m.name}</p>
    <p class="person-degree">${degreeLabel[level](m)}</p>
    ${topic}
    ${personLinks(m)}
    <p class="person-year">${bil('Since', 'Desde')} ${m.start}</p>
  </div>`;
}

function alumniCard(m) {
  const coLabel = m.coadvisee ? coadviseeTag : '';
  return `<div class="person-card">
    <p class="person-name">${m.name}</p>
    <p class="person-degree">${levelLabel[m.level] || m.level} — ${bil('Completed', 'Concluído')}${coLabel}</p>
    <p class="person-topic">${m.topic}</p>
    ${personLinks(m)}
    <p class="person-year">${m.start} – ${m.finish}</p>
  </div>`;
}

const badgeLabel = {
  journal:    bil('Journal', 'Periódico'),
  conference: bil('Conference', 'Conferência'),
  workshop:   bil('Workshop', 'Workshop'),
  preprint:   bil('Preprint', 'Preprint')
};

const pubTabLabel = {
  journal:    bil('Journal', 'Periódicos'),
  conference: bil('Conference', 'Conferências'),
  workshop:   bil('Workshop', 'Workshops'),
  preprint:   bil('Preprint', 'Preprints')
};

// BibTeX entry for a publication, used by the copy button on each card.
const escAttr = s => String(s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/\n/g, '&#10;');

function bibtexFor(p) {
  const ascii = s => s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^A-Za-z]/g, '');
  const firstAuthor = p.authors[0].replace(/^\*/, '').trim();
  const lastName = firstAuthor.split(/\s+/).pop();
  const titleWord = (p.title.match(/[A-Za-zÀ-ÿ]{4,}/) || ['pub'])[0];
  const key = (ascii(lastName) + p.year + ascii(titleWord)).toLowerCase();

  const fields = [
    ['title',  `{${p.title}}`],
    ['author', p.authors.map(a => a.replace(/^\*/, '')).join(' and ')]
  ];
  let entry = 'misc';
  if (p.type === 'journal') {
    entry = 'article';
    fields.push(['journal', p.venue]);
  } else if (p.type === 'conference' || p.type === 'workshop') {
    entry = 'inproceedings';
    fields.push(['booktitle', p.venue]);
  } else {
    fields.push(['howpublished', p.venue]);
  }
  fields.push(['year', String(p.year)]);
  if (p.url) fields.push(['url', p.url]);

  const body = fields.map(([k, v]) => `  ${k} = {${v}}`).join(',\n');
  return `@${entry}{${key},\n${body}\n}`;
}

function pubCard(p) {
  const authors = p.authors
    .map(a => a.startsWith('*') ? `<strong>${a.slice(1)}</strong>` : a)
    .join(', ');
  const title = p.url ? `<a href="${p.url}" target="_blank" rel="noopener">${p.title}</a>` : p.title;
  return `<div class="pub-card">
    <p class="pub-title">${title}</p>
    <p class="pub-authors">${authors}</p>
    <div class="pub-meta">
      <span class="pub-venue">${p.venue}</span>
      <span class="pub-year">${p.year}</span>
      <span class="badge badge-${p.type}">${badgeLabel[p.type] || p.type}</span>
      <button type="button" class="bibtex-btn" data-bibtex="${escAttr(bibtexFor(p))}" aria-label="Copy BibTeX citation to clipboard" title="Copy BibTeX citation to clipboard">BibTeX</button>
    </div>
  </div>`;
}

// Same DOM shape buildTabs() used to create; js/toggles.js hydrates it.
// Emitted with tablist/tab/tabpanel ARIA wiring; toggles.js keeps the
// aria-selected and tabindex state in sync and adds arrow-key navigation.
let tabsSeq = 0;
function tabsHtml(tabs, innerClass = 'people-grid') {
  const group = `tabs-${++tabsSeq}`;
  const btns = tabs.map(({ label }, i) =>
    `<button type="button" class="tab-btn${i === 0 ? ' active' : ''}" role="tab"
      id="${group}-tab-${i}" aria-controls="${group}-panel-${i}"
      aria-selected="${i === 0 ? 'true' : 'false'}"${i === 0 ? '' : ' tabindex="-1"'}>${label}</button>`).join('');
  const panels = tabs.map(({ html }, i) =>
    `<div class="tab-panel${i === 0 ? ' active' : ''}" role="tabpanel"
      id="${group}-panel-${i}" aria-labelledby="${group}-tab-${i}" tabindex="0"><div class="${innerClass}">${html}</div></div>`).join('\n');
  return `<div class="tabs" role="tablist">${btns}</div>\n${panels}`;
}

// ── Section renderers ──
function renderNews() {
  const VISIBLE = 5;
  const dateLabel = iso => {
    const d = new Date(iso + 'T00:00:00');
    return bil(
      d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' }),
      d.toLocaleDateString('pt-BR', { month: 'short', year: 'numeric' })
    );
  };
  const items = [...news].sort((a, b) => b.date.localeCompare(a.date));
  const list = items.map((n, i) => {
    const text = bil(n.en, n.pt);
    const external = n.url && /^https?:/.test(n.url);
    const body = n.url
      ? `<a href="${n.url}"${external ? ' target="_blank" rel="noopener"' : ''}>${text}</a>`
      : text;
    return `<div class="news-item${i >= VISIBLE ? ' news-extra' : ''}">
      <span class="news-date">${dateLabel(n.date)}</span>
      <p class="news-text">${body}</p>
    </div>`;
  }).join('\n');
  const toggle = items.length > VISIBLE
    ? `<button type="button" class="news-toggle">
        <span class="news-label-more">${bil(`Show all (${items.length})`, `Ver todas (${items.length})`)}</span>
        <span class="news-label-less">${bil('Show fewer', 'Ver menos')}</span>
      </button>`
    : '';
  return `<div class="news-list">${list}</div>${toggle}`;
}

function renderProjects() {
  const statusLabel = { active: bil('Active', 'Em andamento') };
  return projects.map(p => {
    const tags = (p.tags || []).map(t => `<span class="project-tag">${t}</span>`).join('');
    const meta = [];
    if (p.status)   meta.push(`<span class="project-status project-status--${p.status}">${statusLabel[p.status] || p.status}</span>`);
    if (p.timeline) meta.push(`<span class="project-timeline">${p.timeline}</span>`);
    return `<div class="project-card">
      ${meta.length ? `<div class="project-card-meta">${meta.join('')}</div>` : ''}
      <h3>${p.title}</h3>
      <p class="project-funder">${bil(p.funder_en, p.funder_pt)}</p>
      <p class="project-blurb">${bil(p.blurb_en, p.blurb_pt)}</p>
      ${tags ? `<div class="project-tags">${tags}</div>` : ''}
      <a class="project-link" href="${p.page}">${bil('View details →', 'Ver detalhes →')}</a>
    </div>`;
  }).join('\n');
}

function renderPublications() {
  const shown = ['journal', 'conference', 'workshop', 'preprint'];
  const all = publications
    .filter(p => shown.includes(p.type))
    .sort((a, b) => (b.year || 0) - (a.year || 0));
  const featured = all.filter(p => p.featured === true);
  const types = shown.filter(t => all.some(p => p.type === t));
  const tabs = [
    { label: bil('Selected', 'Selecionadas'), html: featured.map(pubCard).join('') },
    { label: bil('All', 'Todas'), html: all.map(pubCard).join('') },
    ...types.map(t => ({
      label: pubTabLabel[t] || (t.charAt(0).toUpperCase() + t.slice(1)),
      html: all.filter(p => p.type === t).map(pubCard).join('')
    }))
  ];
  return tabsHtml(tabs, 'pub-list');
}

function renderAdvisees() {
  return tabsHtml(advisees.current.map(group => ({
    label: levelLabel[group.level] || group.level,
    html:  group.members.map(m => personCard(m, group.level)).join('')
  })));
}

function renderAlumni() {
  const levels = [...new Set(advisees.alumni.map(m => m.level))];
  return tabsHtml(levels.map(lvl => ({
    label: levelLabel[lvl] || lvl,
    html:  advisees.alumni.filter(m => m.level === lvl).map(alumniCard).join('')
  })));
}

// ── Project pages: related publications and related content ──
function renderRelatedPubs(projectId) {
  const typeOrder = { conference: 0, journal: 1, workshop: 2, preprint: 3 };
  const pubs = publications
    .filter(p => p.project === projectId)
    .sort((a, b) => (typeOrder[a.type] ?? 9) - (typeOrder[b.type] ?? 9) || b.year - a.year);
  if (!pubs.length) {
    return `<p class="placeholder-note">${bil('Related publications to be added.', 'Publicações relacionadas em breve.')}</p>`;
  }
  return `<div class="pub-list">${pubs.map(pubCard).join('\n')}</div>`;
}

function renderRelatedLinks(projectId) {
  const typeLabel = {
    poster:  bil('Poster', 'Pôster'),
    article: bil('Article', 'Artigo'),
    talk:    bil('Talk', 'Palestra'),
    video:   bil('Video', 'Vídeo'),
    news:    bil('News', 'Notícia'),
    blog:    bil('Blog', 'Blog'),
    code:    bil('Code', 'Código')
  };
  const host = url => {
    try { return new URL(url).hostname.replace(/^www\./, ''); }
    catch (e) { return url; }
  };
  const items = related.filter(r => r.project === projectId);
  if (!items.length) {
    return `<p class="placeholder-note">${bil('Related content to be added.', 'Conteúdo relacionado em breve.')}</p>`;
  }
  const cards = items.map(r => {
    const meta = [`<span class="related-type related-type--${r.type}">${typeLabel[r.type] || r.type}</span>`];
    if (r.source) meta.push(`<span class="related-source">${r.source}</span>`);
    if (r.lang)   meta.push(`<span class="related-lang">${r.lang.toUpperCase()}</span>`);
    return `<a class="related-card" href="${r.url}" target="_blank" rel="noopener">
      <div class="related-meta">${meta.join('')}</div>
      <p class="related-title">${r.title}</p>
      <span class="related-host">${host(r.url)} ↗</span>
    </a>`;
  }).join('\n');
  return `<div class="related-list">${cards}</div>`;
}

// ── JSON-LD: Person plus one ScholarlyArticle per displayed publication ──
function renderJsonLd() {
  const site = 'https://lucaskup.github.io/';
  const person = {
    '@type': 'Person',
    '@id': site + '#person',
    name: 'Lucas Silveira Kupssinskü',
    jobTitle: 'Associate Professor',
    worksFor: {
      '@type': 'CollegeOrUniversity',
      name: 'Pontifical Catholic University of Rio Grande do Sul (PUCRS)'
    },
    url: site,
    image: site + 'img/perfilpucrs.jpg',
    email: 'mailto:lucas.kupssinsku@pucrs.br',
    sameAs: [
      'https://scholar.google.com/citations?user=Uona1HYAAAAJ',
      'https://orcid.org/0000-0003-2580-3996',
      'https://lattes.cnpq.br/7949995756060059',
      'https://www.pucrs.br/pesquisadores/lucas-silveira-kupssinsku/'
    ]
  };
  const shown = ['journal', 'conference', 'workshop', 'preprint'];
  const articles = publications
    .filter(p => shown.includes(p.type))
    .map(p => {
      const a = {
        '@type': 'ScholarlyArticle',
        headline: p.title,
        author: p.authors.map(n => ({ '@type': 'Person', name: n.replace(/^\*/, '') })),
        datePublished: String(p.year),
        isPartOf: { '@type': 'CreativeWork', name: p.venue }
      };
      if (p.url) a.url = p.url;
      return a;
    });
  const graph = { '@context': 'https://schema.org', '@graph': [person, ...articles] };
  return `<script type="application/ld+json">\n${JSON.stringify(graph, null, 1)}\n</script>`;
}

// ── Assemble ──
function replaceOnce(html, marker, replacement, file) {
  if (!html.includes(marker)) {
    throw new Error(`Marker not found in ${file}: ${marker}`);
  }
  return html.replace(marker, replacement);
}

function fillContainer(html, id, extraAttrs, content, file) {
  const marker = `<div id="${id}"${extraAttrs}></div>`;
  return replaceOnce(html, marker, `<div id="${id}"${extraAttrs}>\n${content}\n</div>`, file);
}

// Copy the source tree, then rewrite the pages that have dynamic sections.
fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT);
const EXCLUDE = new Set(['.git', '.github', '.claude', '_site', 'node_modules', 'build.js', '.gitignore', 'README.md']);
for (const entry of fs.readdirSync(ROOT)) {
  if (EXCLUDE.has(entry)) continue;
  fs.cpSync(path.join(ROOT, entry), path.join(OUT, entry), { recursive: true });
}

function transform(relPath, fn) {
  const file = path.join(OUT, relPath);
  const html = fs.readFileSync(file, 'utf8');
  fs.writeFileSync(file, fn(html, relPath));
  console.log('prerendered ' + relPath);
}

transform('index.html', (html, file) => {
  html = fillContainer(html, 'news-container', '', renderNews(), file);
  html = fillContainer(html, 'project-list', ' class="project-list"', renderProjects(), file);
  html = fillContainer(html, 'publications-container', '', renderPublications(), file);
  html = fillContainer(html, 'advisees-container', '', renderAdvisees(), file);
  html = fillContainer(html, 'alumni-container', '', renderAlumni(), file);
  html = replaceOnce(html, '</head>', renderJsonLd() + '\n</head>', file);
  return html;
});

transform('projects/low-resource-ml.html', (html, file) => {
  html = fillContainer(html, 'related-pubs', '', renderRelatedPubs('low-resource-ml'), file);
  html = fillContainer(html, 'related-links', '', renderRelatedLinks('low-resource-ml'), file);
  return html;
});

transform('projects/misinformation-llms.html', (html, file) => {
  html = fillContainer(html, 'related-pubs', '', renderRelatedPubs('misinformation-llms'), file);
  return html;
});

// ── Sitemap: generated at build time so lastmod never goes stale ──
// Each page's lastmod is the date of the last commit touching any of its
// source inputs (requires full git history; deploy.yml checks out with
// fetch-depth: 0). Falls back to today when git is unavailable.
function renderSitemap() {
  const site = 'https://lucaskup.github.io/';
  const pages = [
    { loc: '',                                  src: ['index.html', 'style.css', 'data/news.json', 'data/projects.json', 'data/publications.json', 'data/advisees.json'] },
    { loc: 'prospective-students.html',         src: ['prospective-students.html'] },
    { loc: 'news/kunumi-colabs-rs.html',        src: ['news/kunumi-colabs-rs.html'] },
    { loc: 'projects/low-resource-ml.html',     src: ['projects/low-resource-ml.html', 'data/publications.json', 'data/related.json'] },
    { loc: 'projects/misinformation-llms.html', src: ['projects/misinformation-llms.html', 'data/publications.json'] }
  ];
  const today = new Date().toISOString().slice(0, 10);
  const lastmod = files => {
    try {
      const out = execSync(`git log -1 --format=%cs -- ${files.join(' ')}`, { cwd: ROOT }).toString().trim();
      return /^\d{4}-\d{2}-\d{2}$/.test(out) ? out : today;
    } catch (e) {
      return today;
    }
  };
  const urls = pages.map(p => `  <url>
    <loc>${site}${p.loc}</loc>
    <lastmod>${lastmod(p.src)}</lastmod>
  </url>`).join('\n');
  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls}
</urlset>\n`;
}

fs.writeFileSync(path.join(OUT, 'sitemap.xml'), renderSitemap());
console.log('generated sitemap.xml');

console.log('build complete → _site/');
