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

// ── CV page ──
// Everything on cv.html comes from data/cv.json plus the publication and
// advisee files, so the CV cannot drift from the rest of the site. Elements
// tagged .cv-full-only are hidden when printing the short version.
const cv = read('cv.json');

const present = bil('present', 'atual');
const period = (s, e) => (e === null || e === undefined) ? `${s} – ${present}` : (s === e ? `${s}` : `${s} – ${e}`);
const FULL = ' cv-full-only';

function cvSection(id, en, pt, body, fullOnly = false) {
  return `<div class="cv-section${fullOnly ? FULL : ''}" id="${id}">
    <h2 class="cv-h">${bil(en, pt)}</h2>
    ${body}
  </div>`;
}

function cvItem({ when, title, sub, detail, cls = '' }) {
  return `<div class="cv-item${cls}">
    <div class="cv-when">${when}</div>
    <div class="cv-what">
      <p class="cv-title">${title}</p>
      ${sub ? `<p class="cv-sub">${sub}</p>` : ''}
      ${detail ? `<p class="cv-detail">${detail}</p>` : ''}
    </div>
  </div>`;
}

const cvList = (items, cls = '') => `<ul class="cv-list${cls}">${items.map(i => `<li>${i}</li>`).join('')}</ul>`;
const cvSub = (en, pt, cls = '') => `<h3 class="cv-h3${cls}">${bil(en, pt)}</h3>`;

const CV_PUB_TYPES = ['journal', 'conference', 'workshop', 'preprint'];
const cvPubs = publications.filter(p => CV_PUB_TYPES.includes(p.type));
const cvOngoing = advisees.current.reduce((n, g) => n + g.members.length, 0);

function renderCvHeader() {
  const p = cv.profile;
  const L = p.links;
  const nJournal = cvPubs.filter(x => x.type === 'journal').length;
  const nConf = cvPubs.filter(x => x.type === 'conference' || x.type === 'workshop').length;
  const cit = p.citations;
  const citDate = new Date(cit.date + 'T00:00:00');
  const citWhen = bil(
    citDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' }),
    citDate.toLocaleDateString('pt-BR', { month: 'short', year: 'numeric' })
  );
  const stats = [
    [nJournal, bil('journal articles', 'artigos em periódicos')],
    [nConf, bil('conference papers', 'artigos em conferências')],
    [cit.count, bil(`citations · ${cit.source}, ${citWhen}`, `citações · ${cit.source}, ${citWhen}`)],
    [advisees.alumni.length, bil('supervisions completed', 'orientações concluídas')],
    [cvOngoing, bil('supervisions ongoing', 'orientações em andamento')],
    [cv.grants.length, bil('funded projects', 'projetos financiados')]
  ].map(([n, label]) => `<div class="cv-stat"><span class="cv-stat-n">${n}</span><span class="cv-stat-label">${label}</span></div>`).join('');

  const updated = new Date(cv.updated + 'T00:00:00');
  const updatedLabel = bil(
    updated.toLocaleDateString('en-US', { month: 'long', year: 'numeric' }),
    updated.toLocaleDateString('pt-BR', { month: 'long', year: 'numeric' })
  );

  const links = [
    [L.website, 'lucaskup.github.io'],
    [L.scholar, 'Google Scholar'],
    [L.orcid, 'ORCID'],
    [L.lattes, bil('Lattes CV', 'Currículo Lattes')],
    [L.pucrs, bil('PUCRS profile', 'Perfil PUCRS')]
  ].map(([href, label]) => `<a href="${href}" target="_blank" rel="noopener">${label}</a>`).join(' · ');

  return `<header class="cv-header">
    <div class="cv-header-main">
      <h1>${p.name}</h1>
      <p class="cv-role">${bil(p.title_en, p.title_pt)}</p>
      <p class="cv-affil">${bil(p.affiliation_en, p.affiliation_pt)}</p>
      <p class="cv-contact"><a href="mailto:${p.email}">${p.email}</a> · ${p.phone} · ${bil(p.address_en, p.address_pt)}</p>
      <p class="cv-links">${links}</p>
    </div>
    <div class="cv-actions">
      <button type="button" class="btn btn-cv" data-cv-print="full">⬇ ${bil('Full CV (PDF)', 'CV completo (PDF)')}</button>
      <button type="button" class="btn btn-cv" data-cv-print="short">⬇ ${bil('Short CV (PDF)', 'CV resumido (PDF)')}</button>
      <a class="btn btn-lattes" href="${L.lattes}" target="_blank" rel="noopener">📄 Lattes</a>
    </div>
    <p class="cv-updated">${bil('Last updated', 'Última atualização')}: ${updatedLabel}</p>
  </header>
  <p class="cv-mission">${bil(p.mission_en, p.mission_pt)}</p>
  <div class="cv-stats">${stats}</div>`;
}

function renderCvSummary() {
  const p = cv.profile;
  const tags = (en, pt) => `<div class="cv-interests">${en.map((t, i) => `<span class="tag">${bil(t, pt[i])}</span>`).join('')}</div>`;
  return cvSection('summary', 'Summary', 'Resumo',
    `<p class="cv-text">${bil(p.summary_en, p.summary_pt)}</p>
     ${tags(p.interests_en, p.interests_pt)}`);
}

function renderCvEducation() {
  // The short CV shows the Ph.D. only.
  return cvSection('education', 'Education', 'Formação', cv.education.map(e => cvItem({
    when: period(e.start, e.end),
    title: bil(e.degree_en, e.degree_pt),
    sub: e.institution,
    detail: (e.detail_en || e.detail_pt) ? bil(e.detail_en, e.detail_pt) : '',
    cls: /^Ph\.D\./.test(e.degree_en) ? '' : FULL
  })).join(''));
}

function renderCvPositions() {
  const academic = cv.positions.filter(p => p.kind === 'academic');
  const industry = cv.positions.filter(p => p.kind === 'industry');
  const item = (p, cls) => cvItem({
    when: period(p.start, p.end),
    title: bil(p.title_en, p.title_pt),
    sub: bil(p.institution, p.institution_pt || p.institution),
    detail: bil(p.detail_en, p.detail_pt),
    cls
  });
  return cvSection('positions', 'Positions', 'Atuação Profissional',
    cvSub('Academic', 'Acadêmica') + academic.map(p => item(p, '')).join('') +
    cvSub('Industry', 'Indústria', FULL) + industry.map(p => item(p, FULL)).join(''));
}

function renderCvGrants() {
  return cvSection('grants', 'Grants and Funded Projects', 'Projetos Financiados', cv.grants.map(g => {
    const title = g.page ? `<a href="${g.page}">${bil(g.title_en, g.title_pt)}</a>` : bil(g.title_en, g.title_pt);
    const funder = g.funder_pt ? bil(g.funder, g.funder_pt) : g.funder;
    return cvItem({
      when: period(g.start, g.end),
      title,
      sub: `${bil(g.role_en, g.role_pt)} · ${funder}`,
      detail: (g.detail_en || g.detail_pt) ? bil(g.detail_en, g.detail_pt) : '',
      cls: g.end === null ? '' : FULL
    });
  }).join(''));
}

function cvPubEntry(p) {
  const authors = p.authors.map(a => a.startsWith('*') ? `<strong>${a.slice(1)}</strong>` : a).join(', ');
  const title = p.url ? `<a href="${p.url}" target="_blank" rel="noopener">${p.title}</a>` : p.title;
  return `<li class="cv-pub${p.featured ? ' cv-pub-featured' : FULL}">${authors}. <span class="cv-pub-title">${title}</span>. <em>${p.venue}</em>, ${p.year}.</li>`;
}

function renderCvPublications() {
  const groups = [
    ['journal',    'Journal articles',   'Artigos em periódicos'],
    ['conference', 'Conference papers',  'Artigos em conferências'],
    ['workshop',   'Workshop papers',    'Artigos em workshops'],
    ['preprint',   'Preprints',          'Preprints'],
    ['thesis',     'Theses',             'Teses e dissertações'],
    ['other',      'Other',              'Outras produções']
  ];
  const byYear = (a, b) => (b.year || 0) - (a.year || 0);
  const body = groups.map(([type, en, pt]) => {
    const items = publications.filter(p => p.type === type).sort(byYear);
    if (!items.length) return '';
    return cvSub(`${en} (${items.length})`, `${pt} (${items.length})`, FULL) +
      `<ol class="cv-pubs">${items.map(cvPubEntry).join('')}</ol>`;
  }).join('');
  const intro = `<p class="cv-note${FULL}">${bil(
    'BibTeX entries are available on the <a href="index.html#publications">publications page</a>. The short CV lists selected publications only.',
    'As entradas BibTeX estão na <a href="index.html#publications">página de publicações</a>. O CV resumido lista apenas publicações selecionadas.'
  )}</p>`;
  const selected = cvSub('Selected publications', 'Publicações selecionadas', ' cv-short-only');
  return cvSection('publications', 'Publications', 'Publicações', intro + selected + body);
}

function renderCvSupervision() {
  const countBy = (list) => list.reduce((m, x) => { m[x.level] = (m[x.level] || 0) + 1; return m; }, {});
  const fmt = counts => Object.entries(counts).map(([lvl, n]) => `${n} ${levelLabel[lvl] || lvl}`).join(', ');
  const currentFlat = advisees.current.flatMap(g => g.members.map(m => ({ ...m, level: g.level })));
  const summary = `<p class="cv-text">${bil(
    `<strong>${advisees.alumni.length} completed</strong> (${fmt(countBy(advisees.alumni))}) and <strong>${cvOngoing} ongoing</strong> (${fmt(countBy(currentFlat))}).`,
    `<strong>${advisees.alumni.length} concluídas</strong> (${fmt(countBy(advisees.alumni))}) e <strong>${cvOngoing} em andamento</strong> (${fmt(countBy(currentFlat))}).`
  )}</p>`;
  const person = (m, when) => `<strong>${m.name}</strong>${m.coadvisee ? coadviseeTag : ''}${m.topic && m.topic !== 'A definir' ? `, ${m.topic}` : ''} <span class="cv-muted">(${when})</span>`;
  const ongoingBody = advisees.current.filter(g => g.members.length).map(g =>
    `<p class="cv-list-label${FULL}">${levelLabel[g.level] || g.level}</p>` +
    cvList(g.members.map(m => person(m, `${bil('since', 'desde')} ${m.start}`)), FULL)
  ).join('');
  const levels = [...new Set(advisees.alumni.map(m => m.level))];
  const completedBody = levels.map(lvl =>
    `<p class="cv-list-label${FULL}">${levelLabel[lvl] || lvl}</p>` +
    cvList(advisees.alumni.filter(m => m.level === lvl).map(m => person(m, `${m.start}–${m.finish}`)), FULL)
  ).join('');
  return cvSection('supervision', 'Supervision', 'Orientações',
    summary +
    cvSub('Ongoing', 'Em andamento', FULL) + ongoingBody +
    cvSub('Completed', 'Concluídas', FULL) + completedBody);
}

function renderCvTeaching() {
  const levelCls = { grad: 'level-grad', under: 'level-under', tech: 'level-tech' };
  const levelName = {
    grad:  bil('Graduate', 'Pós-graduação'),
    under: bil('Undergraduate', 'Graduação'),
    tech:  bil('Technical', 'Técnico')
  };
  const rows = cv.teaching.map(t => `<div class="cv-item">
    <div class="cv-when">${bil(t.period, t.period_pt)}</div>
    <div class="cv-what">
      <p class="cv-title">${bil(t.institution_en, t.institution_pt)} <span class="lecture-level ${levelCls[t.level]}">${levelName[t.level]}</span></p>
      <p class="cv-detail${FULL}">${bil(t.courses_en.join(' · '), t.courses_pt.join(' · '))}</p>
    </div>
  </div>`).join('');
  const materials = cvSub('Teaching materials', 'Materiais didáticos', FULL) +
    cvList(cv.teaching_materials.map(m => `<span class="cv-year">${m.year}</span> ${bil(m.en, m.pt)}`), FULL);
  return cvSection('teaching', 'Teaching', 'Ensino', rows + materials);
}

function renderCvAwards() {
  // The short CV keeps only the last five years.
  const cutoff = new Date().getFullYear() - 5;
  const items = cv.awards.map(a =>
    `<li${a.year >= cutoff ? '' : ` class="${FULL.trim()}"`}><span class="cv-year">${a.year}</span> ${bil(a.en, a.pt)}</li>`);
  return cvSection('awards', 'Awards and Honors', 'Prêmios e Títulos', `<ul class="cv-list">${items.join('')}</ul>`);
}

function renderCvService() {
  const s = cv.service;
  const yrs = y => y.join(', ');
  const editorial = cvList(s.editorial.map(e => `<span class="cv-year">${period(e.start, e.end)}</span> ${bil(e.en, e.pt)}`));
  const pcs = cvList(s.program_committees.map(c =>
    `${c.venue} <span class="cv-muted">(${yrs(c.years)}${c.note_en ? '; ' + bil(c.note_en, c.note_pt) : ''})</span>`), FULL);
  const reviewing = cvList(s.reviewing.map(r => `${r.venue} <span class="cv-muted">(${bil('since', 'desde')} ${r.since})</span>`), FULL);
  const org = cvList(s.organization.map(o => `<span class="cv-year">${o.year}</span> ${bil(o.en, o.pt)}`), FULL);
  const committees = cvList(s.thesis_committees.map(c =>
    `<strong>${c.count}</strong> ${bil(c.level_en, c.level_pt)} <span class="cv-muted">(${bil(c.detail_en, c.detail_pt)})</span>`), FULL);
  const inst = cvList(s.institutional.map(i => `<span class="cv-year">${period(i.start, i.end)}</span> ${bil(i.en, i.pt)}`), FULL);
  return cvSection('service', 'Service', 'Atividades de Serviço',
    cvSub('Editorial boards', 'Corpo editorial') + editorial +
    cvSub('Program committees', 'Comitês de programa', FULL) + pcs +
    cvSub('Journal reviewing', 'Revisão para periódicos', FULL) + reviewing +
    cvSub('Event organization', 'Organização de eventos', FULL) + org +
    cvSub('Thesis and examination committees', 'Participação em bancas', FULL) + committees +
    cvSub('Institutional service', 'Atividades institucionais', FULL) + inst);
}

function renderCvTalks() {
  const talks = cvList(cv.talks.map(t => `<span class="cv-year">${t.year}</span> ${bil(t.en, t.pt)}`));
  const media = cvList(cv.media.map(m => `<span class="cv-year">${m.year}</span> ${bil(m.en, m.pt)}`));
  return cvSection('talks', 'Talks and Media', 'Palestras e Mídia',
    cvSub('Invited talks and tutorials', 'Palestras e tutoriais') + talks +
    cvSub('Interviews', 'Entrevistas') + media, true);
}

function renderCvExtension() {
  return cvSection('outreach', 'Outreach and Extension', 'Extensão', cv.extension.map(e => cvItem({
    when: e.year,
    title: bil(e.title_en, e.title_pt),
    detail: bil(e.detail_en, e.detail_pt)
  })).join(''), true);
}

function renderCvLanguages() {
  return cvSection('languages', 'Languages', 'Idiomas', cvList(cv.languages.map(l => bil(l.en, l.pt))), true);
}

const CV_SECTIONS = [
  ['summary',      'Summary',                    'Resumo'],
  ['education',    'Education',                  'Formação'],
  ['positions',    'Positions',                  'Atuação Profissional'],
  ['grants',       'Grants and Funded Projects', 'Projetos Financiados'],
  ['publications', 'Publications',               'Publicações'],
  ['supervision',  'Supervision',                'Orientações'],
  ['teaching',     'Teaching',                   'Ensino'],
  ['awards',       'Awards and Honors',          'Prêmios e Títulos'],
  ['service',      'Service',                    'Atividades de Serviço'],
  ['talks',        'Talks and Media',            'Palestras e Mídia'],
  ['outreach',     'Outreach and Extension',     'Extensão'],
  ['languages',    'Languages',                  'Idiomas']
];

function renderCvToc() {
  return `<ul>${CV_SECTIONS.map(([id, en, pt]) => `<li><a href="#${id}">${bil(en, pt)}</a></li>`).join('')}</ul>`;
}

function renderCv() {
  return [
    renderCvHeader(),
    renderCvSummary(),
    renderCvEducation(),
    renderCvPositions(),
    renderCvGrants(),
    renderCvPublications(),
    renderCvSupervision(),
    renderCvTeaching(),
    renderCvAwards(),
    renderCvService(),
    renderCvTalks(),
    renderCvExtension(),
    renderCvLanguages()
  ].join('\n');
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

transform('cv.html', (html, file) => {
  html = fillContainer(html, 'cv-toc', '', renderCvToc(), file);
  html = fillContainer(html, 'cv-container', '', renderCv(), file);
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
    { loc: 'cv.html',                           src: ['cv.html', 'style.css', 'data/cv.json', 'data/publications.json', 'data/advisees.json'] },
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
