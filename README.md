# lucaskup.github.io

Personal academic site. Plain HTML/CSS/JS with the content data in `data/*.json`.

## How it builds

The dynamic sections (news, projects, publications, advisees, alumni, and the
related content on project pages) are prerendered from the JSON files into
static HTML so crawlers and no-JS visitors see full content. The build also
injects JSON-LD structured data (Person and ScholarlyArticle) into the homepage
and generates `sitemap.xml` with lastmod dates taken from git history.

- Edit content in `data/*.json` as before; the JSON files stay the source of truth.
- `cv.html` is built entirely from `data/cv.json` (education, positions, grants,
  awards, service, talks, teaching) plus `publications.json` and `advisees.json`,
  so the CV never drifts from the rest of the site. Elements tagged
  `cv-full-only` are dropped from the "Short CV (PDF)" print; the citation
  count in `cv.json` (`profile.citations`) is the one number updated by hand.
- On every push to `master`, the GitHub Action in `.github/workflows/deploy.yml`
  runs `node build.js` and deploys the generated `_site/` to GitHub Pages.
- To preview locally: `node build.js`, then serve `_site/`
  (for example `python3 -m http.server -d _site`).

Note: opening the repo's `index.html` directly shows empty dynamic sections;
they are filled by the build.
