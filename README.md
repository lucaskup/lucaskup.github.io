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
- Internal links in `data/*.json` (`news.json`, `related.json`) are written
  relative to the site root, e.g. `news/bracis-2026-cuiaba.html`. The build
  turns them into absolute URLs and fails if the target page does not exist.
- On every push to `master`, the GitHub Action in `.github/workflows/deploy.yml`
  runs `node build.js` and deploys the generated `_site/` to GitHub Pages.
- To preview locally: `node build.js`, then serve `_site/`
  (for example `python3 -m http.server -d _site`).

Note: opening the repo's `index.html` directly shows empty dynamic sections;
they are filled by the build.

## Adding a news post

1. Pick a slug in the form `event-year-place`, lowercase ASCII, hyphens only
   (e.g. `bracis-2026-cuiaba`). The slug becomes the public URL, so it is
   never renamed after publishing.
2. Copy an existing page in `news/` to `news/<slug>.html` and edit its
   content. Canonical, `og:url` and the JSON-LD `url` are rewritten by the
   build from the file name, so leftovers from the copied page are harmless.
   The cover image (`og:image`, `twitter:image`, JSON-LD `image`) is kept as
   written, since it varies per post.
3. Put the photos in `img/news/<slug>/`, numbered `01.jpg`, `02.jpg`, ...
4. Add the entry to `data/news.json` with `"url": "news/<slug>.html"`, and, if
   it belongs to a project, to `data/related.json`.

The sitemap picks up every `news/*.html` automatically; `build.js` does not
need to change.
