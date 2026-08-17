/* Shared theme + language + mobile-nav toggles for the project pages.
   Each initializer is a no-op when its elements are absent on the page. */

(function initTheme() {
  const root = document.documentElement;
  const btn  = document.getElementById('theme-toggle');
  if (!btn) return;
  const icon = btn.querySelector('.theme-icon');
  const stored = localStorage.getItem('theme');
  const initial = stored || (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');

  function apply(theme) {
    root.setAttribute('data-theme', theme);
    if (icon) icon.textContent = theme === 'light' ? '☀️' : '🌙';
    btn.setAttribute('aria-label', theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode');
  }

  apply(initial);

  btn.addEventListener('click', () => {
    const next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    apply(next);
    localStorage.setItem('theme', next);
  });
})();

(function initLanguage() {
  const root = document.documentElement;
  const btn  = document.getElementById('lang-toggle');
  if (!btn) return;
  const label = btn.querySelector('.lang-label');
  const stored = localStorage.getItem('lang');
  const initial = stored || 'en';

  function apply(lang) {
    root.setAttribute('data-lang', lang);
    // Button shows the language you can switch TO.
    if (label) label.textContent = lang === 'en' ? 'PT' : 'EN';
    btn.setAttribute('aria-label', lang === 'en' ? 'Mudar para português' : 'Switch to English');
    btn.setAttribute('title', lang === 'en' ? 'Mudar para português' : 'Switch to English');
  }

  apply(initial);

  btn.addEventListener('click', () => {
    const next = root.getAttribute('data-lang') === 'en' ? 'pt' : 'en';
    apply(next);
    localStorage.setItem('lang', next);
  });
})();

(function initNav() {
  const toggle = document.getElementById('nav-toggle');
  const links  = document.getElementById('nav-links');
  if (!toggle || !links) return;

  function setOpen(open) {
    links.classList.toggle('open', open);
    toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
  }

  toggle.addEventListener('click', () => {
    setOpen(!links.classList.contains('open'));
  });

  links.addEventListener('click', (e) => {
    if (e.target.tagName === 'A') setOpen(false);
  });
})();

(function initTabs() {
  // Tab groups are prerendered by build.js as a .tabs button row followed by
  // sibling .tab-panel divs; this only wires up the switching.
  document.querySelectorAll('.tabs').forEach((row) => {
    const btns = Array.from(row.querySelectorAll('.tab-btn'));
    const panels = [];
    let el = row.nextElementSibling;
    while (el) {
      if (el.classList.contains('tab-panel')) panels.push(el);
      el = el.nextElementSibling;
    }
    btns.forEach((btn, i) => {
      btn.addEventListener('click', () => {
        btns.forEach(b => b.classList.remove('active'));
        panels.forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        if (panels[i]) panels[i].classList.add('active');
      });
    });
  });
})();

(function initNewsToggle() {
  // The button's show/hide labels are both in the markup; CSS picks one
  // based on the container's news-open class.
  const container = document.getElementById('news-container');
  const btn = container && container.querySelector('.news-toggle');
  if (!btn) return;
  btn.addEventListener('click', () => container.classList.toggle('news-open'));
})();

(function initCarousels() {
  const carousels = document.querySelectorAll('[data-carousel]');
  if (!carousels.length) return;

  carousels.forEach((root) => {
    const track  = root.querySelector('.carousel-track');
    const slides = Array.from(root.querySelectorAll('.carousel-slide'));
    const prev   = root.querySelector('.carousel-prev');
    const next   = root.querySelector('.carousel-next');
    const dotsEl = root.querySelector('.carousel-dots');
    const figure = root.closest('figure');
    const capEn  = figure && figure.querySelector('figcaption .lang-en');
    const capPt  = figure && figure.querySelector('figcaption .lang-pt');
    if (!track || slides.length === 0) return;

    // A single image needs no navigation chrome.
    if (slides.length === 1) { root.setAttribute('data-single', ''); return; }

    let index = 0;

    const dots = slides.map((_, i) => {
      const dot = document.createElement('button');
      dot.type = 'button';
      dot.className = 'carousel-dot';
      dot.setAttribute('aria-label', `Go to image ${i + 1}`);
      dot.addEventListener('click', () => go(i));
      dotsEl && dotsEl.appendChild(dot);
      return dot;
    });

    function go(i) {
      index = (i + slides.length) % slides.length; // wrap around
      track.style.transform = `translateX(-${index * 100}%)`;
      dots.forEach((d, j) => d.classList.toggle('active', j === index));

      // Sync the shared caption to the active slide, when per-slide captions exist.
      const slide = slides[index];
      if (capEn && slide.dataset.captionEn) capEn.textContent = slide.dataset.captionEn;
      if (capPt && slide.dataset.captionPt) capPt.textContent = slide.dataset.captionPt;
    }

    prev && prev.addEventListener('click', () => go(index - 1));
    next && next.addEventListener('click', () => go(index + 1));

    // Arrow-key navigation when the carousel has focus.
    root.tabIndex = 0;
    root.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft')  { e.preventDefault(); go(index - 1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); go(index + 1); }
    });

    go(0);
  });
})();
