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
