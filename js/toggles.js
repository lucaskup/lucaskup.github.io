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
