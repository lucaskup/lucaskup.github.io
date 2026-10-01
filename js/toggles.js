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
    // Keep the document language in sync so screen readers switch voice.
    root.lang = lang === 'pt' ? 'pt-BR' : 'en';
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
  // sibling .tab-panel divs; this wires up the switching, keeps the ARIA
  // state (aria-selected, roving tabindex) in sync, and adds the arrow-key
  // navigation expected of a tablist.
  document.querySelectorAll('.tabs').forEach((row) => {
    const btns = Array.from(row.querySelectorAll('.tab-btn'));
    const panels = [];
    let el = row.nextElementSibling;
    while (el) {
      if (el.classList.contains('tab-panel')) panels.push(el);
      el = el.nextElementSibling;
    }

    function select(i, focus) {
      btns.forEach((b, j) => {
        const active = j === i;
        b.classList.toggle('active', active);
        b.setAttribute('aria-selected', active ? 'true' : 'false');
        if (active) b.removeAttribute('tabindex');
        else b.setAttribute('tabindex', '-1');
      });
      panels.forEach((p, j) => p.classList.toggle('active', j === i));
      if (focus) btns[i].focus();
    }

    btns.forEach((btn, i) => {
      btn.addEventListener('click', () => select(i, false));
      btn.addEventListener('keydown', (e) => {
        let to = null;
        if (e.key === 'ArrowRight') to = (i + 1) % btns.length;
        else if (e.key === 'ArrowLeft') to = (i - 1 + btns.length) % btns.length;
        else if (e.key === 'Home') to = 0;
        else if (e.key === 'End') to = btns.length - 1;
        if (to !== null) { e.preventDefault(); select(to, true); }
      });
    });
  });
})();

(function initBibtex() {
  // Each .bibtex-btn carries its citation in data-bibtex (prerendered by
  // build.js); clicking copies it and flashes a confirmation on the button.
  const btns = document.querySelectorAll('.bibtex-btn');
  if (!btns.length) return;

  // Legacy path for non-secure contexts (e.g. served over plain http on a
  // LAN address) or when the async Clipboard API rejects.
  function legacyCopy(text) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
    return ok;
  }

  function copy(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text)
        .catch(() => (legacyCopy(text) ? Promise.resolve() : Promise.reject()));
    }
    return legacyCopy(text) ? Promise.resolve() : Promise.reject();
  }

  function flash(btn, cls, label) {
    btn.classList.add(cls);
    btn.textContent = label;
    setTimeout(() => {
      btn.classList.remove(cls);
      btn.textContent = 'BibTeX';
    }, 1500);
  }

  btns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const text = btn.dataset.bibtex || '';
      copy(text).then(() => {
        flash(btn, 'copied', '✓ BibTeX');
      }).catch(() => {
        // Clipboard unavailable: signal the failure and hand the text over
        // in a prompt so it can still be copied manually.
        flash(btn, 'copy-failed', '✕ BibTeX');
        window.prompt('Copy the BibTeX entry below:', text);
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

    // Focusable even with a single image, so Enter can open the lightbox.
    root.tabIndex = 0;

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
      root.dataset.index = index;
      track.style.transform = `translateX(-${index * 100}%)`;
      dots.forEach((d, j) => d.classList.toggle('active', j === index));

      // Sync the shared caption to the active slide, when per-slide captions exist.
      const slide = slides[index];
      if (capEn && slide.dataset.captionEn) capEn.textContent = slide.dataset.captionEn;
      if (capPt && slide.dataset.captionPt) capPt.textContent = slide.dataset.captionPt;
    }

    prev && prev.addEventListener('click', () => go(index - 1));
    next && next.addEventListener('click', () => go(index + 1));

    // The lightbox asks the carousel to follow the photo last viewed there.
    root.addEventListener('carousel:go', (e) => go(e.detail));

    // Arrow-key navigation when the carousel has focus.
    root.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft')  { e.preventDefault(); go(index - 1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); go(index + 1); }
    });

    go(0);
  });
})();

(function initLightbox() {
  // Clicking a photo inside a .project-figure opens it full screen. Photos in
  // the same figure form one gallery: arrows, keys and swipes move through
  // it, and the browser back button closes it (important on phones).
  const figures = Array.from(document.querySelectorAll('.project-figure'))
    .filter((f) => f.querySelector('img'));
  if (!figures.length || typeof HTMLDialogElement !== 'function') return;

  const root = document.documentElement;
  const t = (en, pt) => (root.getAttribute('data-lang') === 'pt' ? pt : en);

  const dlg = document.createElement('dialog');
  dlg.className = 'lightbox';
  // Close comes first so it receives the initial focus when the dialog opens.
  dlg.innerHTML =
    '<button class="lightbox-close" type="button">×</button>' +
    '<img class="lightbox-img" alt="" />' +
    '<p class="lightbox-caption"></p>' +
    '<button class="lightbox-btn lightbox-prev" type="button">‹</button>' +
    '<button class="lightbox-btn lightbox-next" type="button">›</button>';
  document.body.appendChild(dlg);

  const big     = dlg.querySelector('.lightbox-img');
  const caption = dlg.querySelector('.lightbox-caption');
  const prev    = dlg.querySelector('.lightbox-prev');
  const next    = dlg.querySelector('.lightbox-next');
  const close   = dlg.querySelector('.lightbox-close');

  let gallery = [];
  let index = 0;
  let carousel = null;

  function captionFor(img) {
    const slide = img.closest('.carousel-slide');
    const lang = root.getAttribute('data-lang') === 'pt' ? 'Pt' : 'En';
    if (slide && slide.dataset['caption' + lang]) return slide.dataset['caption' + lang];
    const cap = img.closest('figure').querySelector('figcaption .lang-' + lang.toLowerCase());
    return cap ? cap.textContent.trim() : '';
  }

  function show(i) {
    index = (i + gallery.length) % gallery.length;
    const img = gallery[index];
    big.src = img.currentSrc || img.src;
    big.alt = img.alt;
    const text = captionFor(img);
    caption.textContent = gallery.length > 1
      ? (text ? text + ' · ' : '') + (index + 1) + ' / ' + gallery.length
      : text;
  }

  function open(figure, i) {
    gallery = Array.from(figure.querySelectorAll('img'));
    carousel = figure.querySelector('[data-carousel]');
    dlg.toggleAttribute('data-single', gallery.length === 1);
    dlg.setAttribute('aria-label', t('Image viewer', 'Visualizador de imagens'));
    prev.setAttribute('aria-label', t('Previous image', 'Imagem anterior'));
    next.setAttribute('aria-label', t('Next image', 'Próxima imagem'));
    close.setAttribute('aria-label', t('Close', 'Fechar'));
    show(i);
    dlg.showModal();
    root.classList.add('lightbox-open');
    // A history entry lets the phone's back gesture close the viewer
    // instead of leaving the page.
    history.pushState({ lightbox: true }, '');
  }

  figures.forEach((figure) => {
    figure.querySelectorAll('img').forEach((img, i) => {
      img.addEventListener('click', () => open(figure, i));
    });
    const c = figure.querySelector('[data-carousel]');
    if (c) {
      c.addEventListener('keydown', (e) => {
        if (e.target !== c || (e.key !== 'Enter' && e.key !== ' ')) return;
        e.preventDefault();
        open(figure, Number(c.dataset.index || 0));
      });
    }
  });

  dlg.addEventListener('close', () => {
    root.classList.remove('lightbox-open');
    if (carousel) carousel.dispatchEvent(new CustomEvent('carousel:go', { detail: index }));
    if (history.state && history.state.lightbox) history.back();
  });

  window.addEventListener('popstate', () => { if (dlg.open) dlg.close(); });

  prev.addEventListener('click', () => show(index - 1));
  next.addEventListener('click', () => show(index + 1));
  close.addEventListener('click', () => dlg.close());

  // Tapping anywhere outside the photo and the buttons closes the viewer.
  dlg.addEventListener('click', (e) => {
    if (!e.target.closest('button, .lightbox-img')) dlg.close();
  });

  dlg.addEventListener('keydown', (e) => {
    if (gallery.length < 2) return;
    if (e.key === 'ArrowLeft')  { e.preventDefault(); show(index - 1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); show(index + 1); }
  });

  // Horizontal swipe changes photo; ignored while pinch-zoomed so panning
  // a zoomed photo does not skip to the next one.
  let x0 = null, y0 = null;
  dlg.addEventListener('touchstart', (e) => {
    if (e.touches.length !== 1) { x0 = null; return; }
    x0 = e.touches[0].clientX;
    y0 = e.touches[0].clientY;
  }, { passive: true });
  dlg.addEventListener('touchend', (e) => {
    if (x0 === null || gallery.length < 2) return;
    if (window.visualViewport && window.visualViewport.scale > 1.01) return;
    const dx = e.changedTouches[0].clientX - x0;
    const dy = e.changedTouches[0].clientY - y0;
    x0 = null;
    if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy)) show(index + (dx < 0 ? 1 : -1));
  }, { passive: true });
})();
