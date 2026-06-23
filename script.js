const navLinks = Array.from(document.querySelectorAll('.site-nav a'));
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute('href')))
  .filter(Boolean);

const setActiveLink = () => {
  const scrollY = window.scrollY + 120;
  let activeId = '';

  for (const section of sections) {
    if (section.offsetTop <= scrollY) {
      activeId = section.id;
    }
  }

  navLinks.forEach((link) => {
    const isActive = link.getAttribute('href') === `#${activeId}`;
    link.classList.toggle('active', isActive);
  });
};

window.addEventListener('scroll', setActiveLink, { passive: true });
window.addEventListener('load', setActiveLink);

for (const button of document.querySelectorAll('[data-copy-target]')) {
  button.addEventListener('click', async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;

    try {
      await navigator.clipboard.writeText(target.textContent);
      const previous = button.textContent;
      button.textContent = 'Copied';
      setTimeout(() => {
        button.textContent = previous;
      }, 1600);
    } catch {
      button.textContent = 'Copy failed';
    }
  });
}

const lightbox = document.getElementById('lightbox');
const lightboxImage = document.getElementById('lightbox-image');
const closeLightboxButton = document.querySelector('.lightbox-close');

const openLightbox = (src, alt) => {
  lightboxImage.src = src;
  lightboxImage.alt = alt || '';
  lightbox.classList.add('is-open');
  lightbox.setAttribute('aria-hidden', 'false');
  document.body.classList.add('lightbox-open');
};

const closeLightbox = () => {
  lightbox.classList.remove('is-open');
  lightbox.setAttribute('aria-hidden', 'true');
  document.body.classList.remove('lightbox-open');
  lightboxImage.src = '';
};

for (const preview of document.querySelectorAll('[data-preview-src]')) {
  preview.addEventListener('click', () => {
    openLightbox(preview.dataset.previewSrc, preview.dataset.previewAlt);
  });
}

closeLightboxButton.addEventListener('click', closeLightbox);
lightbox.addEventListener('click', (event) => {
  if (event.target === lightbox) {
    closeLightbox();
  }
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && lightbox.classList.contains('is-open')) {
    closeLightbox();
  }
});
