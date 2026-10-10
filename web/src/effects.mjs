// Scroll-driven effects for theme.css: a shadow under the header once the page
// scrolls, and sections that rise into view. Purely presentational; content stays
// visible when JavaScript, IntersectionObserver or motion is unavailable.
const REVEAL =
  ".charge-panel, .stake-panel, .innocence-section, .ledger-row, .court-library, " +
  ".collection-research-link, .rail-panel, .source-record-row, .outcome-card, .court-cards > article";

export function startEffects() {
  if (typeof window === "undefined") return;
  const header = () => document.querySelector(".product-header");
  const onScroll = () =>
    header()?.classList.toggle("is-scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  if (
    !("IntersectionObserver" in window) ||
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  )
    return;
  const io = new IntersectionObserver(
    (entries) =>
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("is-visible");
          io.unobserve(e.target);
        }
      }),
    { rootMargin: "0px 0px -40px 0px" },
  );
  const scan = () =>
    document.querySelectorAll(REVEAL).forEach((el) => {
      if (el.dataset.reveal) return;
      el.dataset.reveal = "1";
      // Only animate what starts below the fold, so the first screen never flickers.
      if (el.getBoundingClientRect().top < window.innerHeight) return;
      el.classList.add("reveal");
      io.observe(el);
    });
  new MutationObserver(scan).observe(document.body, {
    childList: true,
    subtree: true,
  });
  scan();
}
