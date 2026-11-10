/* Before/after slider. The native range input provides keyboard, touch and
   screen reader support; this script only mirrors its value into CSS. */
(function () {
  "use strict";

  document.querySelectorAll(".compare").forEach((compare) => {
    const input = compare.querySelector('input[type="range"]');
    if (!input) return;
    const update = () => {
      const value = Number(input.value);
      compare.style.setProperty("--pos", `${value}%`);
      input.setAttribute("aria-valuetext", `${100 - value}% ${input.dataset.right || "full pipeline"}`);
    };
    input.addEventListener("input", update);
    update();
  });
})();

/* Keyboard access to scrolling boxes (WCAG 2.1.1). Wide tables, terminal
   transcripts and event logs scroll inside themselves on small screens; a box
   that scrolls gets tabindex="0" so arrow keys can scroll it, and loses it again
   when it fits, so desktop readers do not tab through boxes that never scroll. */
(function () {
  "use strict";

  const SELECTOR = ".table-scroll, pre, .event-log, .hexgrid, .field-tree, .packet-detail";
  const sync = () => {
    document.querySelectorAll(SELECTOR).forEach((el) => {
      const scrolls = el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1;
      const style = getComputedStyle(el);
      const canScroll = /auto|scroll/.test(style.overflowX + style.overflowY);
      if (scrolls && canScroll && !el.hasAttribute("tabindex")) {
        el.setAttribute("tabindex", "0");
        el.dataset.scrollFocus = "";
      } else if (!(scrolls && canScroll) && "scrollFocus" in el.dataset) {
        el.removeAttribute("tabindex");
        delete el.dataset.scrollFocus;
      }
    });
  };
  let queued = false;
  const queue = () => {
    if (queued) return;
    queued = true;
    setTimeout(() => {
      queued = false;
      sync();
    }, 200);
  };
  sync();
  window.addEventListener("resize", queue);
  window.addEventListener("load", queue);
  if (document.fonts) document.fonts.ready.then(queue);
  new MutationObserver(queue).observe(document.body, { childList: true, subtree: true });
})();
