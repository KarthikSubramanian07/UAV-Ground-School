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
      input.setAttribute("aria-valuetext", `${100 - value}% full pipeline`);
    };
    input.addEventListener("input", update);
    update();
  });
})();
