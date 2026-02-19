window.MathJax = {
  loader: {load: ["[tex]/boldsymbol"]},
  tex: {
    packages: {"[+]": ["boldsymbol"]},
    macros: {
      bm: ["{\\boldsymbol{#1}}", 1],
    },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
