const menuButton = document.querySelector(".menu-button");
const navigation = document.querySelector(".site-nav");

menuButton?.addEventListener("click", () => {
  const isOpen = menuButton.getAttribute("aria-expanded") === "true";
  menuButton.setAttribute("aria-expanded", String(!isOpen));
  navigation?.classList.toggle("is-open", !isOpen);
});

navigation?.addEventListener("click", (event) => {
  if (event.target.closest("a")) {
    menuButton?.setAttribute("aria-expanded", "false");
    navigation.classList.remove("is-open");
  }
});

const tocLinks = [...document.querySelectorAll(".toc-link")];
const tocSections = tocLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

const setActiveTocLink = (sectionId) => {
  tocLinks.forEach((link) => {
    const isActive = link.getAttribute("href") === `#${sectionId}`;
    link.classList.toggle("is-active", isActive);
    if (isActive) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  });
};

let tocUpdatePending = false;
const updateActiveToc = () => {
  const marker = window.scrollY + Math.min(window.innerHeight * 0.3, 260);
  let activeSection = tocSections[0];

  tocSections.forEach((section) => {
    if (section.offsetTop <= marker) activeSection = section;
  });
  if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) {
    activeSection = tocSections.at(-1);
  }
  if (activeSection) setActiveTocLink(activeSection.id);
  tocUpdatePending = false;
};

window.addEventListener("scroll", () => {
  if (!tocUpdatePending) {
    window.requestAnimationFrame(updateActiveToc);
    tocUpdatePending = true;
  }
}, { passive: true });
window.addEventListener("resize", updateActiveToc);
updateActiveToc();

tocLinks.forEach((link) => {
  link.addEventListener("click", () => setActiveTocLink(link.getAttribute("href").slice(1)));
});

const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12 },
);

document.querySelectorAll(".reveal").forEach((element) => revealObserver.observe(element));

const resultTabs = document.querySelectorAll("[data-results-tab]");
const resultPanels = document.querySelectorAll("[data-results-panel]");

resultTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const selectedPanel = tab.dataset.resultsTab;

    resultTabs.forEach((item) => {
      item.setAttribute("aria-selected", String(item === tab));
    });
    resultPanels.forEach((panel) => {
      panel.hidden = panel.id !== selectedPanel;
    });
  });
});

const chartRoot = document.querySelector("[data-efficiency-chart]");

if (chartRoot) {
  const svg = chartRoot.querySelector("[data-chart-svg]");
  const legend = chartRoot.querySelector("[data-chart-legend]");
  const tooltip = chartRoot.querySelector("[data-chart-tooltip]");
  const svgNamespace = "http://www.w3.org/2000/svg";
  const width = 920;
  const height = 470;
  const plot = { left: 68, right: 890, top: 28, bottom: 408 };
  const xMin = 50;
  const xMax = 820;
  const yMin = 1.45;
  const yMax = 2.65;

  // Checkpoints shown in Figure 1; final reported values follow Table 1.
  const series = [
    {
      id: "pixeldit",
      name: "PixelDiT-XL",
      color: "#91a1bf",
      points: [[80, 2.36], [160, 1.97], [320, 1.61], [800, 1.54]],
    },
    {
      id: "pixeldit-egfm",
      name: "PixelDiT-XL + EG-FM",
      color: "#07357d",
      points: [[80, 1.99], [140, 1.79], [180, 1.63], [200, 1.55]],
    },
    {
      id: "deco",
      name: "DeCo-XL",
      color: "#97c3c4",
      points: [[80, 2.57], [320, 1.90], [600, 1.69]],
    },
    {
      id: "deco-egfm",
      name: "DeCo-XL + EG-FM",
      color: "#148b8b",
      points: [[80, 2.35], [160, 2.08], [320, 1.71], [440, 1.63]],
    },
  ];

  const makeSvgElement = (tag, attributes = {}) => {
    const element = document.createElementNS(svgNamespace, tag);
    Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, value));
    return element;
  };

  const xScale = (epoch) => plot.left + ((epoch - xMin) / (xMax - xMin)) * (plot.right - plot.left);
  const yScale = (fid) => plot.bottom - ((fid - yMin) / (yMax - yMin)) * (plot.bottom - plot.top);

  const gridLayer = makeSvgElement("g");
  const seriesLayer = makeSvgElement("g");
  svg.append(gridLayer, seriesLayer);

  [1.6, 1.8, 2.0, 2.2, 2.4, 2.6].forEach((value) => {
    const y = yScale(value);
    gridLayer.append(makeSvgElement("line", { x1: plot.left, x2: plot.right, y1: y, y2: y, class: "chart-grid" }));
    const label = makeSvgElement("text", { x: plot.left - 13, y: y + 4, "text-anchor": "end", class: "chart-tick" });
    label.textContent = value.toFixed(1);
    gridLayer.append(label);
  });

  [100, 200, 300, 400, 500, 600, 700, 800].forEach((value) => {
    const x = xScale(value);
    gridLayer.append(makeSvgElement("line", { x1: x, x2: x, y1: plot.top, y2: plot.bottom, class: "chart-grid" }));
    const label = makeSvgElement("text", { x, y: plot.bottom + 24, "text-anchor": "middle", class: "chart-tick" });
    label.textContent = value;
    gridLayer.append(label);
  });

  gridLayer.append(
    makeSvgElement("line", { x1: plot.left, x2: plot.left, y1: plot.top, y2: plot.bottom, class: "chart-axis" }),
    makeSvgElement("line", { x1: plot.left, x2: plot.right, y1: plot.bottom, y2: plot.bottom, class: "chart-axis" }),
  );

  const xLabel = makeSvgElement("text", { x: (plot.left + plot.right) / 2, y: height - 11, "text-anchor": "middle", class: "chart-axis-label" });
  xLabel.textContent = "Training epochs";
  const yLabel = makeSvgElement("text", { x: 17, y: (plot.top + plot.bottom) / 2, "text-anchor": "middle", transform: `rotate(-90 17 ${(plot.top + plot.bottom) / 2})`, class: "chart-axis-label" });
  yLabel.textContent = "FID ↓";
  gridLayer.append(xLabel, yLabel);

  const targetY = yScale(1.55);
  gridLayer.append(makeSvgElement("line", { x1: xScale(200), x2: xScale(800), y1: targetY, y2: targetY, class: "chart-target" }));
  const targetLabel = makeSvgElement("text", { x: xScale(510), y: targetY + 10, "text-anchor": "middle", "dominant-baseline": "hanging", class: "chart-target-label" });
  targetLabel.textContent = "≈4× faster";
  gridLayer.append(targetLabel);

  const hideTooltip = () => { tooltip.hidden = true; };
  const showTooltip = (item, epoch, fid, x, y) => {
    const svgRect = svg.getBoundingClientRect();
    tooltip.innerHTML = `<strong>${item.name}</strong>Epoch ${epoch} · FID ${fid.toFixed(2)}`;
    tooltip.style.left = `${(x / width) * svgRect.width}px`;
    tooltip.style.top = `${(y / height) * svgRect.height}px`;
    tooltip.hidden = false;
  };

  series.forEach((item) => {
    const group = makeSvgElement("g", { class: "chart-series", "data-series": item.id });
    const pathData = item.points.map(([epoch, fid], index) => `${index ? "L" : "M"} ${xScale(epoch)} ${yScale(fid)}`).join(" ");
    group.append(makeSvgElement("path", { d: pathData, stroke: item.color, class: "chart-line" }));

    item.points.forEach(([epoch, fid]) => {
      const x = xScale(epoch);
      const y = yScale(fid);
      const point = makeSvgElement("circle", {
        cx: x,
        cy: y,
        r: 7,
        fill: item.color,
        class: "chart-point",
        tabindex: "0",
        role: "button",
        "aria-label": `${item.name}, epoch ${epoch}, FID ${fid.toFixed(2)}`,
      });
      point.addEventListener("mouseenter", () => showTooltip(item, epoch, fid, x, y));
      point.addEventListener("mouseleave", hideTooltip);
      point.addEventListener("focus", () => showTooltip(item, epoch, fid, x, y));
      point.addEventListener("blur", hideTooltip);
      point.addEventListener("click", () => showTooltip(item, epoch, fid, x, y));
      group.append(point);
    });

    seriesLayer.append(group);

    const control = document.createElement("button");
    control.type = "button";
    control.textContent = item.name;
    control.style.setProperty("--series-color", item.color);
    control.setAttribute("aria-pressed", "true");
    control.addEventListener("click", () => {
      const isVisible = control.getAttribute("aria-pressed") === "true";
      control.setAttribute("aria-pressed", String(!isVisible));
      group.classList.toggle("is-hidden", isVisible);
      hideTooltip();
    });
    legend.append(control);
  });

  svg.addEventListener("mouseleave", hideTooltip);
  svg.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideTooltip();
  });
}
