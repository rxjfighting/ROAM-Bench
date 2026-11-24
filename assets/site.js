"use strict";
(() => {
  const project = window.ROAM_PROJECT;
  const authorRoot = document.getElementById("authors");
  project.authors.forEach(author => {
    const item = document.createElement("span"); item.className = "author";
    item.append(document.createTextNode(author.name));
    const affiliation = document.createElement("sup"); affiliation.textContent = author.affiliations.join(",") + (author.marker || "");
    item.append(affiliation); authorRoot.append(item);
  });
  const affiliationRoot = document.getElementById("affiliations");
  project.affiliations.forEach((name, index) => {
    const item = document.createElement("span"); const number = document.createElement("sup"); number.textContent = index + 1;
    item.append(number, document.createTextNode(" " + name)); affiliationRoot.append(item);
  });
  document.querySelectorAll("[data-resource]").forEach(link => {
    const url = project.resources[link.dataset.resource] || (link.dataset.resource === "scenes" && project.resources.code ? project.resources.code + "#download-scene-assets" : "");
    if (url) { link.href = url; if (/^https?:/.test(url)) { link.target = "_blank"; link.rel = "noopener noreferrer"; } }
  });

  const tabs = document.getElementById("group-tabs");
  const panel = document.getElementById("group-panel");
  const composition = document.getElementById("composition-bar");
  const taskTotal = project.groups.reduce((total, group) => total + group.count, 0);
  project.groups.forEach(group => {
    const segment = document.createElement("span");
    segment.dataset.group = group.code;
    segment.style.flexGrow = group.count;
    segment.textContent = group.code;
    segment.title = group.title + ": " + group.count + " of " + taskTotal + " tasks";
    segment.setAttribute("aria-label", segment.title);
    composition.append(segment);
  });
  const groupIcons = {
    G1: '<circle cx="10.5" cy="10.5" r="5.5"/><path d="m15 15 5 5M8 10.5h5m-2.5-2.5v5"/>',
    G2: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><path d="M14 17.5h7m-3.5-3.5v7"/>',
    G3: '<path d="m3 15 12-12 6 6L9 21ZM7 11l3 3m1-7 3 3m1-7 3 3"/>',
    G4: '<circle cx="12" cy="12" r="9"/><path d="m15.5 8.5-2 5-5 2 2-5Z"/>',
    G5: '<path d="m3 7 5-3 5 3v6l-5 3-5-3ZM3 7l5 3 5-3M8 10v6m6 2h7m-3-3 3 3-3 3"/>'
  };
  function chooseGroup(index, focus = false) {
    [...tabs.children].forEach((button, i) => { button.setAttribute("aria-selected", i === index); button.tabIndex = i === index ? 0 : -1; });
    const group = project.groups[index]; panel.setAttribute("aria-labelledby", "group-" + index);
    panel.dataset.group = group.code;
    panel.replaceChildren();
    const top = document.createElement("div"); top.className = "panel-top";
    const code = document.createElement("span"); code.className = "group-code"; code.textContent = group.code;
    const count = document.createElement("span"); count.className = "task-count"; count.textContent = group.count + " tasks";
    top.append(code, count);
    const heading = document.createElement("h3"); heading.textContent = group.title;
    const description = document.createElement("p"); description.textContent = group.description;
    const tags = document.createElement("div"); tags.className = "task-tags";
    group.types.forEach(type => { const tag = document.createElement("span"); tag.className = "task-tag"; tag.textContent = type; tags.append(tag); });
    const example = document.createElement("div"); example.className = "panel-question";
    const label = document.createElement("span"); label.textContent = "Example task";
    example.append(label, document.createTextNode(group.example));
    const copy = document.createElement("div"); copy.className = "group-copy";
    const overview = document.createElement("div"); overview.className = "group-overview";
    overview.append(top, heading, description, tags); copy.append(overview, example);
    const visual = document.createElement("div"); visual.className = "group-visual";
    const grid = document.createElement("div"); grid.className = "task-illustrations" + (group.stages ? " relocation-stages" : "");
    grid.dataset.count = group.illustrations.length;
    group.illustrations.forEach((task, stage) => {
      const card = document.createElement("figure"); card.className = "task-illustration";
      const view = document.createElement("button"); view.type = "button"; view.className = "task-image"; view.dataset.figure = "task-atlas";
      view.setAttribute("aria-label", "View " + task.title + " in the full task atlas");
      const image = document.createElement("img"); image.src = "assets/tasks/" + task.image + ".webp";
      image.alt = task.alt; image.loading = "lazy"; image.decoding = "async"; image.width = 960; image.height = 540;
      view.append(image);
      const caption = document.createElement("figcaption"); caption.className = "task-figure-caption";
      const taskCopy = document.createElement("div"); taskCopy.className = "task-figure-copy";
      if (group.stages) {
        const number = document.createElement("span"); number.className = "task-stage-number"; number.textContent = "0" + (stage + 1); taskCopy.append(number);
      }
      const title = document.createElement("h4"); title.textContent = task.title; taskCopy.append(title);
      if (task.question) { const question = document.createElement("p"); question.textContent = task.question; taskCopy.append(question); }
      caption.append(taskCopy);
      if (task.diagram) {
        const diagram = document.createElement("img"); diagram.className = "task-diagram"; diagram.src = "assets/tasks/" + task.image + "-diagram.webp";
        diagram.alt = task.title + " schematic from the paper"; diagram.loading = "lazy"; diagram.decoding = "async"; caption.append(diagram);
      }
      card.append(view, caption); grid.append(card);
    });
    const atlas = document.createElement("button"); atlas.type = "button"; atlas.className = "task-atlas-link"; atlas.dataset.figure = "task-atlas";
    atlas.textContent = "View full task atlas ↗"; visual.append(grid, atlas); panel.append(copy, visual);
    if (tabs.scrollWidth > tabs.clientWidth) {
      const button = tabs.children[index];
      const tabBox = button.getBoundingClientRect(), listBox = tabs.getBoundingClientRect();
      tabs.scrollTo({left: Math.max(0, tabs.scrollLeft + tabBox.left - listBox.left - (tabs.clientWidth - button.clientWidth) / 2), behavior: "instant"});
    }
    if (focus) tabs.children[index].focus();
  }
  project.groups.forEach((group, index) => {
    const button = document.createElement("button"); button.className = "group-tab"; button.type = "button"; button.id = "group-" + index;
    button.dataset.group = group.code;
    button.setAttribute("role", "tab"); button.setAttribute("aria-controls", "group-panel");
    const tabHead = document.createElement("span"); tabHead.className = "group-tab-header";
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg"); icon.setAttribute("viewBox", "0 0 24 24"); icon.setAttribute("aria-hidden", "true"); icon.innerHTML = groupIcons[group.code];
    const code = document.createElement("span"); code.className = "group-tab-code"; code.textContent = group.code;
    const label = document.createElement("span"); label.className = "group-tab-label"; label.textContent = group.title;
    const detail = document.createElement("span"); detail.className = "group-tab-detail";
    detail.textContent = group.count + " tasks · " + group.types.length + (group.types.length === 1 ? " type" : " types");
    tabHead.append(icon, code); button.append(tabHead, label, detail);
    button.addEventListener("click", () => chooseGroup(index));
    button.addEventListener("keydown", event => {
      let next;
      if (event.key === "ArrowDown" || event.key === "ArrowRight") next = (index + 1) % project.groups.length;
      if (event.key === "ArrowUp" || event.key === "ArrowLeft") next = (index + project.groups.length - 1) % project.groups.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = project.groups.length - 1;
      if (next !== undefined) { event.preventDefault(); chooseGroup(next, true); }
    }); tabs.append(button);
  }); chooseGroup(0);

  // A second presentation of Table 2: the same percentages, on a common scale.
  const chart = document.getElementById("outcome-chart-body");
  const chartRows = [...project.results].sort((left, right) => right.T - left.T);
  const axis = document.createElement("div"); axis.className = "chart-axis";
  const axisLabel = document.createElement("span"); axisLabel.textContent = "Model";
  const axisTrack = document.createElement("div"); axisTrack.className = "chart-axis-track";
  ["0", "50", "100%"].forEach(value => { const tick = document.createElement("span"); tick.textContent = value; axisTrack.append(tick); });
  const axisValues = document.createElement("span"); axisValues.className = "chart-values-header"; axisValues.textContent = "A / T (%)";
  axis.append(axisLabel, axisTrack, axisValues); chart.append(axis);
  chartRows.forEach(result => {
    const row = document.createElement("div"); row.className = "outcome-chart-row" + (result.human ? " chart-human-row" : "");
    const name = document.createElement("span"); name.className = "chart-model-name"; name.textContent = result.model;
    const tracks = document.createElement("div"); tracks.className = "chart-row-tracks";
    ["A", "T"].forEach(metric => {
      const bar = document.createElement("span"); bar.className = "chart-bar chart-bar-" + metric.toLowerCase();
      bar.style.width = result[metric] + "%"; tracks.append(bar);
    });
    const values = document.createElement("span"); values.className = "chart-row-values";
    values.textContent = result.A.toFixed(1) + " / " + result.T.toFixed(1);
    row.append(name, tracks, values); chart.append(row);
  });
  chart.setAttribute("aria-label", "Outcome correctness A and task success T, percentages from Table 2. " + chartRows.map(result => result.model + ": A " + result.A + "%, T " + result.T + "%").join("; "));

  function renderResults(metric) {
    const rows = [...project.results].sort((left, right) => right[metric] - left[metric]);
    const body = document.getElementById("results-body"); body.replaceChildren();
    rows.forEach(result => {
      const row = document.createElement("tr"); if (result.human) row.className = "human-row";
      if (result.model === "GPT-6-astra") row.classList.add("leading-vlm-row");
      [result.model, result.tasks, result.A.toFixed(1), result.T.toFixed(1), result.N.toFixed(1), result.S.toFixed(1), result.gap, result.errorS.toFixed(1), result.actions.toFixed(2)].forEach((value, index) => {
        const cell = document.createElement("td"); cell.textContent = value;
        if (index === 3) { cell.className = "t-column score-cell"; const bar = document.createElement("span"); bar.className = "score-bar"; bar.style.width = (result.T * .46) + "px"; bar.setAttribute("aria-hidden", "true"); cell.append(bar); }
        row.append(cell);
      }); body.append(row);
    });
  }
  const sort = document.getElementById("sort-metric"); sort.addEventListener("change", () => renderResults(sort.value)); renderResults("T");
  const figures = [
    {id: "overview", title: "A whole-home counting task", height: 1412,
      caption: "Explore several rooms and inspect a closed cupboard. Evaluation checks the final count, target observations, room coverage, and container inspection."},
    {id: "task-atlas", title: "Eleven task types across five ability groups", height: 2109,
      caption: "Recorded camera observations and schematic annotations illustrate grounding, aggregation, geometry, reference frames, and goal-directed action. Instructions are abbreviated."},
    {id: "interaction-protocol", title: "Agent interaction and independent evaluation", height: 1214,
      caption: "Retrieval adds earlier images to the current round. Executing a skill returns new observations and feedback. Finishing triggers independent outcome and task-execution checks."},
    {id: "task-performance", title: "Task-group performance and shared difficulty", height: 735,
      caption: "Left: task success in each ability group for seven VLMs. Right: the 250 tasks grouped by the number of models answering correctly."}
  ];
  const dialog = document.getElementById("figure-dialog");
  const dialogImage = document.getElementById("figure-dialog-image");
  const dialogBody = document.getElementById("figure-dialog-body");
  const zoom = document.getElementById("figure-zoom");
  let figureIndex = 0;
  function fitFigure() {
    dialog.classList.remove("zoomed"); zoom.setAttribute("aria-pressed", "false"); zoom.textContent = "Full size";
    dialogBody.scrollTo({left: 0, top: 0, behavior: "instant"});
  }
  function showFigure(index) {
    figureIndex = (index + figures.length) % figures.length;
    const figure = figures[figureIndex]; fitFigure();
    document.getElementById("figure-dialog-title").textContent = figure.title;
    document.getElementById("figure-dialog-count").textContent = "Paper figure " + (figureIndex + 1) + " / " + figures.length;
    document.getElementById("figure-dialog-caption").textContent = figure.caption;
    dialogImage.src = "assets/figures/" + figure.id + ".svg";
    dialogImage.alt = figure.title + ". " + figure.caption;
    dialogImage.width = 2520; dialogImage.height = figure.height;
    dialog.style.setProperty("--figure-width", "2520px");
  }
  document.addEventListener("click", event => {
    const trigger = event.target.closest("button[data-figure]");
    if (!trigger) return;
    const index = figures.findIndex(figure => figure.id === trigger.dataset.figure);
    if (index < 0) return;
    showFigure(index); dialog.showModal(); document.body.classList.add("figure-viewer-open");
    document.getElementById("figure-close").focus();
  });
  document.getElementById("figure-close").addEventListener("click", () => dialog.close());
  document.getElementById("figure-previous").addEventListener("click", () => showFigure(figureIndex - 1));
  document.getElementById("figure-next").addEventListener("click", () => showFigure(figureIndex + 1));
  zoom.addEventListener("click", () => {
    const enlarged = dialog.classList.toggle("zoomed");
    zoom.setAttribute("aria-pressed", String(enlarged)); zoom.textContent = enlarged ? "Fit to screen" : "Full size";
    dialogBody.scrollTo({left: 0, top: 0, behavior: "instant"});
  });
  dialog.addEventListener("close", () => document.body.classList.remove("figure-viewer-open"));
  dialog.addEventListener("click", event => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
  });
  dialog.addEventListener("keydown", event => {
    if (dialog.classList.contains("zoomed")) return;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault(); showFigure(figureIndex + (event.key === "ArrowRight" ? 1 : -1));
    }
  });

  // Keep section navigation oriented as the reader moves through the page.
  if ("IntersectionObserver" in window) {
    const navLinks = [...document.querySelectorAll('.site-header nav a[href^="#"]')];
    const sectionObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        navLinks.forEach(link => {
          if (link.hash === "#" + entry.target.id) link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        });
      });
    }, {rootMargin: "-12% 0px -60% 0px", threshold: 0});
    navLinks.forEach(link => { const section = document.querySelector(link.hash); if (section) sectionObserver.observe(section); });
  }
})();
