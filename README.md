# ROAM-Bench project website

Source for [the ROAM-Bench project website](https://rxjfighting.github.io/ROAM-Bench/), hosted by GitHub Pages from the `gh-pages` branch.

The website presents the authors and affiliations, the paper, five task groups, evaluation rules, model results, and paper figures. The exploration video illustrates the task and simulation environment with scripted robot motions; it is not a VLM evaluation episode. During evaluation, the VLM selects atomic actions and the platform executes them.

## Local preview

Run from this directory:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`. The page also supports opening `index.html` directly.

## Editing

- `index.html` contains the page structure.
- `assets/project.js` contains authors, affiliations, task descriptions, result values, and resource links.
- `assets/site.js` provides task tabs, figure galleries, and charts.
- The CSS files in `assets/` control the layout and appearance.
- `assets/figures/` contains vector SVG exports of the original figure artwork, with source camera images embedded at their original resolution. Text and lines stay sharp when enlarged.
- `assets/tasks/` contains task example images.

The paper links to [arXiv](https://arxiv.org/abs/2611.33143). All other media and fonts are packaged locally. The Inter and Manrope font licenses are included in `assets/fonts/`. The page contains no analytics or external font requests.

## Deployment and resources

Keep the website files at the root of `gh-pages`, including `.nojekyll`. In GitHub Pages settings, select `gh-pages` and `/(root)`. The `main` branch contains the evaluation code and task set.

- Code and tasks: https://github.com/rxjfighting/ROAM-Bench
- Scene assets: https://huggingface.co/datasets/xjran/ROAM-Bench-scenes
