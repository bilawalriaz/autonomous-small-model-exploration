# Publishing Atlas Results as a GitHub Pages Site

## When to Use

After completing MI-Atlas experiments on one or more models, publish the findings as a static HTML site hosted on GitHub Pages. This makes the research accessible without reading Markdown files or cloning the repo.

## Prerequisites

- GitHub repo with the atlas results (must be PUBLIC for free-tier Pages)
- `docs/` directory in the repo root with HTML files
- `gh` CLI authenticated

## Steps

### 1. Make the repo public (if private)

```bash
gh repo edit <owner>/<repo> --visibility public --accept-visibility-change-consequences
```

### 2. Enable GitHub Pages from docs/ folder

```bash
# The path MUST be /docs (with leading slash), not docs
gh api repos/<owner>/<repo>/pages -X POST -f 'source[branch]=master' -f 'source[path]=/docs'
```

**Pitfall:** The API rejects `path=docs` — it must be `path=/docs` (with leading slash). The error message is confusing: "Invalid property /source/path: `docs` is not a possible value. Must be one of: /, /docs."

### 3. Verify the site

The site will be available at:
```
https://<username>.github.io/<repo-name>/
```

First build takes ~1-2 minutes. Check status:
```bash
gh api repos/<owner>/<repo>/pages --jq '.status'
```

## Site Design (matching llm-fundamentals)

The reference site at https://bilawalriaz.github.io/llm-fundamentals/ uses:
- Bulma CSS 0.9.4 from CDN
- Font Awesome 6.0.0-beta3 from CDN
- Google Sans + Noto Sans fonts from Google Fonts
- Orange gradient text for titles: `linear-gradient(90deg, #d4380d, #fa541c, #fa8c16)`
- Card-based layout (`.paper-card` class: grey bg #f5f5f5, 10px border-radius, hover border+shadow)
- Tier headers (`.tier-header`: Google Sans, 1.5rem, bold, border-bottom 2px solid black)
- Info callouts (`.swi-callout`: flex layout, grey background)
- 2-column grid on desktop, 1-column on mobile
- Clean white background, black text, minimal design
- All CSS inline in `<style>` tags — each page is self-contained

## Page Structure

- `index.html` — Landing page with cards linking to each analysis post
- `01-<model>-analysis.html` — Full analysis for each model
- `02-<model>-analysis.html` — Additional model analyses
- `03-comparison-analysis.html` — Cross-model comparison page

Each page should have:
- Navigation bar linking to all pages
- Footer: "Author · Year · repo path"
- Self-contained CSS (no external .css files)
