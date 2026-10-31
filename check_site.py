"""Headless browser check of the static site (dev tool, needs Playwright).

    pip install playwright && python -m playwright install chromium
    python check_site.py [--screenshots]

Serves site/ over HTTP on a free port, loads it in headless Chromium at
desktop and phone sizes, and fails (non-zero exit) on: console errors or
failed requests, a missing chart or blank embedding plot, a slider that
doesn't update the readout, a run toggle / URL hash that doesn't switch runs,
horizontal scrolling or small touch targets on mobile, or explanation-text
numbers that don't match site/data.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import re
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
SHOTS = ROOT / "notes" / "screenshots"

failures: list[str] = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve():
    handler = functools.partial(QuietHandler, directory=str(SITE))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/"


def open_page(browser, url, **ctx_kwargs):
    ctx = browser.new_context(**ctx_kwargs)
    page = ctx.new_page()
    problems: list[str] = []
    requests: list[tuple[str, int]] = []
    page.on("console", lambda m: m.type == "error" and problems.append(f"console: {m.text}"))
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    page.on("requestfailed", lambda r: problems.append(f"requestfailed: {r.url}"))
    page.on("response", lambda r: r.status >= 400 and problems.append(f"HTTP {r.status}: {r.url}"))
    def on_finished(r):
        try:
            requests.append((r.url, r.sizes()["responseBodySize"]))
        except Exception:  # page closed before sizes were available
            pass
    page.on("requestfinished", on_finished)
    page.goto(url, wait_until="networkidle")
    page.wait_for_function("document.getElementById('readout').textContent.startsWith('Step')")
    return ctx, page, problems, requests


def readout(page):
    return page.locator("#readout").inner_text()


def pressed_run(page):
    return page.locator("#run-toggle button[aria-pressed=true]").get_attribute("data-run")


def embedding_ink(page):
    """Number of non-transparent pixels on the embedding canvas."""
    return page.evaluate("""() => {
        const c = document.getElementById('embedding');
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++; return n;
    }""")


def check_common(page, problems, label):
    check(not problems, f"[{label}] no console errors / failed requests {problems or ''}")
    check(page.locator("#error").is_hidden(), f"[{label}] no error banner")
    has_chart = page.evaluate(
        "() => !!Chart.getChart('curve') && Chart.getChart('curve').data.datasets[0].data.length > 0")
    check(has_chart, f"[{label}] accuracy chart rendered with data")
    check(embedding_ink(page) > 2000, f"[{label}] embedding canvas has points drawn")
    box = page.locator("#embedding").bounding_box()
    check(abs(box["width"] - box["height"]) < 1, f"[{label}] embedding plot is square ({box['width']:.0f}x{box['height']:.0f})")
    canvas_px = page.evaluate("() => document.getElementById('embedding').width")
    dpr = page.evaluate("() => devicePixelRatio")
    check(abs(canvas_px - box["width"] * dpr) <= 1, f"[{label}] embedding canvas is high-DPI ({canvas_px}px backing, dpr {dpr})")


def check_slider(page, label):
    slider = page.locator("#step-slider")
    before = readout(page)
    slider.focus()
    page.keyboard.press("Home")
    at_start = readout(page)
    check(at_start.startswith("Step 1 —"), f"[{label}] Home key -> step 1 ({at_start!r})")
    check(page.locator("#frame-label").inner_text() == "· step 1", f"[{label}] embedding shows frame at step 1")
    ink_start = page.evaluate("() => document.getElementById('embedding').toDataURL()")
    page.keyboard.press("ArrowRight")
    check(readout(page) != at_start, f"[{label}] ArrowRight changes readout ({readout(page)!r})")
    box = slider.bounding_box()
    y = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + 4, y)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.85, y, steps=8)
    page.mouse.up()
    dragged = readout(page)
    check(dragged not in (at_start, before), f"[{label}] mouse drag updates readout ({dragged!r})")
    check(slider.get_attribute("aria-valuetext") == dragged, f"[{label}] aria-valuetext mirrors readout")
    ink_late = page.evaluate("() => document.getElementById('embedding').toDataURL()")
    check(ink_start != ink_late, f"[{label}] embedding changes as the slider moves")
    frame_step = int(page.locator("#frame-label").inner_text().split("step ")[1].replace(",", ""))
    step = int(re.match(r"Step ([\d,]+)", dragged).group(1).replace(",", ""))
    check(frame_step <= step, f"[{label}] embedding frame step {frame_step} <= slider step {step}")


def wait_run(page, rid):
    page.wait_for_function(f"document.body.dataset.run === '{rid}'")


def fmt_readout(rid, step_label):
    """The readout the page should show for run `rid` at 'Step N'."""
    run = json.loads((SITE / "data" / "runs" / f"{rid}.json").read_text())
    i = run["steps"].index(int(step_label.removeprefix("Step ").replace(",", "")))
    pct = lambda x: f"{int(x * 100 + 0.5)}%"  # JS Math.round
    return f"{step_label} — train {pct(run['train_acc'][i])}, test {pct(run['test_acc'][i])}"


def check_toggle(page, label):
    ids = [b.get_attribute("data-run") for b in page.locator("#run-toggle button").all()]
    check(ids == ["baseline", "no_decay", "more_data", "less_data"], f"[{label}] toggle buttons from manifest {ids}")
    step_before = readout(page).split(" —")[0]
    for rid in ids[1:] + ids[:1]:
        page.locator(f"#run-toggle button[data-run={rid}]").click()
        wait_run(page, rid)
        check(pressed_run(page) == rid and page.url.endswith(f"#run={rid}"),
              f"[{label}] click {rid}: pressed + hash ({page.url.split('/')[-1]})")
        check(readout(page).split(" —")[0] == step_before, f"[{label}] {rid}: slider keeps step ({readout(page)!r})")
        expected = fmt_readout(rid, step_before)
        check(readout(page) == expected, f"[{label}] {rid}: readout shows this run's data ({expected!r})")
    page.locator("#run-toggle button[data-run=no_decay]").click()
    wait_run(page, "no_decay")
    page.locator("#step-slider").focus()
    page.keyboard.press("End")
    check(readout(page).endswith("test 3%"), f"[{label}] no_decay at end reads test 3% ({readout(page)!r})")


def check_hash(browser, url):
    ctx, page, problems, _ = open_page(browser, url + "#run=less_data", viewport={"width": 1280, "height": 900})
    wait_run(page, "less_data")
    check(pressed_run(page) == "less_data" and not problems, "[hash] #run=less_data deep link selects that run")
    page.evaluate("location.hash = '#run=more_data'")
    wait_run(page, "more_data")
    check(True, "[hash] changing the hash switches run")
    ctx.close()
    ctx, page, problems, _ = open_page(browser, url + "#run=bogus", viewport={"width": 1280, "height": 900})
    check(pressed_run(page) == "baseline" and not problems, "[hash] unknown run id falls back to baseline")
    ctx.close()


def check_facts():
    """Explanation numbers in index.html must match the exported data."""
    runs = {r: json.loads((SITE / "data" / "runs" / f"{r}.json").read_text())
            for r in ("baseline", "no_decay", "more_data", "less_data")}

    def first(run, key, thr):
        return next(s for s, v in zip(run["steps"], run[key]) if v >= thr)

    b = runs["baseline"]
    expected = {
        "baseline.train99": f"{first(b, 'train_acc', 0.99):,}",
        "baseline.test95": f"{first(b, 'test_acc', 0.95):,}",
        "baseline.ratio": f"{round(first(b, 'test_acc', 0.95) / first(b, 'train_acc', 0.99))}",
        "no_decay.finalTest": f"{round(runs['no_decay']['test_acc'][-1] * 100)}%",
        "more_data.test95": f"{first(runs['more_data'], 'test_acc', 0.95):,}",
        "less_data.finalTest": f"{round(runs['less_data']['test_acc'][-1] * 100)}%",
    }
    html = (SITE / "index.html").read_text()
    found = dict(re.findall(r'data-fact="([\w.]+)">([^<]+)<', html))
    for k, v in expected.items():
        check(found.get(k) == v, f"[facts] {k}: page says {found.get(k)!r}, data says {v!r}")
    check(set(found) == set(expected), f"[facts] every data-fact span is checked {sorted(set(found) ^ set(expected))}")
    # Claims that runs never reached 50% must hold.
    for r in ("no_decay", "less_data"):
        check(max(runs[r]["test_acc"]) < 0.5, f"[facts] {r} never reaches 50% test acc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshots", action="store_true", help="save to notes/screenshots/")
    args = ap.parse_args()

    check_facts()
    httpd, url = serve()
    print(f"serving site/ at {url}")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()

        print("desktop 1280x900")
        ctx, page, problems, requests = open_page(browser, url, viewport={"width": 1280, "height": 900},
                                                  device_scale_factor=2)
        check_common(page, problems, "desktop")
        local = [(u, n) for u, n in requests if u.startswith(url)]
        total_kb = sum(n for _, n in requests) / 1024
        print(f"  info initial view: {len(requests)} requests, {total_kb:.0f} KB (uncompressed bodies); "
              f"local: {[u.removeprefix(url) for u, _ in local]}")
        check(not any("runs/no_decay" in u or "runs/more_data" in u for u, _ in local),
              "[desktop] other runs are lazy-loaded (not fetched up front)")
        check_slider(page, "desktop")
        check_toggle(page, "desktop")
        # Tooltip on hover over a point.
        pt = page.evaluate("""() => { const r = document.getElementById('embedding').getBoundingClientRect();
            return [r.left, r.top, r.width]; }""")
        page.mouse.move(pt[0] + pt[2] / 2, pt[1] + pt[2] / 2)
        for dx in range(-60, 61, 6):
            for dy in range(-60, 61, 6):
                page.mouse.move(pt[0] + pt[2] / 2 + dx, pt[1] + pt[2] / 2 + dy)
                if page.locator("#emb-tooltip").is_visible():
                    break
            if page.locator("#emb-tooltip").is_visible():
                break
        tip = page.locator("#emb-tooltip")
        check(tip.is_visible() and tip.inner_text().isdigit(), f"[desktop] hover shows number tooltip ({tip.inner_text() if tip.is_visible() else None})")
        ctx.close()

        check_hash(browser, url)

        if args.screenshots:
            SHOTS.mkdir(parents=True, exist_ok=True)
            for rid, idx in (("baseline", 199), ("no_decay", 199)):
                ctx, page, _, _ = open_page(browser, f"{url}#run={rid}", viewport={"width": 1280, "height": 900},
                                            device_scale_factor=2)
                wait_run(page, rid)
                page.evaluate(f"() => {{ const s = document.getElementById('step-slider'); s.value = {idx}; "
                              "s.dispatchEvent(new Event('input')); }")
                page.screenshot(path=str(SHOTS / f"desktop_{rid}.png"))
                if rid == "baseline":
                    page.screenshot(path=str(SHOTS / "desktop_full.png"), full_page=True)
                ctx.close()

        print("mobile 390x844")
        ctx, page, problems, _ = open_page(browser, url, viewport={"width": 390, "height": 844},
                                           device_scale_factor=3, is_mobile=True, has_touch=True)
        check_common(page, problems, "mobile")
        check(page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth"),
              "[mobile] no horizontal scroll")
        slider = page.locator("#step-slider")
        box = slider.bounding_box()
        check(box["height"] >= 44, f"[mobile] slider touch target >= 44px ({box['height']:.0f}px)")
        heights = [b.bounding_box()["height"] for b in page.locator("#run-toggle button").all()]
        check(min(heights) >= 44, f"[mobile] run buttons >= 44px tall ({min(heights):.0f}px)")
        before = readout(page)
        page.touchscreen.tap(box["x"] + box["width"] * 0.6, box["y"] + box["height"] / 2)
        check(readout(page) != before, f"[mobile] tap on slider updates readout ({readout(page)!r})")
        page.locator("#run-toggle button[data-run=no_decay]").tap()
        wait_run(page, "no_decay")
        check(True, "[mobile] tapping a run button switches run")
        emb_w = page.locator("#embedding").bounding_box()["width"]
        check(emb_w <= 390 - 32 + 1, f"[mobile] embedding fits the screen ({emb_w:.0f}px)")
        if args.screenshots:
            page.locator("#run-toggle button[data-run=baseline]").tap()
            wait_run(page, "baseline")
            page.evaluate("() => { const s = document.getElementById('step-slider'); s.value = 199; "
                          "s.dispatchEvent(new Event('input')); }")
            page.screenshot(path=str(SHOTS / "mobile.png"))
            page.screenshot(path=str(SHOTS / "mobile_full.png"), full_page=True)
        ctx.close()

        browser.close()
    httpd.shutdown()

    if failures:
        print(f"\n{len(failures)} check(s) FAILED")
        sys.exit(1)
    print("\nall site checks passed")


if __name__ == "__main__":
    main()
