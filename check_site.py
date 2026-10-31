"""Headless browser check of the static site (dev tool, needs Playwright).

    pip install playwright && python -m playwright install chromium
    python check_site.py [--screenshots] [--browsers chromium,webkit,firefox | all]

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


def check_hash(browser, url, name):
    ctx, page, problems, _ = open_page(browser, url + "#run=less_data", viewport={"width": 1280, "height": 900})
    wait_run(page, "less_data")
    check(pressed_run(page) == "less_data" and not problems, f"[{name}/hash] #run=less_data deep link selects that run")
    page.evaluate("location.hash = '#run=more_data'")
    wait_run(page, "more_data")
    check(True, f"[{name}/hash] changing the hash switches run")
    ctx.close()
    ctx, page, problems, _ = open_page(browser, url + "#run=bogus", viewport={"width": 1280, "height": 900})
    check(pressed_run(page) == "baseline" and not problems, f"[{name}/hash] unknown run id falls back to baseline")
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


def set_slider(page, idx):
    page.evaluate(f"() => {{ const s = document.getElementById('step-slider'); s.value = {idx}; "
                  "s.dispatchEvent(new Event('input')); }")


def make_screenshots(browser, url):
    """README/OG images and a GIF of the slider being dragged through training."""
    from PIL import Image

    SHOTS.mkdir(parents=True, exist_ok=True)
    for rid in ("baseline", "no_decay"):
        ctx, page, _, _ = open_page(browser, f"{url}#run={rid}", viewport={"width": 1280, "height": 900},
                                    device_scale_factor=2)
        wait_run(page, rid)
        set_slider(page, 199)
        page.screenshot(path=str(SHOTS / f"desktop_{rid}.png"))
        ctx.close()

    # Open Graph image (1200x630): the visualisation, scrolled into view.
    ctx, page, _, _ = open_page(browser, url, viewport={"width": 1200, "height": 630})
    set_slider(page, 199)
    page.evaluate("() => window.scrollTo(0, document.getElementById('run-toggle').offsetTop - 12)")
    page.screenshot(path=str(SITE / "og-image.png"))
    ctx.close()

    # GIF: drag the slider from step 1 to the end on the baseline run.
    ctx, page, _, _ = open_page(browser, url, viewport={"width": 1100, "height": 1100})
    box = page.locator(".viz").bounding_box()
    clip = {"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]}
    frames = []
    n = int(page.evaluate("() => document.getElementById('step-slider').max"))
    indices = list(range(0, n + 1, 5)) + [n]
    for i in indices:
        set_slider(page, i)
        png = page.screenshot(clip=clip)
        im = Image.open(__import__("io").BytesIO(png)).convert("RGB")
        im = im.resize((800, round(800 * im.height / im.width)), Image.LANCZOS)
        frames.append(im.quantize(colors=255, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE))
    ctx.close()
    durations = [90] * (len(frames) - 1) + [2500]
    frames[0].save(SHOTS / "scrub.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"  info wrote screenshots + scrub.gif ({(SHOTS / 'scrub.gif').stat().st_size // 1024} KB, "
          f"{len(frames)} frames) + site/og-image.png")


def check_desktop(browser, url, label):
    ctx, page, problems, requests = open_page(browser, url, viewport={"width": 1280, "height": 900},
                                              device_scale_factor=2)
    check_common(page, problems, label)
    local = [u.removeprefix(url) for u, _ in requests if u.startswith(url)]
    total_kb = sum(n for _, n in requests) / 1024
    print(f"  info initial view: {len(requests)} requests, {total_kb:.0f} KB (uncompressed bodies); local: {local}")
    check(not any("runs/" in u and "baseline" not in u for u in local),
          f"[{label}] other runs are lazy-loaded (not fetched up front)")
    check_slider(page, label)
    check_toggle(page, label)

    # Hover a point -> tooltip; click it -> stays pinned after the mouse leaves.
    r = page.locator("#embedding").bounding_box()
    cx, cy = r["x"] + r["width"] / 2, r["y"] + r["height"] / 2
    tip = page.locator("#emb-tooltip")
    found = None
    for dx in range(-90, 91, 5):
        for dy in range(-90, 91, 5):
            page.mouse.move(cx + dx, cy + dy)
            if tip.is_visible():
                found = (cx + dx, cy + dy)
                break
        if found:
            break
    check(found is not None and tip.inner_text().isdigit(), f"[{label}] hover shows number tooltip")
    if found:
        number = tip.inner_text()
        page.mouse.click(*found)
        page.mouse.move(r["x"] - 40, r["y"] - 40)
        check(tip.is_visible() and tip.inner_text() == number, f"[{label}] click pins number {number}")
    ctx.close()


def check_load_failure(browser, url, label):
    """If a run fails to load, show an error but keep the current run intact."""
    ctx, page, _, _ = open_page(browser, url, viewport={"width": 1280, "height": 900})
    page.route("**/runs/more_data.json", lambda route: route.abort())
    before = readout(page)
    page.locator("#run-toggle button[data-run=more_data]").click()
    try:
        page.wait_for_selector("#error:not([hidden])", timeout=5000)
        shown = True
    except Exception:
        shown = False
    check(shown, f"[{label}] failed load: error message shown")
    check(pressed_run(page) == "baseline", f"[{label}] failed load: toggle reverts to the shown run")
    check(readout(page) == before, f"[{label}] failed load: readout and data kept")
    page.unroute("**/runs/more_data.json")
    page.locator("#run-toggle button[data-run=more_data]").click()
    wait_run(page, "more_data")
    check(page.locator("#error").is_hidden(), f"[{label}] retry after failure works and clears the error")
    ctx.close()


def check_mobile(browser, url, label, mobile_ctx, screenshots=False):
    ctx, page, problems, _ = open_page(browser, url, **mobile_ctx)
    check_common(page, problems, label)
    check(page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth"),
          f"[{label}] no horizontal scroll")
    slider = page.locator("#step-slider")
    box = slider.bounding_box()
    check(box["height"] >= 44, f"[{label}] slider touch target >= 44px ({box['height']:.0f}px)")
    heights = [b.bounding_box()["height"] for b in page.locator("#run-toggle button").all()]
    check(min(heights) >= 44, f"[{label}] run buttons >= 44px tall ({min(heights):.0f}px)")
    touch_action = page.evaluate("() => getComputedStyle(document.getElementById('embedding')).touchAction")
    check(touch_action != "none", f"[{label}] swipes on the embedding plot can scroll the page ({touch_action})")
    before = readout(page)
    page.touchscreen.tap(box["x"] + box["width"] * 0.6, box["y"] + box["height"] / 2)
    check(readout(page) != before, f"[{label}] tap on slider updates readout ({readout(page)!r})")
    page.locator("#run-toggle button[data-run=no_decay]").tap()
    wait_run(page, "no_decay")
    check(True, f"[{label}] tapping a run button switches run")
    emb_w = page.locator("#embedding").bounding_box()["width"]
    vw = mobile_ctx["viewport"]["width"]
    check(emb_w <= vw - 32 + 1, f"[{label}] embedding fits the screen ({emb_w:.0f}px of {vw}px)")
    if screenshots:
        page.locator("#run-toggle button[data-run=baseline]").tap()
        wait_run(page, "baseline")
        set_slider(page, 199)
        page.screenshot(path=str(SHOTS / "mobile.png"))
    ctx.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshots", action="store_true", help="save to notes/screenshots/")
    ap.add_argument("--browsers", default="chromium",
                    help="comma-separated: chromium,webkit,firefox (or 'all')")
    args = ap.parse_args()
    names = ["chromium", "webkit", "firefox"] if args.browsers == "all" else args.browsers.split(",")

    check_facts()
    httpd, url = serve()
    print(f"serving site/ at {url}")
    with sync_playwright() as pw:
        for name in names:
            browser = getattr(pw, name).launch()
            print(f"--- {name} {browser.version}")
            check_desktop(browser, url, f"{name}/desktop")
            check_load_failure(browser, url, f"{name}/desktop")
            check_hash(browser, url, name)
            if name == "webkit":
                # Closest available stand-in for Safari on an iPhone.
                mobile_ctx = dict(pw.devices["iPhone 13"])
                mobile_ctx.pop("default_browser_type", None)
                label = f"{name}/iPhone 13"
            else:
                mobile_ctx = {"viewport": {"width": 390, "height": 844}, "device_scale_factor": 3,
                              "has_touch": True}
                if name != "firefox":  # Firefox has no is_mobile emulation
                    mobile_ctx["is_mobile"] = True
                label = f"{name}/mobile 390x844"
            shots = args.screenshots and name == "chromium"
            if shots:
                make_screenshots(browser, url)
            check_mobile(browser, url, label, mobile_ctx, screenshots=shots)
            browser.close()
    httpd.shutdown()

    if failures:
        print(f"\n{len(failures)} check(s) FAILED")
        sys.exit(1)
    print(f"\nall site checks passed ({', '.join(names)})")


if __name__ == "__main__":
    main()
