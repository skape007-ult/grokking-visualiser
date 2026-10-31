"""Headless browser check of the static site (dev tool, needs Playwright).

    pip install playwright && python -m playwright install chromium
    python check_site.py [--screenshots]

Serves site/ over HTTP on a free port, loads it in headless Chromium at
desktop and phone sizes, and fails (non-zero exit) on console errors, failed
requests, a missing chart, a slider that doesn't update the readout, or
horizontal scrolling on mobile.
"""

from __future__ import annotations

import argparse
import functools
import http.server
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
    page.on("console", lambda m: m.type == "error" and problems.append(f"console: {m.text}"))
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    page.on("requestfailed", lambda r: problems.append(f"requestfailed: {r.url}"))
    page.on("response", lambda r: r.status >= 400 and problems.append(f"HTTP {r.status}: {r.url}"))
    page.goto(url, wait_until="networkidle")
    page.wait_for_function("document.getElementById('readout').textContent.startsWith('Step')")
    return ctx, page, problems


def readout(page):
    return page.locator("#readout").inner_text()


def check_common(page, problems, label):
    check(not problems, f"[{label}] no console errors / failed requests {problems or ''}")
    check(page.locator("#error").is_hidden(), f"[{label}] no error banner")
    has_chart = page.evaluate("() => !!Chart.getChart('curve') && Chart.getChart('curve').data.datasets[0].data.length > 0")
    check(has_chart, f"[{label}] accuracy chart rendered with data")


def check_slider(page, label):
    slider = page.locator("#step-slider")
    before = readout(page)
    slider.focus()
    page.keyboard.press("Home")
    at_start = readout(page)
    check(at_start.startswith("Step 1 —"), f"[{label}] Home key -> step 1 ({at_start!r})")
    page.keyboard.press("ArrowRight")
    check(readout(page) != at_start, f"[{label}] ArrowRight changes readout ({readout(page)!r})")
    # Mouse drag across the slider track.
    box = slider.bounding_box()
    y = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + 4, y)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.6, y, steps=8)
    page.mouse.up()
    dragged = readout(page)
    check(dragged not in (at_start, before), f"[{label}] mouse drag updates readout ({dragged!r})")
    valuetext = slider.get_attribute("aria-valuetext")
    check(valuetext == dragged, f"[{label}] aria-valuetext mirrors readout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshots", action="store_true", help="save to notes/screenshots/")
    args = ap.parse_args()

    httpd, url = serve()
    print(f"serving {SITE} at {url}")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()

        print("desktop 1280x900")
        ctx, page, problems = open_page(browser, url, viewport={"width": 1280, "height": 900},
                                        device_scale_factor=2)
        check_common(page, problems, "desktop")
        check_slider(page, "desktop")
        if args.screenshots:
            SHOTS.mkdir(parents=True, exist_ok=True)
            page.keyboard.press("End")
            page.screenshot(path=str(SHOTS / "desktop.png"), full_page=True)
        ctx.close()

        print("mobile 390x844")
        ctx, page, problems = open_page(browser, url, viewport={"width": 390, "height": 844},
                                        device_scale_factor=3, is_mobile=True, has_touch=True)
        check_common(page, problems, "mobile")
        no_hscroll = page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")
        check(no_hscroll, "[mobile] no horizontal scroll")
        slider = page.locator("#step-slider")
        box = slider.bounding_box()
        check(box["height"] >= 44, f"[mobile] slider touch target >= 44px ({box['height']:.0f}px)")
        before = readout(page)
        page.touchscreen.tap(box["x"] + box["width"] * 0.3, box["y"] + box["height"] / 2)
        check(readout(page) != before, f"[mobile] tap on slider updates readout ({readout(page)!r})")
        if args.screenshots:
            page.screenshot(path=str(SHOTS / "mobile.png"), full_page=True)
        ctx.close()

        browser.close()
    httpd.shutdown()

    if failures:
        print(f"\n{len(failures)} check(s) FAILED")
        sys.exit(1)
    print("\nall site checks passed")


if __name__ == "__main__":
    main()
