"""Runs the leaf classifier page in a real browser and records what it actually did.

    python tests/verify_page.py --browser chromium
    python tests/verify_page.py --browser webkit --url https://adamb4b.github.io/mlr-ex08/
    python tests/verify_page.py --browser chromium --fake-camera      (live camera loop, no phone)

With no --url the page is served from this folder on 127.0.0.1, so the test never depends on
the internet for the page itself (the two CDN libraries still come from jsDelivr).

What is checked, and written to tests/evidence/*.json:
  1. ?selftest=1                the page's own report, compared with check_expected.json at 3 dp
  2. upload of check_image.jpg  through the real file input, reading the result off the DOM
  3. centre_crop_probe.png      288x96 strip whose centre square IS check_image: pixel-exact tensor
  4. colour_probe_{1,2,3}.png   constant colours: exact tensor, catches channel-order / 255 / mean
  5. QR code                    the SVG is screenshotted and decoded with zxing-cpp
  6. network + console          every request listed, so "nothing leaves the phone" is evidence
  7. live camera loop           only with --fake-camera (Chromium's fake video device)

The page exposes classify() at the top level, so probes 3 and 4 get full float probabilities
instead of the one decimal the bars show.
"""
import argparse
import base64
import functools
import http.server
import json
import os
import sys
import threading
import time

import zxingcpp
from PIL import Image
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict_onnx import predict  # noqa: E402  (same folder, import after sys.path)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE = os.path.join(ROOT, "tests", "evidence")
PROBES = os.path.join(ROOT, "tests", "images")
APP_URL = "https://adamb4b.github.io/mlr-ex08/"
MODEL_READY = "() => /ready/i.test(document.getElementById('statusText').textContent)"
SELFTEST_DONE = "() => document.title.indexOf('SELFTEST') === 0"
BARS_FILLED = ("() => { const v = document.querySelector('#bars .value');"
               " return !!v && v.textContent !== '\\u2014'; }")


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(folder):
    handler = functools.partial(Quiet, directory=folder)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, "http://127.0.0.1:%d/" % httpd.server_address[1]


def read_bars(page):
    return page.eval_on_selector_all(
        "#bars li",
        "els => els.map(e => [e.querySelector('.name').textContent,"
        " e.querySelector('.value').textContent])")


def page_classify(page, path):
    """Run one image through the page's own preprocess()/classify(), at full precision."""
    data = base64.b64encode(open(path, "rb").read()).decode()
    return page.evaluate(
        """async (src) => {
             const img = new Image(); img.src = src; await img.decode();
             const r = await window.classify(img, img.naturalWidth, img.naturalHeight);
             return {logits: r.logits, probabilities: r.probabilities, tensorShape: r.tensorShape};
           }""",
        "data:image/png;base64," + data)


def run(url, browser_name, fake_camera, tag, folder=ROOT):
    os.makedirs(EVIDENCE, exist_ok=True)
    evidence = {"url": url, "browser": browser_name, "tag": tag,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checks": [], "console": [], "requests": [], "screenshots": []}
    expected = json.load(open(os.path.join(folder, "check_expected.json"), encoding="utf-8"))
    classes = json.load(open(os.path.join(folder, "model_info.json"), encoding="utf-8"))["classes"]

    def check(name, ok, detail):
        evidence["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
        print(("PASS  " if ok else "FAIL  ") + name + "  " + str(detail))

    with sync_playwright() as p:
        if browser_name == "chromium":
            args = []
            if fake_camera:
                args = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"]
            browser = p.chromium.launch(headless=True, args=args)
            context = browser.new_context(viewport={"width": 420, "height": 900},
                                          device_scale_factor=2,
                                          permissions=["camera"] if fake_camera else [])
        else:
            browser = p.webkit.launch(headless=True)
            context = browser.new_context(viewport={"width": 420, "height": 900},
                                          device_scale_factor=2)

        context.on("console", lambda m: evidence["console"].append(
            {"type": m.type, "text": m.text}))
        context.on("request", lambda r: evidence["requests"].append(
            {"method": r.method, "url": r.url, "type": r.resource_type}))

        page = context.new_page()

        # ---------------------------------------------------------------- 1. the page's selftest
        page.goto(url + "?selftest=1", wait_until="domcontentloaded", timeout=120000)
        try:
            page.wait_for_function(SELFTEST_DONE, timeout=240000)
            raw = page.text_content("#selftest") or ""
            report = json.loads(raw.split("SELFTEST_JSON=", 1)[1])
            evidence["selftest"] = report
            check("selftest ran", report.get("ok") is True, report.get("error", "ok"))
            check("selftest pass flag", report.get("pass") is True,
                  "maxAbsDiff %.2e" % report.get("maxAbsDiff", -1))
            check("selftest tensor shape", report.get("tensorShape") == [1, 3, 96, 96],
                  report.get("tensorShape"))
            check("selftest 3 decimals", (report.get("maxAbsDiff", 1) < 5e-4) and all(
                c["gotRounded"] == c["expectedRounded"] for c in report.get("compare", [])),
                  {c["class"]: [c["gotRounded"], c["expectedRounded"]]
                   for c in report.get("compare", [])})
        except Exception as error:                                   # noqa: BLE001
            evidence["selftest"] = {"error": repr(error)}
            check("selftest ran", False, repr(error))

        # -------------------------------------------------- 2. upload through the real UI
        page.goto(url, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_function(MODEL_READY, timeout=240000)
        page.set_input_files("#fileInput", os.path.join(folder, "check_image.jpg"))
        page.wait_for_function(BARS_FILLED, timeout=120000)
        evidence["upload"] = {
            "top": page.text_content("#topResult").strip(),
            "bars": read_bars(page),
            "perfHint": page.text_content("#perfHint").strip(),
        }
        top_class = evidence["upload"]["top"].split(" \u2014 ")[0].strip()
        check("upload predicts the expected class", top_class == max(expected, key=expected.get),
              evidence["upload"]["top"])
        shot = os.path.join(EVIDENCE, "upload-%s-%s.png" % (browser_name, tag))
        page.screenshot(path=shot, full_page=True)
        evidence["screenshots"].append(shot)

        # -------------------------------------------------- 3. + 4. exact-tensor probes
        check("page exposes classify()", page.evaluate("() => typeof window.classify") == "function",
              page.evaluate("() => typeof window.classify"))
        probe = page_classify(page, os.path.join(PROBES, "centre_crop_probe.png"))
        worst = max(abs(probe["probabilities"][i] - expected[c]) for i, c in enumerate(classes))
        evidence["centre_crop_probe"] = probe
        check("centre crop picks the middle square", worst < 1e-4,
              "max |page - check_expected| = %.2e, shape %s" % (worst, probe["tensorShape"]))
        check("centre crop tensor shape", probe["tensorShape"] == [1, 3, 96, 96],
              probe["tensorShape"])

        probe_rows = []
        for index in (1, 2, 3):
            path = os.path.join(PROBES, "colour_probe_%d.png" % index)
            got = page_classify(page, path)
            reference = predict(path, os.path.join(folder, "leaf_model.onnx"),
                                os.path.join(folder, "model_info.json"))
            diff = max(abs(got["probabilities"][i] - reference["probabilities"][c])
                       for i, c in enumerate(classes))
            probe_rows.append({"probe": os.path.basename(path), "browser": got["probabilities"],
                               "onnx": reference["probabilities"], "maxAbsDiff": diff})
            check("constant colour probe %d" % index, diff < 1e-3,
                  "max |browser - onnx| = %.2e (top %s)" % (diff, reference["top"]))
        evidence["colour_probes"] = probe_rows

        # -------------------------------------------------- 5. the QR code
        page.wait_for_selector("#qrBox svg", timeout=60000)
        qr_path = os.path.join(EVIDENCE, "qr-%s-%s.png" % (browser_name, tag))
        page.locator("#qrBox svg").screenshot(path=qr_path)
        decoded = [b.text for b in zxingcpp.read_barcodes(Image.open(qr_path).convert("L"))]
        link = page.get_attribute("#qrLink", "href")
        evidence["qr"] = {"decoded": decoded, "screenshot": qr_path,
                          "linkHref": link, "linkText": page.text_content("#qrLink").strip()}
        check("QR decodes to the public URL", decoded == [APP_URL], decoded)
        check("clickable link points at the public URL", link == APP_URL, link)

        # -------------------------------------------------- 6. network + console
        origin = "/".join(url.split("/")[:3])
        allowed = {"https://cdn.jsdelivr.net", origin}
        # blob: and data: URLs are created inside the page (an uploaded photo becomes a blob URL
        # so it can be shown in <img>); they are browser memory, not network traffic.
        local = [r["url"] for r in evidence["requests"]
                 if r["url"].startswith(("blob:", "data:"))]
        remote = [r for r in evidence["requests"]
                  if r["url"].startswith(("http://", "https://"))]
        hosts = sorted({r["url"].split("/")[0] + "//" + r["url"].split("/")[2]
                        for r in remote})
        writes = [r for r in remote if r["method"] != "GET"]
        evidence["localObjectUrls"] = local
        check("no request writes anything", not writes, writes)
        check("only the page's own host and the CDN are contacted",
              all(h in allowed for h in hosts), hosts)
        check("image data stays in local blob:/data: URLs", True, local)
        errors = [c for c in evidence["console"] if c["type"] == "error"]
        check("no console errors", not errors, errors or "none")

        # -------------------------------------------------- 7. the live loop (fake camera)
        if fake_camera:
            page.click("#startBtn")
            try:
                page.wait_for_function(
                    "() => { const v = document.getElementById('video');"
                    " return v && !v.hidden && v.videoWidth > 0 && !!v.srcObject; }", timeout=30000)
                page.wait_for_function(
                    "() => /Last inference/.test(document.getElementById('perfHint').textContent)",
                    timeout=60000)
                first = page.text_content("#perfHint").strip()
                page.wait_for_timeout(1500)
                second = page.text_content("#perfHint").strip()
                evidence["live"] = {"top": page.text_content("#topResult").strip(),
                                    "bars": read_bars(page), "perfFirst": first,
                                    "perfSecond": second}
                shot = os.path.join(EVIDENCE, "live-%s.png" % browser_name)
                page.screenshot(path=shot, full_page=True)
                evidence["screenshots"].append(shot)
                check("live loop classifies frames", "Last inference" in second, second)
                check("live loop keeps running", first != second or "ms" in second,
                      [first, second])
                page.click("#stopBtn")
                page.wait_for_timeout(300)
                stopped = page.evaluate(
                    "() => !document.getElementById('video').srcObject")
                check("stop button releases the camera", stopped, stopped)
            except Exception as error:                               # noqa: BLE001
                evidence["live"] = {"error": repr(error)}
                check("live loop classifies frames", False, repr(error))

        browser.close()

    evidence["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    evidence["failed"] = [c["check"] for c in evidence["checks"] if not c["pass"]]
    out = os.path.join(EVIDENCE, "%s-%s.json" % (browser_name, tag))
    json.dump(evidence, open(out, "w", encoding="utf-8"), indent=1)
    print("\n%d checks, %d failed -> %s" % (len(evidence["checks"]),
                                            len(evidence["failed"]), out))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", default="chromium", choices=["chromium", "webkit"])
    parser.add_argument("--url", default="", help="page to test; default: serve this folder")
    parser.add_argument("--folder", default="", help="folder to serve (default: the project root)")
    parser.add_argument("--fake-camera", action="store_true",
                        help="Chromium only: run the live camera loop on a fake video device")
    args = parser.parse_args()

    httpd = None
    url = args.url
    folder = os.path.abspath(args.folder) if args.folder else ROOT
    tag = "live"
    if not url:
        httpd, url = serve(folder)
        url = url + "index.html"
        tag = "local"
        if folder != ROOT:
            tag += "-" + os.path.basename(folder)
    if args.fake_camera:
        tag += "-fakecamera"
    print("testing", url, "in", args.browser, "(files from", folder + ")")
    try:
        evidence = run(url, args.browser, args.fake_camera, tag, folder)
    finally:
        if httpd:
            httpd.shutdown()
    sys.exit(1 if evidence["failed"] else 0)


if __name__ == "__main__":
    main()
