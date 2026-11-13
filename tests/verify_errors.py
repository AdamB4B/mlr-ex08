"""The two failure paths of the page, tested rather than assumed.

    python tests/verify_errors.py

  model 404        serve a copy of the page without leaf_model.onnx and check that the page says
                   so instead of waiting forever, and that the bars stay empty
  camera denied    load the real page in a browser with no camera and press Start camera; the page
                   must explain it and the upload must still work afterwards

Writes tests/evidence/errors.json.
"""
import json
import os
import shutil
import sys
import tempfile
import time

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_page import EVIDENCE, MODEL_READY, ROOT, serve  # noqa: E402

FILES_WITHOUT_MODEL = ["index.html", "model_info.json", "check_expected.json"]


def main():
    os.makedirs(EVIDENCE, exist_ok=True)
    evidence = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": []}

    def check(name, ok, detail):
        evidence["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
        print(("PASS  " if ok else "FAIL  ") + name + "  " + str(detail))

    staging = tempfile.mkdtemp(prefix="leaf-no-model-")
    for name in FILES_WITHOUT_MODEL:
        shutil.copy2(os.path.join(ROOT, name), staging)
    httpd, base = serve(staging)
    httpd_live, base_live = serve(ROOT)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # ---------------------------------------------------------------- model 404
        page.goto(base + "index.html", wait_until="domcontentloaded")
        page.wait_for_function(
            "() => !document.getElementById('error').hidden", timeout=60000)
        text = page.text_content("#error").strip()
        status = page.text_content("#statusText").strip()
        bars = page.eval_on_selector_all("#bars li", "els => els.length")
        evidence["model404"] = {"error": text, "status": status, "barCount": bars}
        values = page.eval_on_selector_all(
            "#bars .value", "els => els.map(e => e.textContent)")
        check("a missing model is reported, not hidden", "404" in text or "failed" in text.lower(),
              text.replace("\n", " ")[:160])
        check("the status says the model failed", "failed" in status.lower(), status)
        check("no probabilities are shown without a model",
              bars == 4 and all(v == "\u2014" for v in values), values)

        # ---------------------------------------------------------------- camera denied
        # The headless browser has no camera at all; a mocked NotAllowedError is what a phone
        # shows when the permission prompt is refused, so the message mapping is tested here.
        context = browser.new_context()
        context.add_init_script("""
            navigator.mediaDevices.getUserMedia = () => {
              const e = new Error('Permission denied');
              e.name = 'NotAllowedError';
              return Promise.reject(e);
            };
        """)
        page = context.new_page()
        page.goto(base_live + "index.html", wait_until="domcontentloaded")
        page.wait_for_function(MODEL_READY, timeout=120000)
        page.click("#startBtn")
        page.wait_for_function(
            "() => !document.getElementById('error').hidden", timeout=30000)
        text = page.text_content("#error").strip()
        evidence["cameraDenied"] = {"error": text}
        check("a refused camera is explained", "permission" in text.lower(),
              text.replace("\n", " ")[:160])
        check("the message offers the upload instead", "upload" in text.lower(), text)
        check("the model still works after the camera error",
              "ready" in page.text_content("#statusText").lower(),
              page.text_content("#statusText"))

        page.set_input_files("#fileInput", os.path.join(ROOT, "check_image.jpg"))
        page.wait_for_function(
            "() => { const v = document.querySelector('#bars .value');"
            " return !!v && v.textContent !== '\\u2014'; }", timeout=60000)
        top = page.text_content("#topResult").strip()
        evidence["afterCameraError"] = {"top": top}
        check("upload still classifies after the camera error", top.startswith("apple"), top)

        browser.close()

    httpd.shutdown()
    httpd_live.shutdown()
    shutil.rmtree(staging, ignore_errors=True)
    failed = [c["check"] for c in evidence["checks"] if not c["pass"]]
    out = os.path.join(EVIDENCE, "errors.json")
    json.dump(evidence, open(out, "w", encoding="utf-8"), indent=1)
    print("\n%d checks, %d failed -> %s" % (len(evidence["checks"]), len(failed), out))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
