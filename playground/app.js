// sentinel-proof playground. Plain JavaScript, no framework, no build step.
// The page is static; the only network requests are the Pyodide runtime from
// the CDN and the wheel served next to this file. Input text is never sent
// anywhere: the worker (worker.js) verifies it inside the browser.

"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const els = {
    status: $("status"),
    preset: $("preset"),
    share: $("share"),
    passages: $("passages"),
    passagesHint: $("passages-hint"),
    answer: $("answer"),
    question: $("question"),
    style: $("style"),
    max: $("max"),
    verdict: $("verdict"),
    verdictLabel: $("verdict-label"),
    verdictStats: $("verdict-stats"),
    verdictNote: $("verdict-note"),
    claims: $("claims"),
    safeText: $("safe-text"),
    json: $("json"),
    code: $("code"),
    copyTab: $("copy-tab"),
    version: $("version"),
  };

  // ---------------------------------------------------------------- examples

  const EXAMPLES = [
    {
      name: "Refund policy: one figure invented, one citation wrong",
      question: "What is the refund policy?",
      passages: [
        "Customers may request a refund within 30 days of purchase.",
        "Refund requests above 500 USD require manager approval.",
      ],
      answer:
        "Customers can get a refund within 30 days [1]. " +
        "Requests above 500 USD need manager approval [1]. " +
        "Refunds are processed in 2 business days [2].",
      style: "numeric",
      max: 0.5,
    },
    {
      name: "Pricing: the answer is mostly made up, so it is refused",
      question: "How much is the premium plan?",
      passages: [
        "The premium plan costs $1,200 per year with a 4.5% discount for annual billing.",
        "Support is available on weekdays.",
      ],
      answer:
        "The premium plan costs $9,999 per year [1]. It comes with a 7% discount [1]. " +
        "Support is available on weekdays [2].",
      style: "numeric",
      max: 0.5,
    },
    {
      name: "Contract clause: everything checks out",
      question: "How much notice is needed to terminate?",
      passages: [
        "Either party may terminate this agreement with 90 days written notice.",
        "Termination for breach requires a 30 day cure period after written notice of the breach.",
        "Fees already paid are non-refundable on termination.",
      ],
      answer:
        "Either party can terminate with 90 days written notice [1]. " +
        "For breach, the other party gets a 30 day cure period first [2]. " +
        "Fees already paid are not refunded [3].",
      style: "numeric",
      max: 0.5,
    },
    {
      name: "Tag citations: the model writes [chunk:id]",
      question: "",
      passages: [
        "Battery capacity is 5000 mAh and charging takes about 80 minutes.",
        "The device weighs 189 grams.",
      ],
      answer:
        "The battery is 5000 mAh [chunk:p1]. A full charge takes about 80 minutes [chunk:p2]. " +
        "It weighs 189 grams [chunk:p2].",
      style: "tag",
      max: 0.5,
    },
    {
      name: "Model refusal: INSUFFICIENT_CONTEXT passes through",
      question: "What is the parental leave policy for contractors?",
      passages: ["Employees are entitled to 16 weeks of paid parental leave."],
      answer: "INSUFFICIENT_CONTEXT",
      style: "numeric",
      max: 0.5,
    },
  ];

  // ------------------------------------------------------------------- state

  const state = { passages: "", answer: "", question: "", style: "numeric", max: 0.5 };
  let worker = null;
  let ready = false;
  let pendingId = 0;
  let debounceTimer = null;
  let lastVerdict = null;

  function readForm() {
    state.passages = els.passages.value;
    state.answer = els.answer.value;
    state.question = els.question.value;
    state.style = els.style.value === "tag" ? "tag" : "numeric";
    const m = Number.parseFloat(els.max.value);
    state.max = Number.isFinite(m) ? Math.min(1, Math.max(0, m)) : 0.5;
  }

  function writeForm() {
    els.passages.value = state.passages;
    els.answer.value = state.answer;
    els.question.value = state.question;
    els.style.value = state.style;
    els.max.value = String(state.max);
    updateHint();
  }

  function splitPassages(text) {
    return text
      .split(/\n\s*\n/)
      .map((p) => p.trim())
      .filter(Boolean);
  }

  function updateHint() {
    els.passagesHint.textContent =
      els.style.value === "tag"
        ? "one per paragraph, separated by a blank line; cite as [chunk:p1], [chunk:p2]…"
        : "one per paragraph, separated by a blank line; cite as [1], [2]…";
  }

  // --------------------------------------------------------------- share URL

  function encodeState() {
    const json = JSON.stringify({
      v: 1,
      p: state.passages,
      a: state.answer,
      q: state.question,
      s: state.style,
      m: state.max,
    });
    const bytes = new TextEncoder().encode(json);
    let bin = "";
    for (const b of bytes) bin += String.fromCharCode(b);
    return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  function decodeState(hash) {
    try {
      const m = /[#&]s=([A-Za-z0-9_-]+)/.exec(hash);
      if (!m) return false;
      let b64 = m[1].replace(/-/g, "+").replace(/_/g, "/");
      while (b64.length % 4) b64 += "=";
      const bin = atob(b64);
      const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
      const d = JSON.parse(new TextDecoder().decode(bytes));
      if (!d || typeof d !== "object") return false;
      state.passages = typeof d.p === "string" ? d.p : "";
      state.answer = typeof d.a === "string" ? d.a : "";
      state.question = typeof d.q === "string" ? d.q : "";
      state.style = d.s === "tag" ? "tag" : "numeric";
      state.max = typeof d.m === "number" && d.m >= 0 && d.m <= 1 ? d.m : 0.5;
      return true;
    } catch {
      return false;
    }
  }

  function loadExample(index) {
    const ex = EXAMPLES[index];
    if (!ex) return;
    state.passages = ex.passages.join("\n\n");
    state.answer = ex.answer;
    state.question = ex.question;
    state.style = ex.style;
    state.max = ex.max;
    writeForm();
    scheduleVerify();
  }

  // ------------------------------------------------------------------ worker

  function setStatus(text, kind) {
    els.status.textContent = "";
    els.status.className = "status" + (kind ? " " + kind : "");
    if (typeof text === "string") els.status.textContent = text;
    else els.status.append(text);
  }

  function offlineHelp(message) {
    const frag = document.createDocumentFragment();
    frag.append(
      "The browser runtime could not be loaded (" + message + "). " +
        "This usually means the CDN is blocked on your network. The package itself needs nothing but Python: ",
    );
    const code = document.createElement("code");
    code.textContent = "pip install sentinel-proof";
    frag.append(code, ".");
    return frag;
  }

  async function startWorker() {
    let manifest;
    try {
      const res = await fetch("manifest.json", { cache: "no-cache" });
      if (!res.ok) throw new Error("manifest.json returned " + res.status);
      manifest = await res.json();
      if (!manifest || typeof manifest.wheel !== "string") throw new Error("manifest.json has no wheel");
    } catch (err) {
      setStatus(offlineHelp(err.message), "error");
      return;
    }

    const wheelUrl = new URL(manifest.wheel, document.baseURI).href;
    worker = new Worker("worker.js");
    worker.onmessage = (event) => {
      const msg = event.data || {};
      if (msg.type === "status") {
        setStatus(msg.text);
      } else if (msg.type === "ready") {
        ready = true;
        setStatus(
          `Ready. sentinel-proof ${msg.packageVersion} on Python ${msg.pythonVersion} (Pyodide ${msg.pyodideVersion}), running in this tab.`,
          "ready",
        );
        els.version.textContent = `sentinel-proof ${msg.packageVersion} · built ${manifest.built || "locally"}`;
        scheduleVerify(0);
      } else if (msg.type === "result") {
        if (msg.id !== pendingId) return; // stale
        lastVerdict = msg.verdict;
        render(msg.verdict);
      } else if (msg.type === "error") {
        if (msg.phase === "init") {
          ready = false;
          setStatus(offlineHelp(msg.message), "error");
        } else {
          renderError(msg.message);
        }
      }
    };
    worker.onerror = (event) => {
      ready = false;
      setStatus(offlineHelp(event.message || "worker error"), "error");
    };
    worker.postMessage({ type: "init", wheelUrl });
  }

  function scheduleVerify(delay = 250) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runVerify, delay);
  }

  function runVerify() {
    readForm();
    const passages = splitPassages(state.passages);
    const answer = state.answer.trim();
    if (!passages.length || !answer) {
      lastVerdict = null;
      renderPending(
        !passages.length && !answer
          ? "Paste passages and an answer, or pick an example."
          : !passages.length
            ? "Add at least one passage."
            : "Add the answer to check.",
      );
      return;
    }
    if (!ready || !worker) return;
    pendingId += 1;
    worker.postMessage({
      type: "verify",
      id: pendingId,
      payload: {
        passages,
        answer: state.answer,
        question: state.question,
        style: state.style,
        max_stripped: state.max,
      },
    });
    renderCode();
  }

  // ------------------------------------------------------------------ render

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function passageLabel(id) {
    // Passages are p1..pn in retrieval order. In numeric style the user wrote
    // [n]; show it that way. A citation the model made up (a passage that was
    // never retrieved) arrives as "missing-<n>".
    if (typeof id !== "string") return "?";
    const missing = /^missing-(.+)$/.exec(id);
    if (missing) return state.style === "numeric" ? `[${missing[1]}]` : `[chunk:${missing[1]}]`;
    if (state.style === "numeric") {
      const m = /^p(\d+)$/.exec(id);
      if (m) return `[${m[1]}]`;
    }
    return `[chunk:${id}]`;
  }

  function renderPending(note) {
    els.verdict.className = "verdict pending";
    els.verdictLabel.textContent = "Waiting for input";
    els.verdictStats.textContent = "";
    els.verdictNote.textContent = note;
    clear(els.claims);
    els.claims.append(el("li", "empty", "Nothing checked yet."));
    els.safeText.textContent = "";
    els.safeText.className = "safe-text";
    els.json.textContent = "";
    renderCode();
  }

  function renderError(message) {
    els.verdict.className = "verdict fail";
    els.verdictLabel.textContent = "Could not verify";
    els.verdictStats.textContent = "";
    els.verdictNote.textContent = message;
  }

  function render(v) {
    const claims = Array.isArray(v.claims) ? v.claims : [];
    const counts = { supported: 0, repaired: 0, stripped: 0 };
    for (const c of claims) if (c.status in counts) counts[c.status] += 1;

    if (v.is_refusal) {
      els.verdict.className = "verdict refusal";
      els.verdictLabel.textContent = "Model refused";
      els.verdictStats.textContent = "";
      els.verdictNote.textContent =
        "The model answered with the refusal token, so there is no claim to check. The gate passes the refusal through with reason " +
        (v.refusal || "INSUFFICIENT_EVIDENCE") +
        ".";
    } else if (v.ok) {
      els.verdict.className = "verdict ok";
      els.verdictLabel.textContent = "Verified";
      els.verdictStats.textContent =
        `${counts.supported + counts.repaired} of ${claims.length} claims supported · ` +
        `${v.repaired_citations} citation${v.repaired_citations === 1 ? "" : "s"} repaired · ` +
        `${Math.round((v.stripped_ratio || 0) * 100)}% stripped (limit ${Math.round(state.max * 100)}%)`;
      els.verdictNote.textContent =
        counts.stripped || counts.repaired
          ? "The answer ships with the unsupported parts removed and the citations corrected."
          : "Every sentence is supported by the passage it cites.";
    } else {
      els.verdict.className = "verdict fail";
      els.verdictLabel.textContent = "Refused";
      els.verdictStats.textContent =
        `${counts.stripped} of ${claims.length} claims stripped · ` +
        `${Math.round((v.stripped_ratio || 0) * 100)}% (limit ${Math.round(state.max * 100)}%)`;
      els.verdictNote.textContent =
        v.refusal === "UNVERIFIABLE_ANSWER" && claims.length === 0
          ? "No citations could be parsed, so nothing can be checked: the gate fails closed."
          : `Too much of the answer is unsupported. Reason code: ${v.refusal || "UNVERIFIABLE_ANSWER"}. The draft is not shown to the user.`;
    }

    clear(els.claims);
    if (!claims.length) {
      els.claims.append(el("li", "empty", v.is_refusal ? "No claims: the model refused." : "No claims were found."));
    }
    for (const c of claims) {
      const li = el("li", "claim " + c.status);
      li.append(el("span", "badge", c.status));
      const body = el("div");
      body.append(el("p", "claim-text", c.text || "(no text: a citation with nothing in front of it)"));
      const why = el("p", "claim-why");
      const cited = (c.cited || []).map(passageLabel).join(" ") || "nothing";
      const supportedBy = (c.supported_by || []).map(passageLabel).join(" ");
      if (c.status === "supported") {
        why.append("Supported by ");
        why.append(el("span", "cite", supportedBy || cited));
        why.append(".");
      } else if (c.status === "repaired") {
        why.append("Cited ");
        why.append(el("span", "cite", cited));
        why.append(el("span", "arrow", "→"));
        why.append("re-pointed to ");
        why.append(el("span", "cite", supportedBy));
        why.append(", the passage that supports it.");
      } else {
        const nums = c.unsupported_numbers || [];
        if (nums.length) {
          why.append("Figures not in the cited evidence: ");
          nums.forEach((n, i) => {
            if (i) why.append(", ");
            why.append(el("span", "num", n));
          });
          why.append(". ");
        }
        why.append(
          nums.length
            ? "No retrieved passage contains them, so the sentence is removed."
            : "No retrieved passage shares enough of its words, so the sentence is removed.",
        );
      }
      body.append(why);
      li.append(body);
      els.claims.append(li);
    }

    const safe = typeof v.safe_text === "string" ? v.safe_text : "";
    els.safeText.textContent = safe;
    els.safeText.className = "safe-text" + (v.passed ? "" : " refused");

    els.json.textContent = JSON.stringify(v, null, 2);
    renderCode();
  }

  function pyString(s) {
    // JSON string literals are valid Python string literals for the
    // characters that matter here; this keeps the snippet copy-paste safe.
    return JSON.stringify(s);
  }

  function renderCode() {
    const passages = splitPassages(state.passages);
    const lines = [
      "# pip install sentinel-proof",
      "from sentinel_proof import passages_from_texts, verify",
      "",
      "passages = passages_from_texts([",
      ...passages.map((p) => "    " + pyString(p) + ","),
      "])",
      "answer = " + pyString(state.answer),
      "",
      "v = verify(",
      "    answer,",
      "    passages,",
      ...(state.question.trim() ? ["    question=" + pyString(state.question) + ","] : []),
      '    citation_style="' + state.style + '",',
      "    max_stripped_ratio=" + state.max + ",",
      ")",
      "print(v.ok, v.refusal)",
      "print(v.safe_text())",
      "for claim in v.claims:",
      "    print(claim.status.value, claim.text, claim.supported_by)",
    ];
    els.code.textContent = lines.join("\n");
  }

  // ----------------------------------------------------------------- wiring

  let toastTimer = null;
  function toast(text) {
    let node = document.querySelector(".toast");
    if (!node) {
      node = el("div", "toast");
      document.body.append(node);
    }
    node.textContent = text;
    node.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => node.classList.remove("show"), 1800);
  }

  async function copy(text) {
    try {
      await navigator.clipboard.writeText(text);
      toast("Copied");
    } catch {
      toast("Copy failed: your browser blocked clipboard access");
    }
  }

  for (const [i, ex] of EXAMPLES.entries()) {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = ex.name;
    els.preset.append(opt);
  }
  els.preset.addEventListener("change", () => {
    if (els.preset.value !== "") loadExample(Number(els.preset.value));
  });

  for (const input of [els.passages, els.answer, els.question]) {
    input.addEventListener("input", () => {
      els.preset.value = "";
      scheduleVerify();
    });
  }
  els.style.addEventListener("change", () => {
    updateHint();
    scheduleVerify(0);
  });
  els.max.addEventListener("input", () => scheduleVerify());

  els.share.addEventListener("click", () => {
    readForm();
    const url = new URL(window.location.href);
    url.hash = "s=" + encodeState();
    history.replaceState(null, "", url.toString());
    copy(url.toString());
  });

  let activeTab = "json";
  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => {
      activeTab = tab.dataset.tab;
      for (const t of document.querySelectorAll(".tab")) {
        const on = t === tab;
        t.classList.toggle("active", on);
        t.setAttribute("aria-selected", on ? "true" : "false");
      }
      for (const p of document.querySelectorAll(".panel")) p.classList.toggle("hidden", p.dataset.panel !== activeTab);
    });
  }
  els.copyTab.addEventListener("click", () => copy(activeTab === "json" ? els.json.textContent : els.code.textContent));

  // Initial state: a shared link wins, otherwise the first example.
  if (decodeState(window.location.hash)) {
    writeForm();
  } else {
    els.preset.value = "0";
    loadExample(0);
  }
  readForm();
  renderPending("Starting…");
  startWorker();

  // Exposed for the browser test; not part of any API.
  window.__playground = {
    isReady: () => ready,
    lastVerdict: () => lastVerdict,
    loadExample,
    examples: EXAMPLES.length,
  };
})();
