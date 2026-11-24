// Chat UI: streams /api/ask/stream (Server-Sent Events) and renders the answer.
// All model and user text goes through textContent / DOM nodes, never innerHTML.
"use strict";

const form = document.getElementById("ask");
const input = document.getElementById("question");
const send = document.getElementById("send");
const chat = document.getElementById("chat");
const template = document.getElementById("answer-template");
const CITATION = /\[([A-Z][A-Z0-9]*-[A-Za-z0-9-]+)\]/g;

fetch("/health")
  .then((r) => (r.ok ? r.json() : null))
  .then((h) => {
    if (h && h.notice) document.getElementById("notice").textContent = h.notice;
  })
  .catch(() => {});

document.querySelectorAll("#examples button").forEach((b) =>
  b.addEventListener("click", () => {
    input.value = b.textContent;
    input.focus();
  }),
);

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  ask(question);
  input.value = "";
});

function ask(question) {
  const node = template.content.firstElementChild.cloneNode(true);
  node.querySelector(".question").textContent = question;
  chat.appendChild(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });

  const progress = node.querySelector(".progress");
  const text = node.querySelector(".text");
  const started = performance.now();
  let streamed = "";
  send.disabled = true;

  const url = "/api/ask/stream?question=" + encodeURIComponent(question);
  const source = new EventSource(url);

  source.addEventListener("progress", (e) => {
    const data = JSON.parse(e.data);
    progress.querySelectorAll("li").forEach((li) => li.classList.add("done"));
    if (data.stage === "fallback") {
      // the next model starts its answer from scratch
      streamed = "";
      text.textContent = "";
    }
    const li = document.createElement("li");
    li.textContent = data.message || data.stage;
    progress.appendChild(li);
  });

  source.addEventListener("token", (e) => {
    streamed += JSON.parse(e.data).text;
    text.textContent = streamed;
  });

  source.addEventListener("result", (e) => {
    source.close();
    render(node, JSON.parse(e.data), performance.now() - started);
    send.disabled = false;
  });

  source.addEventListener("error", (e) => {
    source.close();
    let message = "Mất kết nối tới máy chủ. Vui lòng thử lại.";
    if (e.data) {
      try {
        message = JSON.parse(e.data).message || message;
      } catch (_) {}
    }
    text.textContent = message;
    text.classList.add("error");
    send.disabled = false;
  });
}

function render(node, result, elapsedMs) {
  node.querySelector(".progress").hidden = true;
  const sources = new Map((result.sources || []).map((s) => [s.chunk_id, s]));
  const cited = result.citations || [];

  // Answer text with citation ids turned into buttons that open the matching card.
  const text = node.querySelector(".text");
  text.textContent = "";
  let last = 0;
  for (const match of result.answer.matchAll(CITATION)) {
    text.appendChild(document.createTextNode(result.answer.slice(last, match.index)));
    const id = match[1];
    if (sources.has(id)) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "cite";
      chip.textContent = shortLabel(sources.get(id));
      chip.title = id;
      chip.addEventListener("click", () => openCard(node, id));
      text.appendChild(chip);
    }
    last = match.index + match[0].length;
  }
  text.appendChild(document.createTextNode(result.answer.slice(last)));

  // Amendment warnings (attached by the server from the cited articles).
  const warnings = result.warnings || [];
  if (warnings.length) {
    const box = node.querySelector(".warnings");
    box.hidden = false;
    for (const w of warnings) {
      const li = document.createElement("li");
      li.textContent = w;
      box.querySelector("ul").appendChild(li);
    }
  }

  // Source cards for the cited chunks.
  if (cited.length) {
    const box = node.querySelector(".sources");
    box.hidden = false;
    for (const id of cited) {
      const s = sources.get(id);
      if (s) box.querySelector(".cards").appendChild(card(s));
    }
  }

  const seconds = (elapsedMs / 1000).toFixed(1).replace(".", ",");
  const parts = [];
  if (result.model) parts.push("Mô hình: " + result.model);
  parts.push("Thời gian: " + seconds + " giây");
  if (result.status === "not_found") parts.push("Không tìm thấy quy định phù hợp");
  node.querySelector(".meta").textContent = parts.join(" · ");
}

function shortLabel(s) {
  const doc = s.doc_short === "ND145" ? "NĐ 145" : "BLLĐ";
  if (s.unit === "appendix") return doc + " PL " + s.appendix_no;
  const clauses = s.clause_nos && s.clause_nos.length ? " k." + s.clause_nos.join(",") : "";
  return doc + " Đ." + s.article_no + clauses;
}

function card(s) {
  const d = document.createElement("details");
  d.className = "card";
  d.dataset.id = s.chunk_id;
  const summary = document.createElement("summary");
  summary.textContent =
    s.unit === "appendix"
      ? s.doc_title + " - Phụ lục " + s.appendix_no
      : s.doc_title + " - Điều " + s.article_no + ". " + s.article_title;
  d.appendChild(summary);
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = s.content;
  d.appendChild(body);
  for (const note of [...(s.amended_by || []), ...(s.upcoming_change || [])]) {
    const p = document.createElement("p");
    p.className = "note";
    p.textContent = "Lưu ý: " + note;
    d.appendChild(p);
  }
  if (s.source_url) {
    const a = document.createElement("a");
    a.href = s.source_url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = "Văn bản gốc trên cổng thông tin chính thức";
    d.appendChild(a);
  }
  return d;
}

function openCard(node, id) {
  const el = node.querySelector('details.card[data-id="' + CSS.escape(id) + '"]');
  if (el) {
    el.open = true;
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}
