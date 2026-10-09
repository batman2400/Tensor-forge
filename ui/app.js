const SAMPLES = [
  {
    language: "English",
    channel: "email",
    subject: "Charged twice",
    text: "I was charged twice for last night's ride. Please refund the extra payment.",
  },
  {
    language: "Singlish",
    channel: "chat",
    subject: "",
    text: "Eh the food never come leh, already 40 min. Can refund or not?",
  },
  {
    language: "Sinhala",
    channel: "chat",
    subject: "",
    text: "මගේ order එක නැත. මුදල් refund කරන්න.",
  },
  {
    language: "Tamil",
    channel: "call_transcript",
    subject: "",
    text: "ஆர்டர் வரல. பணம் திரும்ப வேண்டும்.",
  },
  {
    language: "Tanglish",
    channel: "chat",
    subject: "",
    text: "Order late aa iruku, refund pannunga please.",
  },
  {
    language: "Mixed",
    channel: "email",
    subject: "Login code",
    text: "OTP never arrived and I cannot log in. The ride receipt is also wrong.",
  },
];

const keyInput = document.querySelector("#key");
const textInput = document.querySelector("#text");
const stored = sessionStorage.getItem("tf_api_key");
if (stored) keyInput.value = stored;
keyInput.addEventListener("change", () => {
  sessionStorage.setItem("tf_api_key", keyInput.value);
});

function pretty(value) {
  const text = String(value).replaceAll("_", " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function fixed(value, digits) {
  const factor = 10 ** digits;
  return (Math.round(Number(value) * factor) / factor).toFixed(digits);
}

function syncCount() {
  const count = document.querySelector("#count");
  count.textContent = `${textInput.value.length.toLocaleString()} / 10,000`;
  count.classList.toggle("warn", textInput.value.length > 9000);
}

function syncPredict() {
  document.querySelector("#predict").disabled = textInput.value.trim().length === 0;
}

textInput.addEventListener("input", () => {
  syncCount();
  syncPredict();
});
syncCount();
syncPredict();

const sampleBox = document.querySelector("#samples");
for (const sample of SAMPLES) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ghost";
  button.textContent = sample.language;
  button.setAttribute("aria-pressed", "false");
  button.addEventListener("click", () => {
    for (const other of sampleBox.querySelectorAll("button")) {
      other.setAttribute("aria-pressed", "false");
    }
    button.setAttribute("aria-pressed", "true");
    document.querySelector("#channel").value = sample.channel;
    document.querySelector("#subject").value = sample.subject;
    textInput.value = sample.text;
    textInput.dispatchEvent(new Event("input"));
  });
  sampleBox.appendChild(button);
}

function key() {
  const value = keyInput.value;
  sessionStorage.setItem("tf_api_key", value);
  return value;
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  const value = key();
  if (value) headers["X-API-Key"] = value;
  const response = await fetch(path, { ...options, headers });
  const body = await response.json();
  if (!response.ok) {
    const message = body.error ? body.error.message : response.statusText;
    throw new Error(message);
  }
  return body;
}

function setFlag(selector, on, yes, no) {
  const el = document.querySelector(selector);
  el.textContent = on ? yes : no;
  el.className = on ? "pill warn" : "pill quiet";
}

function showPrediction(prediction) {
  document.querySelector("#single-empty").hidden = true;
  document.querySelector("#single-result").hidden = false;
  document.querySelector("#category-code").textContent = prediction.category;
  document.querySelector("#category").textContent = pretty(prediction.category);
  document.querySelector("#team").textContent = prediction.team;
  document.querySelector("#secondary").textContent = prediction.secondary_category
    ? pretty(prediction.secondary_category)
    : "None";
  setFlag("#urgent", prediction.is_urgent, "Urgent", "Not urgent");
  setFlag("#review", prediction.needs_human_review, "Needs review", "No review");
  const confidence = Number(prediction.confidence);
  document.querySelector("#confidence").textContent = fixed(confidence, 2);
  document.querySelector("#bar").style.width = `${Math.round(confidence * 100)}%`;
  const meter = document.querySelector("#meter");
  meter.setAttribute("aria-valuenow", fixed(confidence, 2));
  meter.setAttribute("aria-valuetext", `${Math.round(confidence * 100)} percent`);
  document.querySelector("#version").textContent = prediction.model_version;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.querySelector("#verdict").scrollIntoView({
    block: "nearest",
    behavior: reduce ? "auto" : "smooth",
  });
}

document.querySelector("#ticket-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const error = document.querySelector("#single-error");
  const button = document.querySelector("#predict");
  const emptyTitle = document.querySelector("#empty-title");
  error.hidden = true;
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  button.textContent = "Predicting…";
  if (document.querySelector("#single-result").hidden) {
    emptyTitle.textContent = "Classifying…";
  }
  try {
    const prediction = await api("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        channel: document.querySelector("#channel").value,
        subject: document.querySelector("#subject").value,
        text: textInput.value,
      }),
    });
    showPrediction(prediction);
  } catch (exc) {
    error.hidden = false;
    error.textContent = exc.message;
    if (document.querySelector("#single-result").hidden) {
      emptyTitle.textContent = "Waiting for a ticket";
    }
  } finally {
    button.textContent = "Predict";
    button.removeAttribute("aria-busy");
    syncPredict();
  }
});

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"') {
        if (text[i + 1] === '"') {
          cell += '"';
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        cell += char;
      }
      continue;
    }
    if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(cell);
      cell = "";
    } else if (char === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else if (char !== "\r") {
      cell += char;
    }
  }
  if (cell.length || row.length) {
    row.push(cell);
    rows.push(row);
  }
  if (!rows.length) return [];
  const header = rows[0].map((item) => item.trim());
  return rows.slice(1).filter((items) => items.some((item) => item.trim())).map((items) => {
    const record = {};
    header.forEach((name, index) => {
      record[name] = items[index] || "";
    });
    return record;
  });
}

let lastResults = [];

function renderResults(predictions) {
  lastResults = predictions;
  const table = document.querySelector("#results");
  const body = table.querySelector("tbody");
  body.replaceChildren();
  for (const prediction of predictions) {
    const tr = document.createElement("tr");
    const cells = [
      prediction.ticket_id || "",
      prediction.category,
      prediction.secondary_category || "",
      prediction.team,
      prediction.is_urgent ? "yes" : "no",
      fixed(Number(prediction.confidence), 2),
    ];
    cells.forEach((value, index) => {
      const td = document.createElement("td");
      td.textContent = value;
      if (index === cells.length - 1) td.className = "num";
      tr.appendChild(td);
    });
    body.appendChild(tr);
  }
  table.hidden = false;
  document.querySelector("#download").hidden = false;
}

function setStatus(text, state) {
  const status = document.querySelector("#job-status");
  status.textContent = text;
  if (state) status.dataset.state = state;
  else delete status.dataset.state;
}

async function runJob(tickets) {
  const submitted = await api("/batch/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tickets }),
  });
  let view = submitted;
  while (view.status === "queued" || view.status === "running") {
    setStatus(`${view.status}: ${view.processed} / ${view.total}`, "wait");
    await new Promise((resolve) => setTimeout(resolve, 2000));
    view = await api(`/batch/jobs/${submitted.job_id}`);
  }
  if (view.status !== "succeeded") {
    throw new Error(view.error ? view.error.message : view.status);
  }
  setStatus(`succeeded: ${view.processed} / ${view.total}`, "ok");
  const page = await api(`/batch/jobs/${submitted.job_id}/results`);
  renderResults(page.predictions);
}

async function handleCsvFile(file) {
  const error = document.querySelector("#csv-error");
  error.hidden = true;
  setStatus("");
  if (!file) return;
  document.querySelector("#file-name").textContent = file.name;
  try {
    const records = parseCsv(await file.text());
    const tickets = records.map((record) => ({
      ticket_id: record.ticket_id,
      channel: record.channel,
      subject: record.subject || "",
      text: record.text,
    }));
    if (!tickets.length) throw new Error("The CSV has no data rows.");
    if (tickets.length > 5000) throw new Error("A job accepts at most 5,000 tickets.");
    if (tickets.length <= 100) {
      setStatus("Predicting…", "wait");
      const body = await api("/predict/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickets }),
      });
      setStatus(`batch: ${body.meta.count} tickets`, "ok");
      renderResults(body.predictions);
      return;
    }
    await runJob(tickets);
  } catch (exc) {
    error.hidden = false;
    error.textContent = exc.message;
    setStatus("");
  }
}

const csvInput = document.querySelector("#csv");
csvInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  await handleCsvFile(file);
  event.target.value = "";
});

const drop = document.querySelector(".drop");
drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.classList.add("drag");
});
drop.addEventListener("dragleave", () => {
  drop.classList.remove("drag");
});
drop.addEventListener("drop", async (event) => {
  event.preventDefault();
  drop.classList.remove("drag");
  const file = event.dataTransfer.files[0];
  await handleCsvFile(file);
});

document.querySelector("#download").addEventListener("click", () => {
  const header = ["ticket_id", "category", "secondary_category", "team", "is_urgent", "confidence"];
  const lines = [header.join(",")];
  for (const prediction of lastResults) {
    const cells = [
      prediction.ticket_id || "",
      prediction.category,
      prediction.secondary_category || "",
      prediction.team,
      prediction.is_urgent ? "true" : "false",
      prediction.confidence,
    ].map((value) => `"${String(value).replaceAll('"', '""')}"`);
    lines.push(cells.join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "tensorforge-results.csv";
  link.click();
  URL.revokeObjectURL(link.href);
});
