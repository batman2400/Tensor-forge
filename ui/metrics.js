const LANGUAGE_NAMES = {
  en: "English",
  mixed: "Mixed",
  si: "Sinhala",
  singlish: "Singlish",
  ta: "Tamil",
  tanglish: "Tanglish",
};

const LANGUAGE_ORDER = ["en", "singlish", "si", "ta", "tanglish", "mixed"];
const CHANNEL_ORDER = ["email", "chat", "call_transcript"];

function pretty(value) {
  const text = String(value).replaceAll("_", " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function fixed(value, digits) {
  const factor = 10 ** digits;
  return (Math.round(Number(value) * factor) / factor).toFixed(digits);
}

function orderedEntries(record, order) {
  const names = [
    ...order.filter((name) => name in record),
    ...Object.keys(record).filter((name) => !order.includes(name)),
  ];
  return names.map((name) => [name, record[name]]);
}

function cell(value, className) {
  const td = document.createElement("td");
  td.textContent = value;
  if (className) td.className = className;
  return td;
}

function addRow(table, cells) {
  const tr = document.createElement("tr");
  for (const item of cells) tr.append(item);
  table.querySelector("tbody").appendChild(tr);
}

function meter(fraction) {
  const wrap = document.createElement("div");
  wrap.className = "meter";
  wrap.setAttribute("aria-hidden", "true");
  const span = document.createElement("span");
  span.style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
  wrap.append(span);
  const td = document.createElement("td");
  td.append(wrap);
  return td;
}

function stat(value, label, title) {
  const article = document.createElement("article");
  const strong = document.createElement("strong");
  strong.textContent = value;
  const span = document.createElement("span");
  span.textContent = label;
  if (title) article.title = title;
  article.append(strong, span);
  return article;
}

fetch("metrics.json")
  .then((response) => {
    if (!response.ok) throw new Error("Could not load metrics.");
    return response.json();
  })
  .then((metrics) => {
    const summary = document.querySelector("#summary");
    summary.append(
      stat(String(metrics.n), "Validation tickets"),
      stat(fixed(metrics.accuracy, 3), "Accuracy"),
      stat(fixed(metrics.macro_f1, 3), "Macro-F1"),
      stat(fixed(metrics.ece, 3), "ECE", "Expected calibration error"),
    );

    const classes = document.querySelector("#classes");
    for (const label of metrics.labels) {
      const f1 = metrics.per_class_f1[label];
      const name = cell(pretty(label));
      name.title = label;
      addRow(classes, [
        name,
        cell(fixed(f1, 3), "num"),
        meter(f1),
        cell(String(metrics.per_class_support[label]), "num"),
      ]);
    }

    const languages = document.querySelector("#languages");
    for (const [name, row] of orderedEntries(metrics.by_language, LANGUAGE_ORDER)) {
      addRow(languages, [
        cell(LANGUAGE_NAMES[name] || pretty(name)),
        cell(String(row.n), "num"),
        cell(fixed(row.accuracy, 3), "num"),
        cell(fixed(row.macro_f1, 3), "num"),
      ]);
    }

    const channels = document.querySelector("#channels");
    for (const [name, row] of orderedEntries(metrics.by_channel, CHANNEL_ORDER)) {
      addRow(channels, [
        cell(pretty(name)),
        cell(String(row.n), "num"),
        cell(fixed(row.accuracy, 3), "num"),
        cell(fixed(row.macro_f1, 3), "num"),
      ]);
    }

    const matrix = document.querySelector("#matrix");
    const head = document.createElement("tr");
    head.appendChild(document.createElement("th"));
    for (const label of metrics.labels) {
      const th = document.createElement("th");
      th.textContent = pretty(label);
      th.title = label;
      head.appendChild(th);
    }
    matrix.appendChild(head);
    metrics.confusion_matrix.forEach((counts, index) => {
      const tr = document.createElement("tr");
      const name = document.createElement("th");
      name.textContent = pretty(metrics.labels[index]);
      name.title = metrics.labels[index];
      tr.appendChild(name);
      const rowSum = counts.reduce((sum, count) => sum + count, 0) || 1;
      counts.forEach((count, column) => {
        const td = document.createElement("td");
        td.textContent = count;
        const share = count / rowSum;
        if (count === 0) {
          td.classList.add("zero");
        } else if (column === index) {
          td.style.backgroundColor = `rgba(12, 107, 82, ${0.14 + share * 0.76})`;
          if (share > 0.55) td.style.color = "#f7fff9";
        } else {
          td.style.backgroundColor = `rgba(154, 52, 18, ${0.16 + share * 0.72})`;
          if (share > 0.45) td.style.color = "#fff8f5";
        }
        tr.appendChild(td);
      });
      matrix.appendChild(tr);
    });
  })
  .catch((exc) => {
    const summary = document.querySelector("#summary");
    summary.textContent = exc.message;
  });
