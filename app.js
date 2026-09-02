"use strict";

const MEASURES = [
  { key: "cooc", tableId: "table-cooc", label: "pairs well with" },
  { key: "chem", tableId: "table-chem", label: "shares flavor profile" },
  { key: "core", tableId: "table-core", label: "similar to" },
];
const BAND_NAMES = ["cold", "tepid", "warm", "hot"];
const MAX_SUGGESTIONS = 8;

let vocab = [];          // canonical names, vocab-index order (underscored)
let displayNames = [];   // underscores -> spaces
let puzzle = null;       // today's data file content
let guesses = [];        // vocab indices in guess order
let won = false;
let hints = 0;           // hints taken today
let suggestionIndices = []; // vocab indices currently shown in the dropdown
let activeSuggestion = -1;  // position within suggestionIndices
let noticeTimer = null;

const input = () => document.getElementById("guess-input");
const storageKey = () => `epicuriosimi:${puzzle.date}`;

function todayStr() {
  const now = new Date();
  const m = String(now.getMonth() + 1).padStart(2, "0");
  const d = String(now.getDate()).padStart(2, "0");
  return `${now.getFullYear()}-${m}-${d}`;
}

function loadState() {
  try {
    const raw = localStorage.getItem(storageKey());
    if (!raw) return;
    const s = JSON.parse(raw);
    guesses = (s.guesses || []).filter(
      (i) => Number.isInteger(i) && i >= 0 && i < vocab.length
    );
    won = Boolean(s.won);
    hints = Number.isInteger(s.hints) && s.hints >= 0 ? s.hints : 0;
  } catch (e) {
    /* storage unavailable: play without persistence */
  }
}

function saveState() {
  try {
    localStorage.setItem(storageKey(), JSON.stringify({ guesses, won, hints }));
  } catch (e) {
    /* ignore */
  }
}

function showNotice(text, sticky = false) {
  const el = document.getElementById("notice");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(noticeTimer);
  if (!sticky) noticeTimer = setTimeout(() => { el.hidden = true; }, 2500);
}

function hideNotice() {
  clearTimeout(noticeTimer);
  document.getElementById("notice").hidden = true;
}

function updateSuggestions() {
  const q = input().value.trim().toLowerCase();
  suggestionIndices = [];
  if (q) {
    // prefix matches first, then substring matches
    for (let i = 0; i < displayNames.length && suggestionIndices.length < MAX_SUGGESTIONS; i++) {
      if (displayNames[i].startsWith(q)) suggestionIndices.push(i);
    }
    for (let i = 0; i < displayNames.length && suggestionIndices.length < MAX_SUGGESTIONS; i++) {
      if (!displayNames[i].startsWith(q) && displayNames[i].includes(q)) {
        suggestionIndices.push(i);
      }
    }
  }
  activeSuggestion = suggestionIndices.length ? 0 : -1;
  renderSuggestions();
}

function renderSuggestions() {
  const listEl = document.getElementById("suggestions");
  listEl.innerHTML = "";
  listEl.hidden = suggestionIndices.length === 0;
  suggestionIndices.forEach((vocabIdx, pos) => {
    const li = document.createElement("li");
    li.textContent = displayNames[vocabIdx];
    if (pos === activeSuggestion) li.classList.add("active");
    li.addEventListener("mousedown", (e) => {
      e.preventDefault(); // keep input focus
      submitGuess(vocabIdx);
    });
    listEl.appendChild(li);
  });
}

function clearSuggestions() {
  suggestionIndices = [];
  activeSuggestion = -1;
  renderSuggestions();
}

function onKeyDown(e) {
  if (suggestionIndices.length === 0) return;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    activeSuggestion = (activeSuggestion + 1) % suggestionIndices.length;
    renderSuggestions();
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    activeSuggestion =
      (activeSuggestion - 1 + suggestionIndices.length) % suggestionIndices.length;
    renderSuggestions();
  } else if (e.key === "Escape") {
    clearSuggestions();
  }
}

function onSubmit(e) {
  e.preventDefault();
  if (won) return;
  let idx = -1;
  if (activeSuggestion >= 0) {
    idx = suggestionIndices[activeSuggestion];
  } else {
    idx = displayNames.indexOf(input().value.trim().toLowerCase());
  }
  if (idx < 0) {
    showNotice("pick an ingredient from the list");
    return;
  }
  submitGuess(idx);
}

function submitGuess(idx) {
  hideNotice();
  if (guesses.includes(idx)) {
    showNotice(`already guessed ${displayNames[idx]}`);
    input().value = "";
    clearSuggestions();
    return;
  }
  guesses.push(idx);
  if (idx === puzzle.target) won = true;
  saveState();
  input().value = "";
  clearSuggestions();
  renderTables();
  if (won) finishGame(true);
}

function updateGuessCount() {
  const n = guesses.length;
  let text = n ? `, ${n} ${n === 1 ? "guess" : "guesses"}` : "";
  if (hints) text += `, ${hints} ${hints === 1 ? "hint" : "hints"}`;
  document.getElementById("guess-count").textContent = text;
}

function takeHint() {
  if (won) return;
  const shuffled = [...MEASURES].sort(() => Math.random() - 0.5);
  for (const { key, label } of shuffled) {
    const candidates = [];
    for (let i = 0; i < vocab.length; i++) {
      if (puzzle.bands[key][i] === 3 && i !== puzzle.target && !guesses.includes(i)) {
        candidates.push(i);
      }
    }
    if (candidates.length) {
      const idx = candidates[Math.floor(Math.random() * candidates.length)];
      hints += 1;
      submitGuess(idx);
      showNotice(`hint: ${displayNames[idx]} runs hot in “${label}”`, true);
      return;
    }
  }
  showNotice("no hints left — every hot ingredient is already on the board");
}

function renderTables() {
  updateGuessCount();
  const last = guesses[guesses.length - 1];
  for (const { key, tableId } of MEASURES) {
    const tbody = document.querySelector(`#${tableId} tbody`);
    tbody.innerHTML = "";
    const sorted = [...guesses].sort(
      (a, b) => puzzle.scores[key][b] - puzzle.scores[key][a]
    );
    for (const idx of sorted) {
      const tr = document.createElement("tr");
      if (idx === last) tr.classList.add("latest");
      const nameTd = document.createElement("td");
      nameTd.textContent = displayNames[idx];
      const scoreTd = document.createElement("td");
      scoreTd.className = "num";
      scoreTd.textContent = (puzzle.scores[key][idx] * 100).toFixed(2);
      const bandTd = document.createElement("td");
      const band = BAND_NAMES[puzzle.bands[key][idx]];
      const badge = document.createElement("span");
      badge.className = `band band-${band}`;
      badge.textContent = band;
      bandTd.appendChild(badge);
      tr.append(nameTd, scoreTd, bandTd);
      tbody.appendChild(tr);
    }
  }
}

function finishGame(celebrate) {
  input().disabled = true;
  document.querySelector("#guess-form button[type=submit]").disabled = true;
  document.getElementById("hint-button").disabled = true;
  clearSuggestions();
  if (!celebrate) return;
  confetti({ particleCount: 160, spread: 80, origin: { y: 0.6 },
             disableForReducedMotion: true });
  setTimeout(() => confetti({ particleCount: 80, spread: 120, origin: { y: 0.4 },
                              disableForReducedMotion: true }), 300);
  const n = guesses.length;
  const hintNote = hints ? ` (${hints} ${hints === 1 ? "hint" : "hints"})` : "";
  document.getElementById("win-text").textContent =
    `${displayNames[puzzle.target]} in ${n} ${n === 1 ? "guess" : "guesses"}${hintNote}!`;
  document.getElementById("win-modal").hidden = false;
}

async function init() {
  // ?date=YYYY-MM-DD plays that day's puzzle instead of today's
  const override = new URLSearchParams(location.search).get("date");
  const dateStr = /^\d{4}-\d{2}-\d{2}$/.test(override || "") ? override : todayStr();
  try {
    const [vres, dres] = await Promise.all([
      fetch("vocab.json"),
      fetch(`data/${dateStr}.json`),
    ]);
    if (!vres.ok || !dres.ok) throw new Error("missing puzzle files");
    vocab = await vres.json();
    puzzle = await dres.json();
  } catch (e) {
    const msg = document.getElementById("message");
    msg.textContent =
      `no puzzle for today (${dateStr}) — run: python precompute/generate_daily.py`;
    msg.hidden = false;
    document.getElementById("guess-form").hidden = true;
    return;
  }
  displayNames = vocab.map((w) => w.replace(/_/g, " "));
  document.getElementById("puzzle-number").textContent = `puzzle #${puzzle.puzzle_number}`;
  loadState();
  renderTables();
  if (won) finishGame(false);

  input().addEventListener("input", updateSuggestions);
  input().addEventListener("keydown", onKeyDown);
  input().addEventListener("blur", () => setTimeout(clearSuggestions, 150));
  document.getElementById("guess-form").addEventListener("submit", onSubmit);
  document.getElementById("win-close").addEventListener("click", () => {
    document.getElementById("win-modal").hidden = true;
  });

  document.getElementById("hint-button").addEventListener("click", takeHint);

  const helpModal = document.getElementById("help-modal");
  document.getElementById("help-open").addEventListener("click", () => {
    helpModal.hidden = false;
  });
  document.getElementById("help-close").addEventListener("click", () => {
    helpModal.hidden = true;
  });
  try {
    if (!localStorage.getItem("epicuriosimi:help-seen")) {
      localStorage.setItem("epicuriosimi:help-seen", "1");
      helpModal.hidden = false;
    }
  } catch (e) {
    /* storage unavailable: open help only via the button */
  }
}

init();
