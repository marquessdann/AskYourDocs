const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const documentList = document.getElementById("document-list");

const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const chatLog = document.getElementById("chat-log");

async function loadDocuments() {
  try {
    const res = await fetch(`${API_BASE}/documents`);
    const documents = await res.json();
    documentList.innerHTML = documents
      .map((doc) => `<li>${escapeHtml(doc.filename)} — ${doc.pages} página(s)</li>`)
      .join("");
  } catch (err) {
    console.error("Failed to load documents", err);
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  const submitButton = uploadForm.querySelector("button");
  submitButton.disabled = true;
  uploadStatus.textContent = "Processando PDF (extraindo texto, gerando embeddings)...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/documents`, { method: "POST", body: formData });
    const body = await res.json();

    if (!res.ok) {
      uploadStatus.textContent = `Erro: ${body.detail || "falha ao enviar o documento."}`;
      return;
    }

    uploadStatus.textContent = `"${body.filename}" indexado: ${body.chunks_created} trechos de ${body.pages} páginas.`;
    fileInput.value = "";
    loadDocuments();
  } catch (err) {
    uploadStatus.textContent = "Erro de rede ao enviar o documento.";
    console.error(err);
  } finally {
    submitButton.disabled = false;
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  appendMessage(question, "user");
  questionInput.value = "";
  const submitButton = chatForm.querySelector("button");
  submitButton.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const body = await res.json();

    if (!res.ok) {
      appendMessage(body.detail || "Não foi possível responder essa pergunta.", "error");
      return;
    }

    appendMessage(body.answer, "bot", body.sources, body.cached);
  } catch (err) {
    appendMessage("Erro de rede ao consultar o assistente.", "error");
    console.error(err);
  } finally {
    submitButton.disabled = false;
  }
});

function appendMessage(text, kind, sources, cached) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${kind}`;
  wrapper.textContent = text;

  if (kind === "bot") {
    if (cached) {
      const badge = document.createElement("div");
      badge.className = "sources";
      badge.textContent = "(resposta em cache)";
      wrapper.appendChild(badge);
    }
    if (sources && sources.length > 0) {
      const sourcesEl = document.createElement("div");
      sourcesEl.className = "sources";
      sourcesEl.innerHTML =
        "Fontes:<ul>" +
        sources.map((s) => `<li>${escapeHtml(s.filename)}, página ${s.page}</li>`).join("") +
        "</ul>";
      wrapper.appendChild(sourcesEl);
    }
  }

  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

loadDocuments();
