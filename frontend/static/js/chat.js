// Wait until DOM is fully loaded
let currentDocumentId = null;


document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");

    if (fileInput) {
        fileInput.addEventListener("change", uploadFiles);
    }
});


function scrollToBottom() {
    const chatBox = document.getElementById("chatBox");
    chatBox.scrollTop = chatBox.scrollHeight;
}


async function askQuestion() {
    const input = document.getElementById("questionInput");
    const chatBox = document.getElementById("chatBox");

    const question = input.value.trim();
    if (!question) return;

    chatBox.innerHTML += `<div class="user-msg">${question}</div>`;
    input.value = "";

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: question })
        });

        const data = await response.json();
        const answer = data.answer || "I don't know";
        const confidence = typeof data.confidence === "number" ? data.confidence.toFixed(2) : "0.00";
        const sources = (data.sources || []).map((s) => `<li>${s.document_id || "doc"} · ${s.section || "section"}</li>`).join("");

        chatBox.innerHTML += `
            <div class="bot-msg">
                <div class="bot-answer">${answer}</div>
                <div class="bot-meta">
                    <span class="confidence-badge">Confidence: ${confidence}</span>
                    <div class="sources">
                        <strong>Sources</strong>
                        <ul>${sources}</ul>
                    </div>
                </div>
            </div>
        `;
    } catch (error) {
        chatBox.innerHTML += `<div class="system-msg">Chat failed.</div>`;
        console.error(error);
    }

    scrollToBottom();
}


function uploadFiles(event) {
    const chatBox = document.getElementById("chatBox");
    const files = event.target.files;

    if (!files.length) return;

    const formData = new FormData();
    formData.append("file", files[0]);

    chatBox.innerHTML += `
        <div class="system-msg">
            Uploading file...<br>
            File: <strong>${files[0].name}</strong>
        </div>`;

    scrollToBottom();

    fetch("/api/upload", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            currentDocumentId = data.document.document_id;

            chatBox.innerHTML += `
                <div class="system-msg">
                    File uploaded successfully.<br>
                    Uploaded: <strong>${data.document.filename}</strong><br>
                    Document ID: <strong>${currentDocumentId || "N/A"}</strong><br>
                    Chunks indexed: <strong>${data.pipeline.chunks_indexed}</strong>
                </div>
            `;

            scrollToBottom();
        })
        .catch(error => {
            chatBox.innerHTML += `<div class="system-msg">Upload failed.</div>`;
            scrollToBottom();
            console.error(error);
        });
}