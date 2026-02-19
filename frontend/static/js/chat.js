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


// Ask question function
async function askQuestion() {
    const input = document.getElementById("questionInput");
    const chatBox = document.getElementById("chatBox");

    const question = input.value.trim();
    if (!question) return;

    chatBox.innerHTML += `<div class="user-msg">${question}</div>`;
    input.value = "";

    // Placeholder response (RAG comes next)
    chatBox.innerHTML += `
        <div class="bot-msg">
            <div class="bot-answer">
                This is a placeholder response that will include grounded citations.
            </div>
            <div class="bot-meta">
                <span class="confidence-badge">Confidence: 0.00</span>
                <div class="sources">
                    <strong>Sources</strong>
                    <ul>
                        <li>Document.pdf · Page 4</li>
                        <li>Guidelines.docx · Page 2</li>
                    </ul>
                </div>
            </div>
        </div>
    `;

    scrollToBottom();
}

// Upload files function
function uploadFiles(event) {
    const chatBox = document.getElementById("chatBox");
    const files = event.target.files;

    if (!files.length) return;

    const formData = new FormData();
    formData.append("file", files[0]); // match backend parameter name

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
                    Structured JSON: <strong>${data.pipeline.structured_output}</strong>
                </div>
            `;

            scrollToBottom();

            console.log("Current Document ID:", currentDocumentId);
        })
        .catch(error => {
            chatBox.innerHTML += `<div class="system-msg">Upload failed.</div>`;
            scrollToBottom();
            console.error(error);
        });
}