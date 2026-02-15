// Wait until DOM is fully loaded
document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");

    if (fileInput) {
        fileInput.addEventListener("change", uploadFiles);
    }
});

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

    chatBox.scrollTop = chatBox.scrollHeight;
}

// Upload files function
function uploadFiles(event) {
    const chatBox = document.getElementById("chatBox");
    const files = event.target.files;

    if (!files.length) return;

    const formData = new FormData();
    formData.append("file", files[0]); // match backend parameter name

    chatBox.innerHTML += `<div class="system-msg">Uploading file...</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;

    fetch("/api/upload", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            chatBox.innerHTML += `<div class="system-msg">File uploaded successfully.</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
            console.log(data);
        })
        .catch(error => {
            chatBox.innerHTML += `<div class="system-msg">Upload failed.</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
            console.error(error);
        });
}