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
}