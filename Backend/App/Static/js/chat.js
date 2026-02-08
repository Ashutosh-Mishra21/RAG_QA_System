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
            This is a placeholder response.<br>
            <small>Confidence: 0.00</small>
        </div>
    `;
}
