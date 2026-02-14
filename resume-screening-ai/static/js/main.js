document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');
    const resultsSection = document.getElementById('results-section');
    const uploadSection = document.querySelector('.upload-section');
    const fileNameDisplay = document.getElementById('file-name-display');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const loader = analyzeBtn.querySelector('.loader');

    let selectedFile = null;

    // Trigger file input
    dropZone.addEventListener('click', () => fileInput.click());

    // File selection
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            selectedFile = files[0];
            fileNameDisplay.textContent = `Selected: ${selectedFile.name}`;
            analyzeBtn.disabled = false;
        }
    }

    // Analysis
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('resume', selectedFile);

        // UI state
        analyzeBtn.disabled = true;
        btnText.textContent = "Analyzing...";
        loader.classList.remove('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                alert(data.error);
                resetUI();
            } else {
                displayResults(data);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Something went wrong during analysis.');
            resetUI();
        }
    });

    function displayResults(data) {
        uploadSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        document.getElementById('predicted-role').textContent = data.role;
        document.getElementById('match-value').textContent = `${data.match_percentage}%`;

        // AI Feedback display
        const feedbackEl = document.getElementById('ai-feedback');
        feedbackEl.innerHTML = formatFeedback(data.ai_feedback);
    }

    function formatFeedback(text) {
        // Convert markdown-style bullet points and bolding to HTML
        return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>')
            .replace(/\* (.*?)(?=<br>|$)/g, '<li>$1</li>')
            .replace(/(\d\.) (.*?)(?=<br>|$)/g, '<strong>$1 $2</strong>');
    }

    function resetUI() {
        analyzeBtn.disabled = false;
        btnText.textContent = "Analyze Resume";
        loader.classList.add('hidden');
    }

    resetBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        selectedFile = null;
        fileInput.value = '';
        fileNameDisplay.textContent = '';
        resetUI();
    });
});
