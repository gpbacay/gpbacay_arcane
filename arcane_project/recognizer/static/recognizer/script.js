// A.R.C.A.N.E. Neuromimetic Language Model Interface

document.addEventListener('DOMContentLoaded', function() {
    // UI Elements
    const seedTextArea = document.getElementById('seed-text');
    const temperatureSlider = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperature-value');
    const maxLengthSlider = document.getElementById('max-length');
    const lengthValue = document.getElementById('length-value');
    const generateBtn = document.getElementById('generate-btn');
    const generatedText = document.getElementById('generated-text');
    const generationInfo = document.getElementById('generation-info');
    const loader = document.getElementById('loader');
    const aboutBtn = document.getElementById('about-btn');
    const aboutModal = document.getElementById('about-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');
    const modelStatus = document.getElementById('model-status');

    // Update slider displays
    temperatureSlider.addEventListener('input', function() {
        temperatureValue.textContent = this.value;
        updateTemperatureDescription();
    });

    maxLengthSlider.addEventListener('input', function() {
        lengthValue.textContent = this.value;
    });

    function updateTemperatureDescription() {
        const temp = parseFloat(temperatureSlider.value);
        let description;
        if (temp < 0.6) {
            description = 'Conservative';
        } else if (temp < 1.1) {
            description = 'Balanced';
        } else {
            description = 'Creative';
        }
        temperatureValue.textContent = `${temp} (${description})`;
    }

    // Initialize temperature description
    updateTemperatureDescription();

    // Generate Text Function
    generateBtn.addEventListener('click', async function() {
        const seedText = seedTextArea.value.trim();
        
        if (!seedText) {
            alert('Please enter some seed text to start generation.');
            return;
        }

        if (seedText.length > 200) {
            alert('Seed text is too long. Please limit to 200 characters.');
            return;
        }

        // Disable button and show loader
        generateBtn.disabled = true;
        loader.classList.remove('hidden');
        generatedText.textContent = '';
        generationInfo.classList.add('hidden');

        try {
            const response = await fetch('/generate/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    seed_text: seedText,
                    temperature: parseFloat(temperatureSlider.value),
                    max_length: parseInt(maxLengthSlider.value)
                })
            });

            const data = await response.json();

            if (response.ok) {
                // Display generated text
                generatedText.textContent = data.generated_text;
                
                // Show generation info
                generationInfo.innerHTML = `
                    <strong>Seed:</strong> "${data.seed_text}" | 
                    <strong>Temperature:</strong> ${data.temperature} | 
                    <strong>Max Length:</strong> ${data.max_length} words
                `;
                generationInfo.classList.remove('hidden');
            } else {
                generatedText.textContent = `Error: ${data.error}`;
                generatedText.className = 'text-red-500';
            }

        } catch (error) {
            console.error('Generation error:', error);
            generatedText.textContent = `Network error: ${error.message}`;
            generatedText.className = 'text-red-500';
        }

        // Re-enable button and hide loader
        generateBtn.disabled = false;
        loader.classList.add('hidden');
    });

    // Modal functionality
    aboutBtn.addEventListener('click', function() {
        aboutModal.classList.remove('hidden');
        aboutModal.classList.add('flex');
    });

    closeModalBtn.addEventListener('click', function() {
        aboutModal.classList.add('hidden');
        aboutModal.classList.remove('flex');
    });

    aboutModal.addEventListener('click', function(e) {
        if (e.target === aboutModal) {
            aboutModal.classList.add('hidden');
            aboutModal.classList.remove('flex');
        }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to generate
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
            if (!generateBtn.disabled) {
                generateBtn.click();
            }
        }
        
        // Escape to close modal
        if (e.key === 'Escape') {
            aboutModal.classList.add('hidden');
            aboutModal.classList.remove('flex');
        }
    });

    // Auto-resize textarea
    seedTextArea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Check model status periodically
    async function checkModelStatus() {
        try {
            const response = await fetch('/health/');
            const data = await response.json();
            
            const statusIndicator = modelStatus.querySelector('.w-3.h-3.rounded-full');
            const statusText = modelStatus.querySelector('span');
            
            if (data.model_loaded && data.tokenizer_loaded) {
                statusIndicator.className = statusIndicator.className.replace('bg-red-500', 'bg-green-500');
                modelStatus.className = modelStatus.className.replace('border-red-500 bg-red-500/10', 'border-green-500 bg-green-500/10');
                statusText.textContent = 'Model Ready';
                generateBtn.disabled = false;
            } else {
                statusIndicator.className = statusIndicator.className.replace('bg-green-500', 'bg-red-500');
                modelStatus.className = modelStatus.className.replace('border-green-500 bg-green-500/10', 'border-red-500 bg-red-500/10');
                statusText.textContent = 'Model Loading...';
                generateBtn.disabled = true;
            }
        } catch (error) {
            console.error('Health check failed:', error);
        }
    }

    // Check status every 10 seconds
    setInterval(checkModelStatus, 10000);

    // Sample text suggestions
    const sampleTexts = [
        "to be or not to be",
        "once upon a time",
        "in a distant galaxy",
        "the meaning of life",
        "artificial intelligence will",
        "consciousness is defined as"
    ];

    // Add sample text buttons
    function addSampleButtons() {
        const container = document.createElement('div');
        container.className = 'mb-4';
        container.innerHTML = '<div class="text-sm text-muted-foreground mb-2">Quick starts:</div>';
        
        const buttonsContainer = document.createElement('div');
        buttonsContainer.className = 'flex flex-wrap gap-2';
        
        sampleTexts.forEach(text => {
            const button = document.createElement('button');
            button.className = 'px-2 py-1 text-xs bg-muted hover:bg-muted/80 text-muted-foreground rounded border border-border transition-colors';
            button.textContent = `"${text}"`;
            button.addEventListener('click', () => {
                seedTextArea.value = text;
                seedTextArea.focus();
            });
            buttonsContainer.appendChild(button);
        });
        
        container.appendChild(buttonsContainer);
        seedTextArea.parentNode.insertBefore(container, seedTextArea);
    }

    // Add sample buttons
    addSampleButtons();

    // Get model info on load
    async function loadModelInfo() {
        try {
            const response = await fetch('/model-info/');
            const data = await response.json();
            
            if (response.ok && data.parameters) {
                console.log('Neuromimetic Model Info:', data);
                
                // Could display model parameters in the architecture section
                const archContainer = document.getElementById('architecture-container');
                if (archContainer) {
                    const infoDiv = document.createElement('div');
                    infoDiv.className = 'mt-4 text-xs text-muted-foreground text-center';
                    infoDiv.innerHTML = `
                        <div>Parameters: ${data.parameters.total_parameters.toLocaleString()}</div>
                        <div>Layers: ${data.parameters.layers}</div>
                    `;
                    archContainer.appendChild(infoDiv);
                }
            }
        } catch (error) {
            console.error('Failed to load model info:', error);
        }
    }

    // Load model info
    loadModelInfo();
}); 