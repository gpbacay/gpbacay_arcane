document.addEventListener('DOMContentLoaded', function() {
    // Canvas and UI Elements
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clear-btn');
    const predictionSpan = document.getElementById('prediction');
    const confidenceSpan = document.getElementById('confidence');
    const loader = document.getElementById('loader');

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // Canvas Setup
    function initializeCanvas() {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
    }

    // Event Listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    clearBtn.addEventListener('click', clearCanvas);

    // Drawing Handlers
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }

    function draw(e) {
        if (!isDrawing) return;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
        predictDebounced();
    }

    function stopDrawing() {
        if (isDrawing) {
            predictDebounced();
        }
        isDrawing = false;
    }

    function handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;

        if (e.type === 'touchstart') {
            isDrawing = true;
            [lastX, lastY] = [x, y];
        } else if (e.type === 'touchmove' && isDrawing) {
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();
            [lastX, lastY] = [x, y];
            predictDebounced();
        }
    }

    function clearCanvas() {
        initializeCanvas();
        predictionSpan.textContent = '-';
        confidenceSpan.textContent = '-';
        resetCuboctahedron();
    }

    // Debounce function
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    const predictDebounced = debounce(predict, 300);

    // Prediction Logic
    function predict() {
        loader.style.display = 'block';

        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        
        tempCtx.fillStyle = 'black';
        tempCtx.fillRect(0, 0, 28, 28);
        tempCtx.drawImage(canvas, 0, 0, 280, 280, 0, 0, 28, 28);
    
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const data = new Float32Array(28 * 28);
        for (let i = 0; i < imageData.data.length; i += 4) {
            data[i / 4] = imageData.data[i] / 255.0; // Use the red channel for grayscale
        }

        const apiUrl = '/predict/';

        fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: Array.from(data)
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Network response was not ok (status: ${response.status})`);
            }
            return response.json();
        })
        .then(result => {
            if (result.error) {
                throw new Error(result.error);
            }
            predictionSpan.textContent = result.digit;
            confidenceSpan.textContent = `${(result.confidence * 100).toFixed(2)}%`;
            updateCuboctahedron(result.confidence);
        })
        .catch(error => {
            console.error('Error:', error);
            predictionSpan.textContent = 'Error';
            confidenceSpan.textContent = 'N/A';
        })
        .finally(() => {
            loader.style.display = 'none';
        });
    }

    // Three.js Scene for Cuboctahedron
    const threeContainer = document.getElementById('three-container');
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, threeContainer.clientWidth / threeContainer.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(threeContainer.clientWidth, threeContainer.clientHeight);
    threeContainer.appendChild(renderer.domElement);

    // Function to get color from CSS variable
    function getCssVar(varName) {
        return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    }

    const geometry = new THREE.IcosahedronGeometry(1.5, 1);
    const material = new THREE.MeshStandardMaterial({ 
        color: new THREE.Color(getCssVar('--primary')), 
        wireframe: true,
        metalness: 0.5,
        roughness: 0.8
    });
    const shape = new THREE.Mesh(geometry, material);
    scene.add(shape);
    camera.position.z = 4;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const pointLight = new THREE.PointLight(new THREE.Color(getCssVar('--primary')), 0.8); 
    pointLight.position.set(5, 5, 5);
    scene.add(pointLight);

    function animate() {
        requestAnimationFrame(animate);
        shape.rotation.x += 0.005;
        shape.rotation.y += 0.005;
        renderer.render(scene, camera);
    }

    function resetCuboctahedron() {
        shape.scale.set(1, 1, 1);
        shape.rotation.set(0, 0, 0);
    }

    function updateCuboctahedron(confidence) {
        const scale = 1 + confidence * 0.7;
        shape.scale.set(scale, scale, scale);
    }
    
    // Handle window resize
    window.addEventListener('resize', () => {
        camera.aspect = threeContainer.clientWidth / threeContainer.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(threeContainer.clientWidth, threeContainer.clientHeight);
    });

    // Initial setup
    initializeCanvas();
    animate();
}); 