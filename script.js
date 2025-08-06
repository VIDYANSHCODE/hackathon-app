// --- Page Navigation ---
const btnDetector = document.getElementById('btn-detector');
const btnAnalysis = document.getElementById('btn-analysis');
const btnProjectInfo = document.getElementById('btn-project-info');

const detectorPage = document.getElementById('detector-page');
const analysisPage = document.getElementById('analysis-page');
const projectInfoPage = document.getElementById('project-info-page');

const tabBtns = document.querySelectorAll('.tab-btn');
const pages = [detectorPage, analysisPage, projectInfoPage];

function showPage(pageToShow) {
    pages.forEach(page => page.classList.add('hidden'));
    pageToShow.classList.remove('hidden');
}

tabBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
        // Deactivate all buttons
        tabBtns.forEach(b => {
            b.classList.remove('tab-active');
            b.classList.add('text-gray-400');
        });
        // Activate clicked button
        e.currentTarget.classList.add('tab-active');
        e.currentTarget.classList.remove('text-gray-400');
    });
});

btnDetector.addEventListener('click', () => showPage(detectorPage));
btnAnalysis.addEventListener('click', () => showPage(analysisPage));
btnProjectInfo.addEventListener('click', () => showPage(projectInfoPage));


// --- ENHANCED: 3D Background & UI ---
const clock = new THREE.Clock();
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('bg-canvas'), alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);

const createParticles = (count, size, color, speed) => {
    const particlesGeometry = new THREE.BufferGeometry();
    const posArray = new Float32Array(count * 3);
    for (let i = 0; i < count * 3; i++) {
        posArray[i] = (Math.random() - 0.5) * (10 + Math.random() * 10);
    }
    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    const particlesMaterial = new THREE.PointsMaterial({
        size: size,
        color: color,
        blending: THREE.AdditiveBlending,
        transparent: true,
        opacity: 0.8
    });
    const particles = new THREE.Points(particlesGeometry, particlesMaterial);
    particles.userData.speed = speed;
    scene.add(particles);
    return particles;
};

const particles1 = createParticles(8000, 0.01, 0x3b82f6, 0.02);
const particles2 = createParticles(2000, 0.04, 0x1d4ed8, 0.01);

camera.position.z = 5;

const mainPanel = document.getElementById('main-panel');
document.addEventListener('mousemove', (event) => {
    const mouseX = (event.clientX / window.innerWidth) * 2 - 1;
    const mouseY = -(event.clientY / window.innerHeight) * 2 + 1;

    const targetRotationX = mouseY * 0.2;
    const targetRotationY = mouseX * 0.2;
    particles1.rotation.x += 0.05 * (targetRotationX - particles1.rotation.x);
    particles1.rotation.y += 0.05 * (targetRotationY - particles1.rotation.y);
    particles2.rotation.x += 0.02 * (targetRotationX - particles2.rotation.x);
    particles2.rotation.y += 0.02 * (targetRotationY - particles2.rotation.y);

    const rotX = mouseY * -7;
    const rotY = mouseX * 7;
    mainPanel.style.transform = `rotateX(${rotX}deg) rotateY(${rotY}deg)`;
});

function animate() {
    requestAnimationFrame(animate);
    const elapsedTime = clock.getElapsedTime();

    particles1.rotation.y += elapsedTime * 0.00001 * particles1.userData.speed;
    particles2.rotation.y += elapsedTime * 0.00001 * particles2.userData.speed;

    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- ONNX Model and Detection Logic ---
const MODEL_PATH = './bestonsite.onnx';
const CLASSES = ["FireExtinguisher", "ToolBox", "OxygenTank"];
const CONFIDENCE_THRESHOLD = 0.5;
const IOU_THRESHOLD = 0.5;

const fileInput = document.getElementById('image-input');
const dropZone = document.getElementById('drop-zone');
const outputContainer = document.getElementById('output-container');
const outputCanvas = document.getElementById('output-canvas');
const loader = document.getElementById('loader');
const modelStatusDiv = document.getElementById('model-status');
const ctx = outputCanvas.getContext('2d');
let session;

async function loadModel() {
    try {
        modelStatusDiv.textContent = 'Loading AI Model... Please wait.';
        dropZone.classList.add('opacity-50', 'cursor-not-allowed');
        session = await ort.InferenceSession.create(MODEL_PATH);
        console.log('Model loaded successfully.');
        modelStatusDiv.textContent = 'Model Ready. Drop an image or click above.';
        modelStatusDiv.className = 'mt-4 text-center text-green-400';
        dropZone.classList.remove('opacity-50', 'cursor-not-allowed');
        fileInput.disabled = false;
    } catch (e) {
        console.error('Failed to load the model:', e);
        modelStatusDiv.textContent = `Error: Failed to load AI model. Check console for details.`;
        modelStatusDiv.className = 'mt-4 text-center text-red-400 font-semibold';
    }
}

loadModel();

fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); });
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (file && session) runObjectDetection(file);
}

async function runObjectDetection(file) {
    outputContainer.classList.add('hidden');
    loader.classList.remove('hidden');
    const image = new Image();
    image.src = URL.createObjectURL(file);
    image.onload = async () => {
        try {
            const [input, xRatio, yRatio] = await preprocess(image);
            const feeds = { images: input };
            const results = await session.run(feeds);
            const output = results.output0.data;
            const boxes = postprocess(output, xRatio, yRatio);
            draw(image, boxes);
            outputContainer.classList.remove('hidden');
        } catch (e) {
            console.error('Error during object detection:', e);
            alert(`An error occurred while analyzing the image: ${e.message}`);
        } finally {
            loader.classList.add('hidden');
        }
    };
}

async function preprocess(img) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const modelWidth = 1088;
    const modelHeight = 1088;
    canvas.width = modelWidth;
    canvas.height = modelHeight;
    const xRatio = modelWidth / img.width;
    const yRatio = modelHeight / img.height;
    context.drawImage(img, 0, 0, modelWidth, modelHeight);
    const imageData = context.getImageData(0, 0, modelWidth, modelHeight);
    const { data } = imageData;
    const red = [], green = [], blue = [];
    for (let i = 0; i < data.length; i += 4) {
        red.push(data[i] / 255.0);
        green.push(data[i + 1] / 255.0);
        blue.push(data[i + 2] / 255.0);
    }
    const transposedData = [...red, ...green, ...blue];
    const float32Data = new Float32Array(transposedData);
    return [new ort.Tensor('float32', float32Data, [1, 3, modelHeight, modelWidth]), xRatio, yRatio];
}

function postprocess(output, xRatio, yRatio) {
    const transposed = [];
    const numOutputChannels = CLASSES.length + 4;
    const numDetections = output.length / numOutputChannels;
    for (let i = 0; i < numDetections; i++) {
        const row = [];
        for (let j = 0; j < numOutputChannels; j++) {
            row.push(output[i + j * numDetections]);
        }
        transposed.push(row);
    }
    const boxes = [];
    for (const row of transposed) {
        const [x, y, w, h, ...classScores] = row;
        let maxScore = 0, classId = -1;
        for (let i = 0; i < classScores.length; i++) {
            if (classScores[i] > maxScore) {
                maxScore = classScores[i];
                classId = i;
            }
        }
        if (maxScore > CONFIDENCE_THRESHOLD) {
            boxes.push({
                box: [(x - w / 2) / xRatio, (y - h / 2) / yRatio, w / xRatio, h / yRatio],
                score: maxScore,
                classId: classId
            });
        }
    }
    return nms(boxes, IOU_THRESHOLD);
}

function nms(boxes, iouThreshold) {
    boxes.sort((a, b) => b.score - a.score);
    const result = [];
    while (boxes.length > 0) {
        result.push(boxes[0]);
        boxes = boxes.filter(box => iou(boxes[0], box) < iouThreshold);
    }
    return result;
}

function iou(boxA, boxB) {
    const [ax1, ay1, aw, ah] = boxA.box;
    const [bx1, by1, bw, bh] = boxB.box;
    const ax2 = ax1 + aw, ay2 = ay1 + ah, bx2 = bx1 + bw, by2 = by1 + bh;
    const x_left = Math.max(ax1, bx1), y_top = Math.max(ay1, by1);
    const x_right = Math.min(ax2, bx2), y_bottom = Math.min(ay2, by2);
    if (x_right < x_left || y_bottom < y_top) return 0.0;
    const intersectionArea = (x_right - x_left) * (y_bottom - y_top);
    const boxAArea = aw * ah, boxBArea = bw * bh;
    return intersectionArea / (boxAArea + boxBArea - intersectionArea);
}

function draw(img, boxes) {
    outputCanvas.width = img.width;
    outputCanvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    boxes.forEach(({ box, score, classId }) => {
        const [x, y, w, h] = box;
        const label = `${CLASSES[classId]} (${(score * 100).toFixed(1)}%)`;
        ctx.strokeStyle = '#0ea5e9';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = '#0ea5e9';
        ctx.font = '16px Inter';
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x - 1, y - 22, textWidth + 10, 22);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x + 4, y - 6);
    });
}
