let streamActive = false;
let metricsSocket = null;

// UI Elements
const modelDropdown = document.getElementById('model');
const backendDropdown = document.getElementById('backend');
const deviceDropdown = document.getElementById('device');

const startStreamButton = document.getElementById('start-stream');
const stopStreamButton = document.getElementById('stop-stream');
const resetChartButton = document.getElementById('reset-chart');

const downloadPNGButton = document.getElementById('download-png');
const downloadCSVButton = document.getElementById('download-csv');

// Webcam UI
const useWebcamCheckbox = document.getElementById('use-webcam');
const cameraIdInput = document.getElementById('camera-id');
const videoStreamImg = document.getElementById('video-stream');
const streamPlaceholder = document.getElementById('stream-placeholder');

// Storage for CSV Export (Holds full history)
let fullSessionMetrics = [];

// === 1. State Management (API Calls) ===

async function setSourceFromUI() {
  const use_camera = !!(useWebcamCheckbox && useWebcamCheckbox.checked);
  const camera_id = parseInt((cameraIdInput && cameraIdInput.value) || '0', 10);
  try {
    const res = await fetch('/set-source', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ use_camera, camera_id })
    });
    const result = await res.json().catch(() => ({}));
    if (result?.status && result?.message) {
        console.log(result.message);
    }
  } catch (err) {
    console.error('set-source failed', err);
  }
}

function handleModelChange() {
  const name = modelDropdown.value;
  console.log('Selected model:', name);

  return fetch('/set-model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: name })
  }).catch(error => console.error('Error:', error));
}

function handleParamsChange() {
  const backend = backendDropdown.value;
  const device = deviceDropdown.value;
  console.log(`Selected params - Backend: ${backend}, Device: ${device}`);

  return fetch('/set-params', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ backend: backend, device: device })
  }).catch(error => console.error('Error:', error));
}

// === 2. Event Listeners ===

modelDropdown.addEventListener('change', handleModelChange);
backendDropdown.addEventListener('change', handleParamsChange);
deviceDropdown.addEventListener('change', handleParamsChange);

window.addEventListener('DOMContentLoaded', async () => {
  await handleModelChange();
  await handleParamsChange();
  await setSourceFromUI();
});

// === 3. Stream Control ===

startStreamButton.addEventListener('click', async function () {
  await setSourceFromUI();

  const ts = new Date().getTime();
  videoStreamImg.src = `/video-feed?ts=${ts}`;
  videoStreamImg.classList.remove('d-none');
  if(streamPlaceholder) streamPlaceholder.classList.add('d-none');

  // Lock UI
  startStreamButton.classList.add('d-none');
  stopStreamButton.classList.remove('d-none');
  
  modelDropdown.disabled = true;
  backendDropdown.disabled = true;
  deviceDropdown.disabled = true;
  
  resetChartButton.disabled = true;
  if (useWebcamCheckbox) useWebcamCheckbox.disabled = true;
  if (cameraIdInput) cameraIdInput.disabled = true;
  
  streamActive = true;
  downloadPNGButton.disabled = true;
  downloadCSVButton.disabled = true;

  const modelLabel = modelDropdown.options[modelDropdown.selectedIndex].text;
  const configLabel = `${modelLabel} - ${backendDropdown.value} (${deviceDropdown.value})`;
  
  chartData.datasets = chartData.datasets.filter(ds => ds.label !== configLabel);

  currentDataset = {
    label: configLabel,
    data: [],
    borderColor: `hsl(${Math.floor(Math.random() * 360)}, 70%, 50%)`,
    fill: false,
    tension: 0.1
  };
  
  // Reset Frame Counter for the CHART (visual x-axis reset)
  // But we DO NOT clear 'fullSessionMetrics' here, so CSV keeps growing.
  frameCounter = 0; 

  chartData.datasets.push(currentDataset);
  chart.update();

  connectMetricsWebSocket();
});

stopStreamButton.addEventListener('click', function () {
  videoStreamImg.src = '';
  videoStreamImg.classList.add('d-none');
  if(streamPlaceholder) streamPlaceholder.classList.remove('d-none');

  // Unlock UI
  startStreamButton.classList.remove('d-none');
  stopStreamButton.classList.add('d-none');
  
  modelDropdown.disabled = false;
  backendDropdown.disabled = false;
  deviceDropdown.disabled = false;
  
  resetChartButton.disabled = false;
  if (useWebcamCheckbox) useWebcamCheckbox.disabled = false;
  if (cameraIdInput) cameraIdInput.disabled = false;
  
  streamActive = false;
  downloadPNGButton.disabled = false;
  downloadCSVButton.disabled = false;

  // Stop server side
  fetch('/stop-stream', { method: 'POST' }).catch(console.error);

  if (metricsSocket) {
    metricsSocket.close();
    metricsSocket = null;
  }
});

resetChartButton.addEventListener('click', function () {
  chart.data.labels = [];
  chart.data.datasets = [];
  chart.update();
  
  // Clear CSV history only on explicit Reset
  fullSessionMetrics = []; 
  
  console.log('Reset chart and metrics history');
});

// === 4. Chart & Websocket ===

const chartData = { labels: [], datasets: [] };
const chart = new Chart(document.getElementById('fpsChart'), {
  type: 'line',
  data: chartData,
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: { display: false }, 
      y: { beginAtZero: true, title: {display: true, text: 'FPS'} }
    },
    plugins: {
        legend: { position: 'bottom' }
    }
  }
});

let currentDataset = null;
let frameCounter = 0;
const SAMPLE_EVERY = 2; 

function connectMetricsWebSocket() {
  if (metricsSocket) return;

  const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
  metricsSocket = new WebSocket(`${protocol}${window.location.host}/ws/metrics`);

  metricsSocket.onopen = () => console.log("WS Connected");

  metricsSocket.onmessage = (event) => {
    if (!currentDataset || !streamActive) return;

    const response = JSON.parse(event.data);
    if (response.status === 'error') {
      console.error(response.message);
      return;
    }

    const m = response.metrics;
    frameCounter++;

    // 1. Store Full Data for CSV
    // We include the current Label so we know which run this data belongs to
    fullSessionMetrics.push({
        run_label: currentDataset.label,
        frame: frameCounter,
        fps: m.fps,
        infer_ms: m.inference_time,
        proc_ms: m.processing_time
    });

    // 2. Update Visual Chart (Throttled)
    if (frameCounter % SAMPLE_EVERY === 0) {
        chartData.labels.push(frameCounter);
        currentDataset.data.push(m.fps);
        
        if(chartData.labels.length > 100) {
            chartData.labels.shift();
            chartData.datasets.forEach(ds => ds.data.shift());
        }
        chart.update();
    }
  };
}

// === 5. Downloads ===

downloadPNGButton.addEventListener('click', function () {
  const link = document.createElement('a');
  link.download = 'benchmark_chart.png';
  link.href = chart.toBase64Image();
  link.click();
});

downloadCSVButton.addEventListener('click', function () {
    // UPDATED CSV: Includes "Run Name" column
    let csv = 'run_name,frame,fps,inference_ms,processing_ms\n';
    
    if (fullSessionMetrics.length > 0) {
        fullSessionMetrics.forEach(row => {
            const fps = row.fps ? row.fps.toFixed(2) : '0.00';
            const inf = row.infer_ms ? row.infer_ms.toFixed(3) : '0.000';
            const proc = row.proc_ms ? row.proc_ms.toFixed(3) : '0.000';
            // Wrap label in quotes just in case
            const label = `"${row.run_label}"`;
            
            csv += `${label},${row.frame},${fps},${inf},${proc}\n`;
        });
    } else {
        alert("No metrics data recorded yet. Did the stream run?");
    }

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `benchmark_data_${new Date().getTime()}.csv`;
    a.click();
});

// === 6. Video Upload / Reset ===
// (Same as before...)
const uploadVideoForm = document.getElementById('upload-video-form');
const videoFileInput = document.getElementById('upload-video-file');
const uploadVideoButton = document.getElementById('upload-video-button');
const resetVideoButton = document.getElementById('reset-video');

if(uploadVideoForm) {
    uploadVideoForm.addEventListener('submit', async function (e) {
      e.preventDefault();
      alert("Video upload endpoint not fully connected in this demo version.");
    });
}

if(resetVideoButton) {
    resetVideoButton.addEventListener('click', async function () {
      try {
        videoFileInput.value = '';
        const response = await fetch('/delete-video', { method: 'POST' });
        const result = await response.json();
        
        if (useWebcamCheckbox) {
          useWebcamCheckbox.checked = false;
          await setSourceFromUI();
        }
      } catch (err) {
        console.error(err);
      }
    });
}

// === 7. Webcam Toggles ===
if (useWebcamCheckbox) {
  useWebcamCheckbox.addEventListener('change', async () => {
    await setSourceFromUI();
    if (streamActive) {
        stopStreamButton.click();
        setTimeout(() => startStreamButton.click(), 500);
    }
  });
}
if (cameraIdInput) {
  cameraIdInput.addEventListener('change', async () => {
    if (useWebcamCheckbox.checked) {
      await setSourceFromUI();
      if (streamActive) {
        stopStreamButton.click();
        setTimeout(() => startStreamButton.click(), 500);
      }
    }
  });
}