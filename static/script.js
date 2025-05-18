(() => {
  const current = { label: null };
  const trainBtn = document.getElementById('train-btn');
  const trainStatus = document.getElementById('train-status');
  const inferBtn = document.getElementById('infer-btn');
  const inferResult = document.getElementById('infer-result');
  const modelSelect = document.getElementById('model-select');
  const audioFileInput = document.getElementById('audio-file');

  function updateStatus(msg, color = 'black') {
    const status = document.getElementById('status');
    status.textContent = msg;
    status.style.color = color;
  }

  // Render label buttons
  const container = document.getElementById('label-buttons');
  labels.forEach(label => {
    const btn = document.createElement('button');
    btn.textContent = label;
    btn.addEventListener('click', () => {
      current.label = label;
      updateStatus(`選択中: ${label}`, 'blue');
    });
    container.appendChild(btn);
  });

  // Add label form
  document.getElementById('add-label-form').addEventListener('submit', async e => {
    e.preventDefault();
    const name = document.getElementById('new-label').value.trim();
    if (name) {
      await fetch('/add_label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: name })
      });
      location.reload();
    }
  });

  // Recording
  let mediaRecorder, audioChunks;
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.onloadend = async () => {
          await fetch('/save_audio', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label: current.label, audio: reader.result })
          });
          updateStatus('保存しました ✅', 'green');
        };
        reader.readAsDataURL(blob);
      };
    })
    .catch(err => {
      console.error('MediaDevices Error:', err);
      updateStatus('マイクへのアクセスを許可してください', 'red');
    });

  document.getElementById('record-btn').addEventListener('click', () => {
    if (!current.label) {
      return updateStatus('⚠ ラベルを選択してください', 'red');
    }
    audioChunks = [];
    mediaRecorder.start();
    updateStatus('録音中...', 'red');
    document.getElementById('record-btn').disabled = true;
    document.getElementById('stop-btn').disabled = false;
  });
  document.getElementById('stop-btn').addEventListener('click', () => {
    mediaRecorder.stop();
    document.getElementById('record-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
  });

  // Train
  trainBtn.addEventListener('click', async () => {
    trainStatus.textContent = '';
    const res = await fetch('/train', { method: 'POST' });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      trainStatus.textContent += decoder.decode(value);
    }
  });

  // Infer
  inferBtn.addEventListener('click', async () => {
    inferResult.textContent = '';
    const model = modelSelect.value;
    const file = audioFileInput.files[0];
    if (!model) {
      inferResult.textContent = '⚠ モデルを選択してください';
      return;
    }
    if (!file) {
      inferResult.textContent = '⚠ 音声ファイルを選択してください';
      return;
    }
    const form = new FormData();
    form.append('model', model);
    form.append('audio', file);
    const res = await fetch('/infer', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json();
      inferResult.textContent = `Error: ${err.error || res.statusText}`;
      return;
    }
    const data = await res.json();
    inferResult.textContent = `${data.label} (${(data.confidence * 100).toFixed(2)}%)`;
  });
})();

