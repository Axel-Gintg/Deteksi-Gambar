/**
 * ForensiX — Frontend Logic
 * Handles file upload, API communication, and result rendering.
 */

const API_BASE = window.location.origin;

// ─── DOM Elements ───────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const uploadZone = $("#uploadZone");
const fileInput = $("#fileInput");
const uploadPreview = $("#uploadPreview");
const previewImage = $("#previewImage");
const previewName = $("#previewName");
const previewSize = $("#previewSize");
const previewRemove = $("#previewRemove");
const thresholdSlider = $("#thresholdSlider");
const thresholdValue = $("#thresholdValue");
const detectBtn = $("#detectBtn");
const resultsContainer = $("#resultsContainer");
const navStatus = $("#navStatus");
const btnNewAnalysis = $("#btnNewAnalysis");

let selectedFile = null;

// ─── Health Check ───────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      navStatus.className = "nav-status online";
      navStatus.querySelector(".status-text").textContent = "Model Ready";
    } else throw new Error();
  } catch {
    navStatus.className = "nav-status offline";
    navStatus.querySelector(".status-text").textContent = "Offline";
  }
}
checkHealth();
setInterval(checkHealth, 30000);

// ─── Navbar scroll effect ───────────────────────────────────
window.addEventListener("scroll", () => {
  $("#navbar").classList.toggle("scrolled", window.scrollY > 30);
});

// ─── Smooth nav links ───────────────────────────────────────
document.querySelectorAll(".nav-link").forEach((link) => {
  link.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach((l) => l.classList.remove("active"));
    link.classList.add("active");
  });
});

// ─── File Upload ────────────────────────────────────────────
uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  const files = e.dataTransfer.files;
  if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  const validTypes = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
  if (!validTypes.includes(file.type)) {
    alert("Format file tidak didukung. Gunakan PNG, JPG, atau WEBP.");
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    alert("File terlalu besar. Maksimal 20MB.");
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewName.textContent = file.name;
    previewSize.textContent = formatSize(file.size);
    uploadZone.style.display = "none";
    uploadPreview.style.display = "block";
    detectBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

previewRemove.addEventListener("click", resetUpload);

function resetUpload() {
  selectedFile = null;
  fileInput.value = "";
  uploadZone.style.display = "flex";
  uploadPreview.style.display = "none";
  detectBtn.disabled = true;
  resultsContainer.style.display = "none";
}

// ─── Threshold Slider ───────────────────────────────────────
thresholdSlider.addEventListener("input", () => {
  thresholdValue.textContent = thresholdSlider.value + "%";
});

// ─── Detection ──────────────────────────────────────────────
detectBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  const btnText = detectBtn.querySelector(".btn-text");
  const btnLoader = detectBtn.querySelector(".btn-loader");
  btnText.style.display = "none";
  btnLoader.style.display = "flex";
  detectBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("threshold", (parseInt(thresholdSlider.value) / 100).toString());

  try {
    const res = await fetch(`${API_BASE}/api/detect`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Terjadi kesalahan pada server");
    }

    const data = await res.json();
    displayResults(data);
  } catch (err) {
    alert("Gagal menganalisis gambar: " + err.message);
  } finally {
    btnText.style.display = "flex";
    btnLoader.style.display = "none";
    detectBtn.disabled = false;
  }
});

// ─── Display Results ────────────────────────────────────────
function displayResults(data) {
  const { result, confidence, filename, images, regions } = data;
  const isTampered = result === "TAMPERED";

  // Verdict
  const verdictIcon = $("#verdictIcon");
  verdictIcon.className = `verdict-icon ${isTampered ? "tampered-icon" : "authentic-icon"}`;
  verdictIcon.innerHTML = isTampered
    ? '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>'
    : '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg>';
  verdictIcon.style.color = isTampered ? "var(--red)" : "var(--green)";

  const verdictLabel = $("#verdictLabel");
  verdictLabel.textContent = isTampered ? "🔴 TAMPERED — Gambar Dimanipulasi" : "🟢 AUTHENTIC — Gambar Asli";
  verdictLabel.className = `verdict-label ${isTampered ? "tampered-text" : "authentic-text"}`;

  $("#verdictFile").textContent = filename;

  // Scores
  $("#scoreTampered").textContent = confidence.tampered.toFixed(1) + "%";
  $("#scoreAuthentic").textContent = confidence.authentic.toFixed(1) + "%";

  // Animate bars
  setTimeout(() => {
    $("#barTampered").style.width = confidence.tampered + "%";
    $("#barAuthentic").style.width = confidence.authentic + "%";
  }, 100);

  // ─── Annotated Images (Bounding Box + Mask Overlay) ───────
  const annotatedSection = $("#annotatedSection");
  if (images.annotated && images.mask_overlay) {
    $("#imgAnnotated").src = `data:image/png;base64,${images.annotated}`;
    $("#imgMaskOverlay").src = `data:image/png;base64,${images.mask_overlay}`;
    annotatedSection.style.display = "block";
  } else {
    annotatedSection.style.display = "none";
  }

  // ─── Regions Detail Panel ─────────────────────────────────
  const regionsList = $("#regionsList");
  const regionsEmpty = $("#regionsEmpty");
  regionsList.innerHTML = "";

  if (regions && regions.length > 0) {
    regionsEmpty.style.display = "none";
    regions.forEach((region, idx) => {
      const severityClass = `severity-${region.severity.toLowerCase()}`;
      const item = document.createElement("div");
      item.className = "region-item";
      item.style.borderLeftColor = region.color;

      item.innerHTML = `
        <div class="region-item-header">
          <span class="region-name">
            <span class="region-color-dot" style="background:${region.color}"></span>
            Area ${region.id}
          </span>
          <span class="severity-badge ${severityClass}">${region.severity}</span>
        </div>
        <div class="region-details">
          <div class="region-detail">
            Posisi: <strong>${region.bbox.x}, ${region.bbox.y}</strong>
          </div>
          <div class="region-detail">
            Ukuran: <strong>${region.bbox.width}×${region.bbox.height}px</strong>
          </div>
          <div class="region-detail">
            Luas: <strong>${region.area_percent}%</strong>
          </div>
          <div class="region-detail">
            Intensitas: <strong>${region.intensity}%</strong>
          </div>
          <div class="region-intensity-bar">
            <div class="region-intensity-fill" style="width:${region.intensity}%;background:${region.color}"></div>
          </div>
        </div>
      `;
      regionsList.appendChild(item);
    });
  } else {
    regionsEmpty.style.display = "flex";
  }

  // ─── ELA + Grad-CAM ───────────────────────────────────────
  $("#imgOriginal").src = `data:image/png;base64,${images.original}`;
  $("#imgELA").src = `data:image/png;base64,${images.ela}`;

  // Multi-quality ELA images
  if (images.ela_75) {
    $("#imgELA75").src = `data:image/png;base64,${images.ela_75}`;
  }
  if (images.ela_60) {
    $("#imgELA60").src = `data:image/png;base64,${images.ela_60}`;
  }

  // Setup ELA tab switching
  setupElaTabs();

  if (images.gradcam) {
    $("#imgGradCAM").src = `data:image/png;base64,${images.gradcam}`;
    $("#cardGradCAM").style.display = "block";
  } else {
    $("#cardGradCAM").style.display = "none";
  }

  // Show results
  resultsContainer.style.display = "block";
  resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ─── ELA Tab Switching ──────────────────────────────────────
function setupElaTabs() {
  const tabs = document.querySelectorAll(".ela-tab");
  const imgELA = $("#imgELA");
  const imgELA75 = $("#imgELA75");
  const imgELA60 = $("#imgELA60");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const quality = tab.dataset.quality;

      // Hide all ELA images
      imgELA.style.display = "none";
      imgELA75.style.display = "none";
      imgELA60.style.display = "none";

      // Show selected
      if (quality === "90") {
        imgELA.style.display = "block";
      } else if (quality === "75") {
        imgELA75.style.display = "block";
      } else if (quality === "60") {
        imgELA60.style.display = "block";
      }
    });
  });

  // Reset to Q90
  imgELA.style.display = "block";
  imgELA75.style.display = "none";
  imgELA60.style.display = "none";
}

// ─── New Analysis ───────────────────────────────────────────
btnNewAnalysis.addEventListener("click", () => {
  resetUpload();
  $("#annotatedSection").style.display = "none";
  $("#detect").scrollIntoView({ behavior: "smooth" });
});
