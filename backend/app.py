from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import soundfile as sf
import librosa
import re
import io
import os
import gc
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from huggingface_hub import hf_hub_download
import os
# Use HF token for authenticated downloads
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HF_TOKEN", "")
app = FastAPI(title="Bio-acoustica API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ─────────────────────────────────────────────────────────
REPO_ID       = "Kritzzz11/bio-acosutica"
SR            = 32000
N_WINDOWS     = 12
WINDOW_SAMPLES= SR * 5
FILE_SAMPLES  = SR * 60
N_MELS        = 256
N_FFT         = 2048
HOP           = 512
FMIN          = 20
FMAX          = 16000
TOP_DB        = 80
EPS           = 1e-5
FNAME_RE      = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

# ── Global model holders ──────────────────────────────────────────────
models = {}
data   = {}

# ── TaxaMoE architecture ──────────────────────────────────────────────
class CrossTaxaContrastiveAdapter(nn.Module):
    def __init__(self, input_dim=1536, output_dim=512, hidden_dim=768, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))
    def forward(self, x):
        return F.normalize(self.encoder(x), dim=-1)

class AvesHead(nn.Module):
    def __init__(self, input_dim=512, n_species=162, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(256, n_species))
    def forward(self, x): return self.net(x)

class InsectaHead(nn.Module):
    def __init__(self, input_dim=512, n_species=28, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(128, n_species))
    def forward(self, x): return self.net(x)

class PrototypicalHead(nn.Module):
    def __init__(self, input_dim=512, n_species=35, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(dropout))
        self.prototypes  = nn.Parameter(torch.randn(n_species, 256))
        self.temperature = nn.Parameter(torch.tensor(10.0))
    def forward(self, x):
        h = F.normalize(self.proj(x), dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        return torch.matmul(h, p.T) * F.softplus(self.temperature)

class TaxaMoE(nn.Module):
    def __init__(self, taxonomy_df, input_dim=1536):
        super().__init__()
        self.n_aves     = (taxonomy_df['taxa_group']==0).sum()
        self.n_insecta  = (taxonomy_df['taxa_group']==1).sum()
        self.n_amphibia = (taxonomy_df['taxa_group']==2).sum()
        self.n_mammalia = (taxonomy_df['taxa_group']==3).sum()
        self.n_reptilia = (taxonomy_df['taxa_group']==4).sum()
        self.n_classes  = len(taxonomy_df)
        self.register_buffer('aves_idx',
            torch.tensor(taxonomy_df[taxonomy_df['taxa_group']==0].index.tolist()))
        self.register_buffer('insecta_idx',
            torch.tensor(taxonomy_df[taxonomy_df['taxa_group']==1].index.tolist()))
        self.register_buffer('amphibia_idx',
            torch.tensor(taxonomy_df[taxonomy_df['taxa_group']==2].index.tolist()))
        self.register_buffer('mammalia_idx',
            torch.tensor(taxonomy_df[taxonomy_df['taxa_group']==3].index.tolist()))
        self.register_buffer('reptilia_idx',
            torch.tensor(taxonomy_df[taxonomy_df['taxa_group']==4].index.tolist()))
        self.adapter      = CrossTaxaContrastiveAdapter(input_dim, 512)
        self.router       = nn.Sequential(
            nn.Linear(512,128), nn.GELU(), nn.Linear(128,5))
        self.aves_head     = AvesHead(512, self.n_aves)
        self.insecta_head  = InsectaHead(512, self.n_insecta)
        self.amphibia_head = PrototypicalHead(512, self.n_amphibia)
        self.mammalia_head = PrototypicalHead(512, self.n_mammalia)
        self.reptilia_head = PrototypicalHead(512, self.n_reptilia)

    def forward(self, x):
        z  = self.adapter(x)
        rw = F.softmax(self.router(z), dim=-1)
        batch  = x.shape[0]
        logits = torch.zeros(batch, self.n_classes, device=x.device)
        logits[:, self.aves_idx]     = self.aves_head(z)     * rw[:, 0:1]
        logits[:, self.insecta_idx]  = self.insecta_head(z)  * rw[:, 1:2]
        logits[:, self.amphibia_idx] = self.amphibia_head(z) * rw[:, 2:3]
        logits[:, self.mammalia_idx] = self.mammalia_head(z) * rw[:, 3:4]
        logits[:, self.reptilia_idx] = self.reptilia_head(z) * rw[:, 4:5]
        return logits

    def get_router_weights(self, x):
        z  = self.adapter(x)
        rw = F.softmax(self.router(z), dim=-1)
        return rw.detach().numpy()

# ── Model loading ─────────────────────────────────────────────────────
def download(filename):
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="model",
        cache_dir="/tmp/wildecho_cache",
        token=os.getenv("HF_TOKEN"),
    )

@app.on_event("startup")
async def load_models():
    print("Loading models from HuggingFace...")

    # Load taxonomy + labels
    tax_path    = download("datasets/taxonomy.csv")
    sub_path    = download("datasets/sample_submission.csv")
    lbl_path    = download("datasets/labels.csv")
    labels_path = download("datasets/train_soundscapes_labels.csv")

    taxonomy   = pd.read_csv(tax_path)
    sample_sub = pd.read_csv(sub_path)
    bc_labels  = pd.read_csv(lbl_path).reset_index()
    bc_labels.columns = ['bc_index', 'scientific_name']

    def get_taxa_group(class_name):
        return {'Aves':0,'Insecta':1,'Amphibia':2,'Mammalia':3,'Reptilia':4}.get(class_name,-1)

    taxonomy['taxa_group'] = taxonomy['class_name'].map(get_taxa_group)

    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    N_CLASSES      = len(PRIMARY_LABELS)
    label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}

    NO_LABEL = len(bc_labels)
    mapping  = taxonomy.merge(bc_labels, on='scientific_name', how='left')
    mapping['bc_index'] = mapping['bc_index'].fillna(NO_LABEL).astype(int)
    lbl2bc   = mapping.set_index('primary_label')['bc_index']
    BC_INDICES  = [int(lbl2bc.loc[c]) if c in lbl2bc.index else NO_LABEL for c in PRIMARY_LABELS]
    MAPPED_MASK = [idx != NO_LABEL for idx in BC_INDICES]
    MAPPED_POS  = [i for i, m in enumerate(MAPPED_MASK) if m]
    MAPPED_BC   = [BC_INDICES[i] for i in MAPPED_POS]

    data['PRIMARY_LABELS'] = PRIMARY_LABELS
    data['N_CLASSES']      = N_CLASSES
    data['label_to_idx']   = label_to_idx
    data['taxonomy']       = taxonomy
    data['MAPPED_POS']     = MAPPED_POS
    data['MAPPED_BC']      = MAPPED_BC

    # Build Bayesian prior
    sc_df = pd.read_csv(labels_path)
    sc_df['end_sec'] = pd.to_timedelta(sc_df['end']).dt.total_seconds().astype(int)

    def parse_meta(fname):
        m = FNAME_RE.match(fname)
        if not m: return 'unknown', -1
        _, site, _, hms = m.groups()
        return site, int(hms[:2])

    sc_df['site'] = sc_df['filename'].apply(lambda x: parse_meta(x)[0])
    sc_df['hour'] = sc_df['filename'].apply(lambda x: parse_meta(x)[1])

    global_counts = np.zeros(N_CLASSES, dtype=np.float64)
    sh_counts, sh_windows = {}, {}
    total_windows = 0

    for _, row in sc_df.iterrows():
        key = (row['site'], row['hour'])
        if key not in sh_counts:
            sh_counts[key]  = np.zeros(N_CLASSES, dtype=np.float64)
            sh_windows[key] = 0
        for lbl in str(row['primary_label']).split(';'):
            lbl = lbl.strip()
            if lbl in label_to_idx:
                global_counts[label_to_idx[lbl]] += 1
                sh_counts[key][label_to_idx[lbl]] += 1
        sh_windows[key] += 1
        total_windows   += 1

    data['global_prior'] = global_counts / (total_windows + 1e-8)
    data['sh_counts']    = sh_counts
    data['sh_windows']   = sh_windows

    # Load Perch ONNX
    perch_path = download("perch_v2.onnx")
    perch_sess = ort.InferenceSession(perch_path, providers=["CPUExecutionProvider"])
    models['perch']      = perch_sess
    models['perch_in']   = perch_sess.get_inputs()[0].name
    models['perch_out']  = {o.name: i for i, o in enumerate(perch_sess.get_outputs())}

    # Load ProtoSSM
    download("protossm_v4.onnx.data")
    proto_path = download("protossm_v4.onnx")
    models['proto'] = ort.InferenceSession(proto_path, providers=["CPUExecutionProvider"])

    # Load SED folds (use fold0 + fold2 only for memory)
    sed_sessions = []
    for fold in [0, 2]:
        p = download(f"sed_fold{fold}.onnx")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        sed_sessions.append(
            ort.InferenceSession(p, sess_options=so, providers=["CPUExecutionProvider"]))
    models['sed'] = sed_sessions

    # Load TaxaMoE
    taxamoe_path = download("taxamoe_model_v3.pt")
    torch.serialization.add_safe_globals([np.dtype])
    taxonomy_indexed = taxonomy.reset_index(drop=True)
    taxamoe = TaxaMoE(taxonomy_indexed, input_dim=1770)
    ckpt = torch.load(taxamoe_path, map_location='cpu', weights_only=False)
    taxamoe.load_state_dict(ckpt['model_state'])
    taxamoe.eval()
    models['taxamoe'] = taxamoe

    print(f"All models loaded. N_CLASSES={N_CLASSES}")

# ── Audio processing ──────────────────────────────────────────────────
def read_audio(audio_bytes):
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32', always_2d=False)
    if y.ndim == 2: y = y.mean(1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:
        y = y[:FILE_SAMPLES]
    return y

def audio_to_mel(chunks):
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
        s = librosa.power_to_db(s, top_db=TOP_DB)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)

def get_prior(site, hour):
    SHRINKAGE = 8
    key = (site, hour)
    sh_counts  = data['sh_counts']
    sh_windows = data['sh_windows']
    global_prior = data['global_prior']
    if key in sh_counts:
        n = sh_windows[key]
        w = n / (n + SHRINKAGE)
        local = sh_counts[key] / (n + 1e-8)
        return np.clip(w * local + (1-w) * global_prior, 1e-4, 1-1e-4).astype(np.float32)
    return np.clip(global_prior, 1e-4, 1-1e-4).astype(np.float32)

# ── Main inference endpoint ───────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    N_CLASSES     = data['N_CLASSES']
    PRIMARY_LABELS= data['PRIMARY_LABELS']
    MAPPED_POS    = data['MAPPED_POS']
    MAPPED_BC     = data['MAPPED_BC']

    audio_bytes = await file.read()
    y = read_audio(audio_bytes)
    chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

    # ── Perch embeddings ──────────────────────────────────────────────
    perch_sess = models['perch']
    outs = perch_sess.run(None, {models['perch_in']: chunks})
    emb_te     = outs[models['perch_out']['embedding']].astype(np.float32)
    logits_raw = outs[models['perch_out']['label']].astype(np.float32)

    perch_logits = np.zeros((N_WINDOWS, N_CLASSES), dtype=np.float32)
    perch_logits[:, MAPPED_POS] = logits_raw[:, MAPPED_BC]
    emb_full = np.hstack([emb_te, perch_logits])

    # ── TaxaMoE ───────────────────────────────────────────────────────
    taxamoe = models['taxamoe']
    with torch.no_grad():
        emb_t      = torch.tensor(emb_full, dtype=torch.float32)
        taxa_logits= taxamoe(emb_t)
        taxa_probs = torch.sigmoid(taxa_logits).numpy()
        router_w   = taxamoe.get_router_weights(emb_t)

    # ── ProtoSSM ──────────────────────────────────────────────────────
    proto_sess = models['proto']

    def site_to_int(s):
        try: return int(str(s).replace('S','').lstrip('0') or '0')
        except: return 0

    site_arr = np.array([0], dtype=np.int64)
    hour_arr = np.array([12], dtype=np.int64)

    out = proto_sess.run(None, {
        'emb':    emb_te[None].astype(np.float32),
        'logits': perch_logits[None].astype(np.float32),
        'site':   site_arr,
        'hour':   hour_arr,
    })
    proto_probs = (1 / (1 + np.exp(-out[0].squeeze(0)))).astype(np.float32)

    # Apply prior in logit space
    prior_p = get_prior('unknown', 12)
    proto_logits_raw = np.log(np.clip(proto_probs, 1e-7, 1) /
                              np.clip(1-proto_probs, 1e-7, 1))
    prior_logodds    = np.log(prior_p) - np.log(1-prior_p)
    proto_probs_adj  = (1/(1+np.exp(-(proto_logits_raw + 0.625*prior_logodds)))).astype(np.float32)

    # ── SED ───────────────────────────────────────────────────────────
    mel   = audio_to_mel(chunks)
    p_sum = np.zeros((N_WINDOWS, N_CLASSES), dtype=np.float32)
    for sess in models['sed']:
        o = sess.run(None, {sess.get_inputs()[0].name: mel})
        p_sum += 0.5*(1/(1+np.exp(-o[0]))) + 0.5*(1/(1+np.exp(-o[1].max(axis=1))))
    sed_probs = p_sum / len(models['sed'])
    sed_probs = gaussian_filter1d(sed_probs, sigma=0.65, axis=0, mode='nearest')

    # ── Rank blend ────────────────────────────────────────────────────
    def rank(x): return pd.DataFrame(np.clip(x, EPS, 1-EPS)).rank(
        axis=0, pct=True).to_numpy(np.float32)

    rb = np.power(rank(sed_probs),    1.2)
    rc = np.power(rank(proto_probs_adj), 1.2)
    rd = np.power(rank(1/(1+np.exp(-np.clip(perch_logits,-10,10)))), 1.2)
    blended = 0.52 * rb + 0.37 * rc + 0.11 * rd

    # ── Per-window max → file-level score ─────────────────────────────
    file_scores = blended.max(axis=0)

    # ── Top 10 detections ─────────────────────────────────────────────
    top_idx   = np.argsort(file_scores)[::-1][:10]
    taxonomy  = data['taxonomy']

    detections = []
    for idx in top_idx:
        label     = PRIMARY_LABELS[idx]
        score     = float(file_scores[idx])
        tax_row   = taxonomy[taxonomy['primary_label'] == label]
        class_name= tax_row['class_name'].values[0] if len(tax_row) else 'Unknown'
        sci_name  = tax_row['scientific_name'].values[0] if len(tax_row) else label
        detections.append({
            "label":      label,
            "score":      round(score, 4),
            "class_name": class_name,
            "sci_name":   sci_name,
        })

    # ── Router weights (mean across windows) ──────────────────────────
    router_mean = router_w.mean(axis=0).tolist()
    taxa_names  = ['Aves', 'Insecta', 'Amphibia', 'Mammalia', 'Reptilia']

    # ── Spectrogram for frontend ───────────────────────────────────────
    S = librosa.feature.melspectrogram(
        y=y[:SR*10], sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=128, fmin=FMIN, fmax=FMAX, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    spectrogram = S_db.tolist()

    return {
        "detections":   detections,
        "router_weights": {taxa_names[i]: round(router_mean[i], 4) for i in range(5)},
        "spectrogram":  spectrogram,
        "n_windows":    N_WINDOWS,
        "duration_sec": 60,
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Bio-acoustica API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": len(models) > 0}