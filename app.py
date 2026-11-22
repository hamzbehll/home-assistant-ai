import os
import cv2
import json
import time
import uuid
import sqlite3
import threading
from pathlib import Path
from datetime import datetime
from collections import deque, defaultdict

import numpy as np
import streamlit as st
from ultralytics import YOLO

# Ev Asistani v2.6 - single webcam, consistent object preprocessing,
# local-feature matching with temporal confirmation.
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "ev_asistani.db"
ITEMS_DIR = BASE_DIR / "item_samples"
ITEMS_DIR.mkdir(exist_ok=True)
MODEL_PATH = str(BASE_DIR / "yolo11n.pt")

st.set_page_config(page_title="Ev Asistani v2.6", page_icon="🏠", layout="wide")

TR_NAMES = {
    "person": "İnsan", "bicycle": "Bisiklet", "car": "Araba", "motorcycle": "Motosiklet",
    "airplane": "Uçak", "bus": "Otobüs", "train": "Tren", "truck": "Kamyon", "boat": "Tekne",
    "traffic light": "Trafik isigi", "fire hydrant": "Yangin muslugu", "stop sign": "Dur levhasi",
    "parking meter": "Park sayaci", "bench": "Bank", "bird": "Kus", "cat": "Kedi", "dog": "Köpek",
    "horse": "At", "sheep": "Koyun", "cow": "Inek", "elephant": "Fil", "bear": "Ayi", "zebra": "Zebra",
    "giraffe": "Zürafa", "backpack": "Sırt çantası", "umbrella": "Şemsiye", "handbag": "El çantası",
    "tie": "Kravat", "suitcase": "Bavul", "frisbee": "Frizbi", "skis": "Kayak", "snowboard": "Snowboard",
    "sports ball": "Spor topu", "kite": "Ucurtma", "baseball bat": "Beyzbol sopasi",
    "baseball glove": "Beyzbol eldiveni", "skateboard": "Kaykay", "surfboard": "Surf tahtasi",
    "tennis racket": "Tenis raketi", "bottle": "Sise", "wine glass": "Şarap kadehi", "cup": "Bardak",
    "fork": "Çatal", "knife": "Bıçak", "spoon": "Kaşık", "bowl": "Kase", "banana": "Muz", "apple": "Elma",
    "sandwich": "Sandvic", "orange": "Portakal", "broccoli": "Brokoli", "carrot": "Havuc", "hot dog": "Sosisli",
    "pizza": "Pizza", "donut": "Donut", "cake": "Kek", "chair": "Sandalye", "couch": "Kanepe",
    "potted plant": "Saksı bitkisi", "bed": "Yatak", "dining table": "Yemek masası", "toilet": "Tuvalet",
    "tv": "Televizyon", "laptop": "Dizüstü bilgisayar", "mouse": "Bilgisayar faresi", "remote": "Kumanda",
    "keyboard": "Klavye", "cell phone": "Cep telefonu", "microwave": "Mikrodalga", "oven": "Fırın",
    "toaster": "Ekmek kızartma makinesi", "sink": "Lavabo", "refrigerator": "Buzdolabı", "book": "Kitap",
    "clock": "Saat", "vase": "Vazo", "scissors": "Makas", "teddy bear": "Oyuncak ayı",
    "hair drier": "Saç kurutma makinesi", "toothbrush": "Diş fırçası",
}

PERSONAL_CATEGORIES = {
    "cell phone": "Cep telefonu", "handbag": "El çantası", "backpack": "Sırt çantası"
}


def db_connect():
    return sqlite3.connect(str(DB_PATH), timeout=20, check_same_thread=False)


def init_db():
    # Existing v2.4/v2.5 databases are retained; these statements are additive only.
    with db_connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS detections(
            id INTEGER PRIMARY KEY, object_name TEXT, camera_name TEXT, room TEXT,
            seen_at TEXT, confidence REAL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS cameras(
            id INTEGER PRIMARY KEY, name TEXT UNIQUE, source TEXT, room TEXT)""")
        con.execute("""CREATE TABLE IF NOT EXISTS personal_items(
            id INTEGER PRIMARY KEY, item_name TEXT UNIQUE, category TEXT, owner TEXT,
            description TEXT, samples_json TEXT DEFAULT '[]', created_at TEXT)""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_detections_object_seen ON detections(object_name, id DESC)")


def get_items():
    with db_connect() as con:
        rows = con.execute("SELECT id,item_name,category,owner,description,samples_json FROM personal_items ORDER BY item_name").fetchall()
    out = []
    for r in rows:
        try:
            samples = json.loads(r[5] or "[]")
        except Exception:
            samples = []
        out.append({"id": r[0], "name": r[1], "category": r[2], "owner": r[3],
                    "description": r[4] or "", "samples": samples})
    return out


@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(MODEL_PATH if Path(MODEL_PATH).exists() else "yolo11n.pt")


MODEL_LOCK = threading.Lock()


def safe_crop(image, box=None, pad=0.06):
    """Crop the object with a small context margin; used for both samples and live detections."""
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    if box is None:
        # Center-square fallback for sample photos where the detector misses the object.
        side = min(w, h)
        x1, y1 = (w - side) // 2, (h - side) // 2
        x2, y2 = x1 + side, y1 + side
    else:
        x1, y1, x2, y2 = [int(v) for v in box]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        px, py = int(bw * pad), int(bh * pad)
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2, y2 = min(w, x2 + px), min(h, y2 + py)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # Normalize aspect ratio with border padding instead of stretching the object.
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    top = (side - ch) // 2
    bottom = side - ch - top
    left = (side - cw) // 2
    right = side - cw - left
    crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_REPLICATE)
    crop = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_AREA if side > 320 else cv2.INTER_CUBIC)
    return crop


def prepare_reference(raw_image, category, model):
    """Run the same detector/crop pipeline on uploaded examples as on live camera objects."""
    if raw_image is None:
        return None, False
    box = None
    try:
        with MODEL_LOCK:
            pred = model.predict(raw_image, verbose=False, conf=0.12, imgsz=512, max_det=15)[0]
        candidates = []
        for b in pred.boxes:
            cls_id = int(b.cls[0].item())
            if pred.names[cls_id] == category:
                xyxy = b.xyxy[0].tolist()
                area = max(0, xyxy[2] - xyxy[0]) * max(0, xyxy[3] - xyxy[1])
                candidates.append((area, xyxy))
        if candidates:
            box = max(candidates, key=lambda x: x[0])[1]
    except Exception:
        pass
    return safe_crop(raw_image, box), box is not None


# Keep ORB keypoint coordinates to enable RANSAC geometric verification.
def extract_features(crop):
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    orb = cv2.ORB_create(nfeatures=900, scaleFactor=1.2, nlevels=8, edgeThreshold=15, fastThreshold=10)
    keypoints, descriptors = orb.detectAndCompute(gray, None)            
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1), hist, alpha=1.0, norm_type=cv2.NORM_L1)
    points = np.float32([kp.pt for kp in keypoints]) if keypoints else np.empty((0, 2), dtype=np.float32)
    return {"des": descriptors, "points": points, "hist": hist, "kp": len(keypoints or [])}


def compare_features(query, sample):
    if query is None or sample is None:
        return {"score": 0.0, "inliers": 0, "good": 0, "color": 0.0}
    qd, sd = query["des"], sample["des"]
    good = []
    inliers = 0
    if qd is not None and sd is not None and len(qd) >= 6 and len(sd) >= 6:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(qd, sd, k=2)
        good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.76 * n.distance]
        if len(good) >= 6:
            src = np.float32([query["points"][m.queryIdx] for m in good]).reshape(-1, 1, 2)
            dst = np.float32([sample["points"][m.trainIdx] for m in good]).reshape(-1, 1, 2)
            try:
                _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                inliers = int(mask.ravel().sum()) if mask is not None else 0
            except cv2.error:
                inliers = 0
    corr = float(cv2.compareHist(query["hist"], sample["hist"], cv2.HISTCMP_CORREL))
    color = max(0.0, min(1.0, (corr + 1.0) / 2.0))
    local = min(1.0, inliers / 12.0) * 0.75 + min(1.0, len(good) / 30.0) * 0.25
    score = 0.88 * local + 0.12 * color
    return {"score": float(score), "inliers": inliers, "good": len(good), "color": color}


FEATURE_CACHE = {}
FEATURE_LOCK = threading.Lock()
ITEM_CACHE = {"time": 0.0, "items": []}


def get_item_features(item, model):
    all_features = []
    for path in item["samples"]:
        try:
            stat = os.stat(path)
            cache_key = (path, stat.st_mtime_ns, stat.st_size)
        except OSError:
            continue
        with FEATURE_LOCK:
            feat = FEATURE_CACHE.get(cache_key)
        if feat is None:
            raw = cv2.imread(path)
            if raw is None:
                continue
            feat = extract_features(raw)
            if feat is not None:
                with FEATURE_LOCK:
                    FEATURE_CACHE[cache_key] = feat
        if feat is not None:
            all_features.append(feat)
    return all_features


def identify_item(live_crop, category, model, items):
    qfeat = extract_features(live_crop)
    if qfeat is None:
        return None, []
    ranked = []
    for item in items:
        if item["category"] != category or not item["samples"]:
            continue
        sample_feats = get_item_features(item, model)
        if not sample_feats:
            continue
        scores = [compare_features(qfeat, feat) for feat in sample_feats]
        scores.sort(key=lambda x: x["score"], reverse=True)
        top = scores[:3]
        # Best view matters, but supporting views reduce single-photo coincidences.
        aggregate = 0.65 * top[0]["score"] + 0.35 * float(np.mean([x["score"] for x in top]))
        best = top[0]
        ranked.append({"item": item, "score": aggregate, "inliers": best["inliers"],
                       "good": best["good"], "color": best["color"], "samples": len(sample_feats)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    if not ranked:
        return None, []
    best = ranked[0]
    margin = best["score"] - ranked[1]["score"] if len(ranked) > 1 else 1.0
    # Conservative starting thresholds; the lab exposes scores for later calibration.
    accepted = best["score"] >= 0.34 and best["inliers"] >= 6 and margin >= 0.07
    return (best if accepted else None), ranked[:5]


class WebcamEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.thread = None
        self.running = False
        self.frame = None
        self.detections = []
        self.status = "Kamera kapalı"
        self.error = None
        self.fps = 0.0
        self.camera_index = 0
        self.confidence = 0.30
        self.last_log = {}
        self.vote_buffers = defaultdict(lambda: deque(maxlen=15))
        self.last_items_refresh = 0.0
        self.items = []
        self.debug = []

    def snapshot(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy(), list(self.detections),
                    self.status, self.error, self.fps, list(self.debug))

    def start(self, model, camera_index, confidence):
        if self.thread and self.thread.is_alive():
            return
        self.camera_index = int(camera_index)
        self.confidence = float(confidence)
        self.error = None
        self.running = True
        self.status = "Kamera açılıyor"
        self.vote_buffers.clear()
        self.thread = threading.Thread(target=self._loop, args=(model,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.status = "Kapatılıyor"

    def _refresh_items(self):
        if time.time() - self.last_items_refresh > 3:
            self.items = get_items()
            self.last_items_refresh = time.time()

    def _record(self, label, room, conf):
        now = time.time()
        if now - self.last_log.get(label, 0) < 15:
            return
        try:
            with db_connect() as con:
                con.execute("INSERT INTO detections(object_name,camera_name,room,seen_at,confidence) VALUES(?,?,?,?,?)",
                            (label, "Bilgisayar Kamerası", room, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), conf))
            self.last_log[label] = now
        except sqlite3.Error:
            pass

    def _loop(self, model):
        cap = None
        try:
            if os.name == "nt":
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(self.camera_index)
            else:
                cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                self.error = "Kamera açılamadı. Kamera numarasini kontrol et ve baska uygulamalarin kamerayi kullanmadigindan emin ol."
                self.status = "Kamera bağlantı hatası"
                self.running = False
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.status = "Canlı analiz"
            frames = 0
            fps_start = time.time()
            last_infer = 0.0
            last_detections = []
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                h, w = frame.shape[:2]
                if w > 960:
                    frame = cv2.resize(frame, (960, int(h * 960 / w)), interpolation=cv2.INTER_AREA)
                now = time.time()
                if now - last_infer >= 0.18:
                    self._refresh_items()
                    try:
                        with MODEL_LOCK:
                            result = model.track(
                                frame,
                                persist=True,
                                tracker="bytetrack.yaml",
                                verbose=False,
                                conf=self.confidence,
                                imgsz=512,
                                max_det=25,
                                device="cpu"
                            )[0]
                    except Exception as exc:
                        self.error = f"Model tahmin hatasi: {exc}"
                        time.sleep(0.2)
                        continue
                    found, debug_rows = [], []
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        label = result.names[cls_id]
                        conf = float(box.conf[0].item())
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        crop = safe_crop(frame, (x1, y1, x2, y2))
                        display = TR_NAMES.get(label, label)
                        owner = None
                        item_name = None
                        score_rows = []
                        if label in PERSONAL_CATEGORIES:
                            match, score_rows = identify_item(crop, label, model, self.items)
                            track_key = f"{label}:{int((x1+x2)/2)//100}:{int((y1+y2)/2)//100}"

                            buffer = self.vote_buffers[track_key]
                            buffer.append((now, match["item"]["id"] if match else None))

                            # 2.5 saniyeden eski gözlemleri temizle
                            while buffer and now - buffer[0][0] > 2.5:
                                buffer.popleft()

                            # Başarılı ve başarısız gözlemleri birlikte değerlendir
                            votes = [item_id for timestamp, item_id in buffer if item_id is not None]

                            if votes:
                                counts = {v: votes.count(v) for v in set(votes)}
                                voted_id, vote_count = max(counts.items(), key=lambda x: x[1])
                            else:
                                voted_id, vote_count = None, 0

                            # En az 4 olumlu oy ve %65 tutarlılık gerekli
                            if (voted_id is not None and vote_count >= 4
                                    and vote_count / len(buffer) >= 0.65):
                                selected = next((it for it in self.items if it["id"] == voted_id), None)
                                if selected:
                                    display = selected["name"]
                                    owner = selected["owner"]
                                    item_name = selected["name"]
                            elif self.items:
                                display = f"{display} · Kimlik doğrulanıyor"
                        found.append({"label": label, "confidence": conf, "display": display,
                                      "owner": owner, "item_name": item_name, "box": (x1,y1,x2,y2)})
                        if score_rows:
                            debug_rows.append({"Nesne": display, "Aday": score_rows[0]["item"]["name"],
                                               "Eşleşme skoru": f"%{score_rows[0]['score']*100:.1f}",
                                               "Geometrik eşleşme": score_rows[0]["inliers"],
                                               "Iyi eşleşme": score_rows[0]["good"]})
                    last_detections = found
                    self.debug = debug_rows
                    last_infer = now
                    for item in found:
                        self._record(item["label"], "Belirtilmedi", item["confidence"])
                # Draw last inference result on the latest camera frame.
                for item in last_detections:
                    x1,y1,x2,y2 = item["box"]
                    color = (40, 200, 80) if item["owner"] else (30, 180, 230)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    caption = item["display"]
                    if item["owner"]:
                        caption += f" | {item['owner']}"
                    cv2.putText(frame, caption[:55], (x1, max(22,y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, color, 2, cv2.LINE_AA)
                frames += 1
                elapsed = now - fps_start
                if elapsed >= 2:
                    self.fps = frames / elapsed
                    frames = 0
                    fps_start = now
                with self.lock:
                    self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.detections = last_detections
                    self.status = "Canlı analiz"
            self.status = "Kamera durduruldu"
        except Exception as exc:
            self.error = f"Kamera dongusu hatasi: {exc}"
            self.status = "Hata"
        finally:
            if cap is not None:
                cap.release()
            self.running = False


init_db()
if "engine" not in st.session_state:
    st.session_state.engine = WebcamEngine()
engine = st.session_state.engine

st.title("Ev Asistani v2.6")
st.caption("Tek bilgisayar kamerasi · YOLO nesne tespiti · ORB + geometrik doğrulama · coklu kare kimlik karari")

with st.sidebar:
    st.header("Kamera ve model")
    cam_index = st.number_input("Bilgisayar kamerasi numarasi", min_value=0, max_value=5, value=0, step=1)
    confidence = st.slider("YOLO nesne bulma esigi", 0.10, 0.85, 0.30, 0.05)
    st.caption("Bu esik nesnenin telefon/kitap vb. oldugunu bulma guvenidir; kisiye ait eşyanin kimligini gostermez.")
    start_col, stop_col = st.columns(2)
    start_clicked = start_col.button("Başlat", type="primary", use_container_width=True, disabled=engine.running)
    stop_clicked = stop_col.button("Durdur", use_container_width=True, disabled=not engine.running)
    if start_clicked:
        try:
            with st.spinner("YOLO modeli yukleniyor..."):
                detector = load_model()
            engine.start(detector, int(cam_index), confidence)
        except Exception as exc:
            st.error(f"Model yuklenemedi: {exc}")
    if stop_clicked:
        engine.stop()
    st.divider()
    st.caption("İşlemci dostu ayar: CPU, 512 px analiz, yaklaşık 5 analiz/sn. Ilk açılışta model indirilebilir.")

live_tab, items_tab, history_tab = st.tabs(["Canlı kamera", "Eşya tanımlama", "Tespit geçmişi"])

with live_tab:
    st.subheader("Bilgisayar kamerasinda canli tanima")
    st.markdown("Kamerayi baslat. Nesnelerin etrafinda kutular gorunecek. Kayitli kişisel eşyalar icin kimlik, birden fazla karede dogrulaninca gosterilecek.")
    @st.fragment(run_every=0.6)
    def live_panel():
        frame, detections, status, error, fps, debug = engine.snapshot()
        if error:
            st.error(error)
        st.caption(f"Durum: {status} · İşleme hızı: {fps:.1f} FPS")
        if frame is not None:
            st.image(frame, channels="RGB", use_container_width=True)
        elif not error:
            st.info("Kamera görüntüsu bekleniyor. Sol menuden Başlat'a bas.")
        if detections:
            cols = st.columns(min(3, len(detections)))
            for i, det in enumerate(detections):
                with cols[i % len(cols)]:
                    st.metric(det["display"], f"YOLO %{det['confidence']*100:.0f}",
                              delta=(det["owner"] or "Kimlik henüz doğrulanmadı"))
        if debug:
            with st.expander("Kimlik eşleşme ayrintilari"):
                st.dataframe(debug, use_container_width=True, hide_index=True)
        st.caption("Not: YOLO yuzdesi nesne sinifini bulma guvenidir. Eşleşme skoru ise kayitli fotoğrafla benzerlik sinyalidir; dogruluk garantisi degildir.")
    live_panel()

with items_tab:
    st.subheader("Kişisel eşya kayitlari")
    st.info("Bu surumde yalnizca bilgisayar kamerasi kullanılır. Esya fotoğrafi kaydedilirken YOLO ile nesne kirpilir; kirpma başarısızsa fotoğraf merkezinden kare bolge kullanılır.")
    with st.expander("Daha iyi referans fotoğraflari icin", expanded=True):
        st.markdown("""
        - Her eşya icin 12–20 fotoğraf yukle.
        - Telefonun arka yuzunu, on yuzunu, yanlarini ve capraz acilarini cek.
        - Cihazi kadraja buyuk ve net yerlestir; arka plan sade olsun.
        - Farklı acilar cek ama telefonun kendisini tamamen degistirme (kilif takiliysa orneklerin cogunda ayni kilif olsun).
        - Bir eşyanin butun fotoğraflarini ayni acida cekme; sistem farklı görünümleri de gormeli.
        """)
    with st.form("add_personal_item_v26", clear_on_submit=True):
        name = st.text_input("Eşya adı", placeholder="Siyah telefonum")
        category = st.selectbox("Eşya türü", list(PERSONAL_CATEGORIES), format_func=lambda x: PERSONAL_CATEGORIES[x])
        owner = st.text_input("Sahibi", placeholder="Ben")
        description = st.text_area("Ayırt edici ayrıntılar", placeholder="Mat siyah kilif, uc arka kamera, kosede cizik...")
        uploads = st.file_uploader("Örnek fotoğraflar (birden fazla)", type=["jpg","jpeg","png"], accept_multiple_files=True)
        save = st.form_submit_button("Eşya kaydını oluştur", type="primary", use_container_width=True)
        if save:
            if not name.strip() or not owner.strip():
                st.error("Eşya adı ve sahibi zorunlu.")
            elif len(uploads or []) < 3:
                st.error("En az 3 fotoğraf gerekli; önerimiz 12–20 farklı gorunum.")
            else:
                try:
                    detector = load_model()
                    saved_paths = []
                    used_detector = 0
                    for up in uploads:
                        raw = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                        if raw is None:
                            continue
                        crop, detected = prepare_reference(raw, category, detector)
                        if crop is None:
                            continue
                        used_detector += int(detected)
                        path = ITEMS_DIR / f"{uuid.uuid4().hex}.jpg"
                        if cv2.imwrite(str(path), crop):
                            saved_paths.append(str(path))
                    if len(saved_paths) < 3:
                        st.error("Fotoğraflar okunamadı. JPG/PNG dosyalariyla tekrar dene.")
                    else:
                        with db_connect() as con:
                            con.execute("INSERT INTO personal_items(item_name,category,owner,description,samples_json,created_at) VALUES(?,?,?,?,?,?)",
                                        (name.strip(), category, owner.strip(), description.strip(), json.dumps(saved_paths, ensure_ascii=False), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        ITEM_CACHE["time"] = 0
                        st.success(f"{name} kaydedildi: {len(saved_paths)} fotoğraf. {used_detector}/{len(saved_paths)} fotoğraf YOLO ile nesne kirpilabildi.")
                        st.rerun()
                except sqlite3.IntegrityError:
                    st.error("Bu eşya adi zaten kayitli. Mevcut eşyaya fotoğraf ekleyebilirsin.")
                except Exception as exc:
                    st.error(f"Kayit sirasinda hata: {exc}")

    items = get_items()
    st.divider()
    if items:
        for item in items:
            with st.expander(f"{item['name']} · {item['owner']} · {len(item['samples'])} fotoğraf"):
                st.write(f"Tur: {PERSONAL_CATEGORIES.get(item['category'], item['category'])}")
                st.write(f"Detay: {item['description'] or 'Yok'}")
                existing = [p for p in item["samples"] if os.path.exists(p)]
                if existing:
                    st.image(existing[:8], width=105)
                extra = st.file_uploader("Yeni referans fotoğrafları ekle", type=["jpg","jpeg","png"], accept_multiple_files=True, key=f"extra_{item['id']}")
                if st.button("Fotoğrafları ekle", key=f"add_{item['id']}", disabled=not extra):
                    try:
                        detector = load_model()
                        paths = []
                        for up in extra:
                            raw = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                            crop, _ = prepare_reference(raw, item["category"], detector)
                            if crop is not None:
                                path = ITEMS_DIR / f"{uuid.uuid4().hex}.jpg"
                                if cv2.imwrite(str(path), crop): paths.append(str(path))
                        if paths:
                            with db_connect() as con:
                                con.execute("UPDATE personal_items SET samples_json=? WHERE id=?",
                                            (json.dumps(item["samples"] + paths, ensure_ascii=False), item["id"]))
                            FEATURE_CACHE.clear()
                            st.success(f"{len(paths)} fotoğraf eklendi.")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Fotoğraf eklenemedi: {exc}")
                if st.button("Eşya kaydıni sil", key=f"delete_{item['id']}"):
                    with db_connect() as con:
                        con.execute("DELETE FROM personal_items WHERE id=?", (item["id"],))
                    for p in item["samples"]:
                        try: os.remove(p)
                        except OSError: pass
                    FEATURE_CACHE.clear()
                    st.rerun()
    else:
        st.warning("Henüz kişisel eşya kaydi yok.")

with history_tab:
    st.subheader("Son gorulme kayitlari")
    with db_connect() as con:
        rows = con.execute("SELECT object_name,camera_name,room,seen_at,confidence FROM detections ORDER BY id DESC LIMIT 100").fetchall()
    if rows:
        st.dataframe([{"Nesne": TR_NAMES.get(r[0],r[0]), "Kamera": r[1], "Konum": r[2],
                       "Zaman": r[3], "YOLO guveni": f"%{r[4]*100:.0f}"} for r in rows],
                     use_container_width=True, hide_index=True)
    else:
        st.info("Henüz tespit kaydi yok.")

st.divider()
st.caption("Ev Asistani v2.6 · Tek kamera odakli prototip · Kimlik karari coklu kare doğrulamasi gerektirir.")