"""使用 Streamlit 打造的 AI 輔助互動式影像處理與人臉分析工具。"""
import io

import cv2
import matplotlib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

st.set_page_config(page_title="影像處理與人臉分析", layout="wide")

MAX_SIDE = 2000  # 縮小過大的影像，以維持應用程式的執行效率


# ----------------------------------------------------------------- 輔助函式
def load_image(uploaded):
    """回傳（RGB uint8 陣列、影像中繼資料字典）。"""
    raw = uploaded.getvalue()
    img = Image.open(io.BytesIO(raw))
    fmt, mode = img.format, img.mode
    img = ImageOps.exif_transpose(img)
    orig_w, orig_h = img.size
    img = img.convert("RGB")  # 移除 alpha，並將調色盤或灰階模式轉為 RGB
    resized = False
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        resized = True
    meta = {
        "檔案名稱": uploaded.name,
        "格式": fmt,
        "解析度": f"{orig_w} x {orig_h} 像素",
        "原始色彩模式": mode,
        "色彩通道": "轉換後為 3 個通道（RGB）",
        "檔案大小": f"{len(raw) / 1024:.1f} KB",
    }
    if resized:
        meta["注意"] = f"已縮小至 {img.size[0]} x {img.size[1]} 以進行處理"
    return np.array(img), meta


def histogram_figure(channels):
    """channels：由（標籤、2D uint8 陣列、顏色）組成的清單。x 軸為 0–255，y 軸為像素數量。"""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for label, data, color in channels:
        hist = cv2.calcHist([np.ascontiguousarray(data)], [0], None, [256], [0, 256]).ravel()
        ax.plot(np.arange(256), hist, color=color, label=label, linewidth=1.2)
        ax.fill_between(np.arange(256), hist, color=color, alpha=0.15)
    ax.set_xlim(0, 255)
    ax.set_xlabel("像素強度（0–255）")
    ax.set_ylabel("像素數量")
    if len(channels) > 1:
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def parse_kernel(text):
    """解析文字矩陣，回傳（核心、錯誤訊息）。"""
    rows = [r for r in text.strip().splitlines() if r.strip()]
    if not rows:
        return None, "核心矩陣不可為空。"
    try:
        mat = [[float(v) for v in r.replace(",", " ").split()] for r in rows]
    except ValueError:
        return None, "核心矩陣包含非數值內容。"
    if len({len(r) for r in mat}) != 1:
        return None, "每一列的數值數量必須相同。"
    k = np.array(mat, dtype=np.float32)
    if not np.all(np.isfinite(k)):
        return None, "核心矩陣包含 inf 或 NaN。"
    h, w = k.shape
    if h % 2 == 0 or w % 2 == 0:
        return None, f"核心矩陣大小為 {h}x{w}。請使用奇數維度（3x3、5x5、7x7），讓矩陣具有中心。"
    if h > 15 or w > 15:
        return None, "核心矩陣過大（最大為 15x15）。"
    return k, None


def apply_kernel(rgb, kernel, normalize):
    """執行真正的 2D 卷積（翻轉核心），邊界採鏡射延伸。回傳（uint8 影像、說明）。"""
    note = ""
    k = kernel.copy()
    if normalize:
        s = float(k.sum())
        if abs(s) < 1e-8:
            note = "核心總和為 0，為避免除以零，已略過正規化。"
        else:
            k = k / s
    flipped = cv2.flip(k, -1)  # filter2D 執行的是互相關，因此需翻轉兩個軸以進行卷積
    out = cv2.filter2D(rgb.astype(np.float32), -1, flipped, borderType=cv2.BORDER_REFLECT)
    return np.clip(out, 0, 255).astype(np.uint8), note


def psnr(a, b):
    """PSNR = 10*log10(255^2 / MSE)，單位為 dB。"""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf"), mse
    return 10 * np.log10(255.0**2 / mse), mse


import os
import urllib.request

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detection_yunet_2023mar.onnx")


@st.cache_resource(show_spinner="正在下載 YuNet 人臉模型（僅需下載一次，約 230 KB）…")
def ensure_yunet():
    """回傳（路徑、錯誤訊息）。若本機檔案有效則直接使用，否則下載模型。"""
    def valid(p):
        # 模型約 230 KB；排除 Git LFS 指標檔與儲存的 HTML 頁面
        if not os.path.exists(p) or not 100_000 < os.path.getsize(p) < 2_000_000:
            return False
        with open(p, "rb") as f:
            head = f.read(16)
        return not head.lstrip().startswith((b"<", b"version"))
    if os.path.exists(YUNET_PATH) and not valid(YUNET_PATH):
        os.remove(YUNET_PATH)  # 移除先前下載失敗的檔案，以便重新下載
    if valid(YUNET_PATH):
        return YUNET_PATH, None
    try:
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    except Exception as exc:
        return None, f"無法下載 YuNet 模型（{exc}）。"
    if not valid(YUNET_PATH):
        os.remove(YUNET_PATH)
        return None, "下載的 YuNet 檔案無效。"
    return YUNET_PATH, None


def detect_yunet(rgb, model_path, score_thr, nms_thr):
    """YuNet 深度學習人臉偵測器。回傳邊界框（x,y,w,h）、分數、標註影像與縮放倍率。"""
    h, w = rgb.shape[:2]
    factor = min(4.0, 640 / max(h, w)) if max(h, w) < 640 else 1.0
    work = cv2.resize(rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC) if factor > 1 else rgb
    wh, ww = work.shape[:2]
    with open(model_path, "rb") as f:
        model_buf = np.frombuffer(f.read(), dtype=np.uint8)
    det = cv2.FaceDetectorYN.create("onnx", model_buf, np.empty(0, dtype=np.uint8),
                                    (ww, wh), score_thr, nms_thr, 5000)
    det.setInputSize((ww, wh))
    _, faces = det.detect(cv2.cvtColor(work, cv2.COLOR_RGB2BGR))
    out = rgb.copy()
    if faces is None:
        return [], [], out, factor
    faces = faces.copy()
    faces[:, :14] /= factor  # 將邊界框與 5 個特徵點轉回原始影像座標
    boxes = np.round(faces[:, :4]).astype(int)
    scores = faces[:, 14]
    thick = max(1, w // 300)
    for i, (f, (x, y, bw, bh), sc) in enumerate(zip(faces, boxes, scores), 1):
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 200, 0), thick)
        for lx, ly in f[4:14].reshape(5, 2):
            cv2.circle(out, (int(lx), int(ly)), max(1, thick), (255, 0, 0), -1)
        cv2.putText(out, f"#{i} {sc:.2f}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.4, w / 1200), (0, 200, 0), max(1, thick))
    return boxes, scores, out, factor


PRESETS = {
    "單位核心": "0 0 0\n0 1 0\n0 0 0",
    "邊緣銳化": "0 -1 0\n-1 5 -1\n0 -1 0",
    "高斯模糊 3x3": "1 2 1\n2 4 2\n1 2 1",
    "高斯模糊 5x5": "1 4 6 4 1\n4 16 24 16 4\n6 24 36 24 6\n4 16 24 16 4\n1 4 6 4 1",
    "方框模糊 3x3": "1 1 1\n1 1 1\n1 1 1",
    "浮雕": "-2 -1 0\n-1 1 1\n0 1 2",
    "邊緣偵測（拉普拉斯）": "-1 -1 -1\n-1 8 -1\n-1 -1 -1",
    "自訂": "",
}

# ----------------------------------------------------------------- 使用者介面
st.title("影像處理與人臉分析工具")

uploaded = st.sidebar.file_uploader("上傳影像", type=["png", "jpg", "jpeg", "bmp"])
if uploaded is None:
    st.info("請從側邊欄上傳 PNG、JPG/JPEG 或 BMP 影像以開始使用。")
    st.stop()

try:
    rgb, meta = load_image(uploaded)
except Exception as exc:  # corrupt or unsupported file
    st.error(f"無法讀取此影像：{exc}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. 影像資訊", "2. 色彩空間與直方圖", "3. 卷積與 PSNR", "4. 人臉偵測"]
)

# --- 功能 1
with tab1:
    c1, c2 = st.columns([2, 1])
    c1.image(rgb, caption="已載入的影像", use_container_width=True)
    c2.subheader("中繼資料")
    for k, v in meta.items():
        c2.markdown(f"**{k}:** {v}")

# --- 功能 2
with tab2:
    mode = st.radio(
        "檢視模式", ["原始影像", "R（紅色通道）", "G（綠色通道）", "B（藍色通道）",
                     "Y（亮度）", "U（色度）", "V（色度）"],
        horizontal=True,
    )
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    if mode == "原始影像":
        shown, hist = rgb, [("R", rgb[..., 0], "red"), ("G", rgb[..., 1], "green"), ("B", rgb[..., 2], "blue")]
    elif mode.startswith(("R（", "G（", "B（")):
        i = "RGB".index(mode[0])
        shown = rgb[..., i]
        hist = [(mode[0], shown, "rgb"[i] if False else ["red", "green", "blue"][i])]
    else:
        i = "YUV".index(mode[0])
        shown = yuv[..., i]
        hist = [(mode[0], shown, ["black", "tab:purple", "tab:orange"][i])]
    a, b = st.columns(2)
    a.image(shown, caption=mode, use_container_width=True, clamp=True)
    b.pyplot(histogram_figure(hist))
    st.caption("YUV 使用 OpenCV 的 BT.601 全範圍轉換；U 與 V 通道的偏移量為 128。")

# --- 功能 3
with tab3:
    left, right = st.columns([1, 2])
    with left:
        preset = st.selectbox("預設效果", list(PRESETS))
        text = st.text_area(
            "核心矩陣（每列一行，數值以空格或逗號分隔）",
            value=PRESETS[preset] or "1 1 1\n1 1 1\n1 1 1",
            height=180,
            key=f"kernel_{preset}",
        )
        normalize = st.checkbox("正規化（除以核心總和）", value=preset.startswith(("高斯模糊", "方框模糊")))
    kernel, err = parse_kernel(text)
    if err:
        right.error(err)
    else:
        filtered, note = apply_kernel(rgb, kernel, normalize)
        if note:
            right.warning(note)
        value, mse = psnr(rgb, filtered)
        view = right.radio("比較方式", ["並排比較", "混合疊加"], horizontal=True)
        if view == "並排比較":
            x, y = right.columns(2)
            x.image(rgb, caption="處理前", use_container_width=True)
            y.image(filtered, caption="處理後", use_container_width=True)
        else:
            alpha = right.slider("處理後影像的不透明度", 0.0, 1.0, 0.5)
            blend = cv2.addWeighted(rgb, 1 - alpha, filtered, alpha, 0)
            right.image(blend, caption="疊加結果", use_container_width=True)
        m1, m2 = right.columns(2)
        m1.metric("PSNR", "無限大 dB（影像相同）" if np.isinf(value) else f"{value:.2f} dB")
        m2.metric("均方誤差（MSE）", f"{mse:.2f}")
        st.latex(r"\mathrm{PSNR}=10\log_{10}\frac{255^{2}}{\mathrm{MSE}},\quad "
                 r"\mathrm{MSE}=\frac{1}{HWC}\sum_{i,j,c}\left(I(i,j,c)-K(i,j,c)\right)^{2}")

# --- 功能 4
with tab4:
    yunet_path, yunet_err = ensure_yunet()
    if yunet_err:
        st.error(f"{yunet_err} 請從 {YUNET_URL} 下載 YuNet 模型，並將檔案放在 app.py 同一個資料夾。")
        st.stop()
    y1, y2 = st.columns(2)
    score_thr = y1.slider("信心分數門檻（調高可減少誤判）", 0.3, 0.99, 0.7, 0.01)
    nms_thr = y2.slider("重疊門檻（NMS）", 0.1, 0.7, 0.3, 0.05)
    try:
        boxes, conf, annotated, factor = detect_yunet(rgb, yunet_path, score_thr, nms_thr)
    except cv2.error as exc:
        st.error("YuNet 無法載入模型檔案。請刪除應用程式資料夾中的 face_detection_yunet_2023mar.onnx，"
                 "再重新執行以重新下載。")
        st.caption(str(exc))
        st.stop()
    note = "YuNet 信心分數是模型的人臉分數（0–1）。藍點為人臉特徵點（眼睛、鼻子與嘴角）。"
    if factor > 1:
        st.caption(f"影像尺寸較小，已放大 {factor:.1f} 倍以進行偵測；邊界框會標示在原始影像上。")
    st.image(annotated, caption="人臉偵測結果", use_container_width=True)
    st.metric("偵測到的人臉數量", len(boxes))
    if len(boxes):
        st.table({
            "人臉": [f"#{i}" for i in range(1, len(boxes) + 1)],
            "邊界框（x, y, 寬, 高）": [str(tuple(int(v) for v in b)) for b in boxes],
            "信心分數": [f"{c:.2f}" for c in conf],
        })
    else:
        st.warning("找不到人臉。請嘗試降低信心分數門檻。")
    st.caption(note)