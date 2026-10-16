"""قناة "كلمة اليوم": يختار كلمة جديدة، يصنع فيديو Shorts، يرفعه على يوتيوب.
كل كلمة تُستعمل مرة واحدة فقط (تُسجَّل في used.json بعد نجاح الرفع).
"""
import asyncio
import glob
import io
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import arabic_reshaper
import edge_tts
import requests
from bidi.algorithm import get_display
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).parent
WORDS_FILE = ROOT / "words.json"
USED_FILE = ROOT / "used.json"

W, H = 1080, 1920
VOICE = os.getenv("TTS_VOICE", "ar-SA-HamedNeural")
REFILL_THRESHOLD = 60          # إذا بقي أقل من هذا العدد نولّد كلمات جديدة تلقائياً
GEMINI_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

PALETTES = [
    ((15, 76, 129), (6, 30, 60)),
    ((142, 36, 170), (49, 10, 76)),
    ((0, 121, 107), (0, 48, 43)),
    ((191, 54, 12), (80, 20, 5)),
    ((40, 53, 147), (15, 20, 70)),
    ((173, 20, 87), (70, 8, 36)),
    ((56, 142, 60), (20, 60, 25)),
]
ACCENTS = [(255, 214, 79), (255, 255, 255), (129, 236, 236), (255, 171, 145)]


# ---------------------------------------------------------------- أدوات عامة
_TASHKEEL = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def norm(word: str) -> str:
    """توحيد الكلمة لمنع التكرار (إزالة التشكيل وتوحيد الهمزات)."""
    w = _TASHKEEL.sub("", word.strip())
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")):
        w = w.replace(a, b)
    return w


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[-2000:])
        raise RuntimeError(f"فشل الأمر: {cmd[0]}")
    return p.stdout


# ---------------------------------------------------------------- Gemini
def gemini_json(prompt: str):
    key = os.environ["GEMINI_API_KEY"]
    last = None
    for model in GEMINI_MODELS:
        for _ in range(2):
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "temperature": 0.8,
                        },
                    },
                    timeout=90,
                )
                r.raise_for_status()
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
                return json.loads(text)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(4)
    raise RuntimeError(f"فشل Gemini: {last}")


def describe_word(word: str) -> dict:
    data = gemini_json(
        f"""الكلمة العربية: "{word}".
أعد JSON فقط بهذا الشكل:
{{"english": "ترجمة الكلمة بالإنجليزية بكلمة أو كلمتين مناسبتين للبحث عن صورة لها",
  "explanation": "جملة عربية واحدة بسيطة وواضحة (10 إلى 18 كلمة) تشرح معنى الكلمة لطفل أو متعلم مبتدئ، دون أن تبدأ بالكلمة نفسها ودون تشكيل"}}"""
    )
    english = str(data["english"]).strip()
    explanation = str(data["explanation"]).strip()
    if not english or len(explanation) < 8:
        raise ValueError("رد Gemini غير صالح")
    return {"english": english, "explanation": explanation}


def refill_words(words, used):
    """توليد كلمات جديدة تلقائياً حتى لا تنتهي القائمة أبداً."""
    seen = {norm(w) for w in words} | {norm(w) for w in used}
    sample = random.sample(words, min(30, len(words)))
    try:
        data = gemini_json(
            f"""أعطني 150 كلمة عربية جديدة، كلها أسماء محسوسة يمكن تصويرها في صورة
(حيوانات، نباتات، أدوات، أماكن، مهن، طعام، ظواهر طبيعية، ملابس، مركبات...).
كلمة واحدة أو كلمتان على الأكثر، دون تشكيل، دون تكرار، وبعيدة عن هذه الأمثلة:
{', '.join(sample)}
أعد JSON فقط: {{"words": ["...", "..."]}}"""
        )
        new = []
        for w in data["words"]:
            w = str(w).strip()
            if 2 <= len(w) <= 25 and norm(w) not in seen:
                seen.add(norm(w))
                new.append(w)
        print(f"تمت إضافة {len(new)} كلمة جديدة")
        return new
    except Exception as e:  # noqa: BLE001
        print("تعذّر توليد كلمات جديدة:", e)
        return []


# ---------------------------------------------------------------- الصور
def fetch_image(query: str):
    r = requests.get(
        "https://api.pexels.com/v1/search",
        headers={"Authorization": os.environ["PEXELS_API_KEY"]},
        params={"query": query, "orientation": "portrait", "per_page": 15},
        timeout=30,
    )
    r.raise_for_status()
    photos = r.json().get("photos", [])
    if not photos:
        return None, None
    p = random.choice(photos[:10])
    raw = requests.get(p["src"]["large2x"], timeout=60)
    raw.raise_for_status()
    img = Image.open(io.BytesIO(raw.content)).convert("RGB")
    return img, p.get("photographer", "")


# ---------------------------------------------------------------- الرسم
def find_font() -> str:
    patterns = [
        "**/NotoNaskhArabic*Bold*.ttf",
        "**/NotoNaskhArabic*.ttf",
        "**/NotoSansArabic*Bold*.ttf",
        "**/NotoSansArabic*.ttf",
        "**/Amiri*Bold*.ttf",
        "**/Amiri*.ttf",
        "**/DejaVuSans-Bold.ttf",
    ]
    for pat in patterns:
        found = sorted(glob.glob("/usr/share/fonts/" + pat, recursive=True))
        if found:
            return found[0]
    raise RuntimeError("لم يُعثر على خط عربي")


FONT_PATH = None


def shape(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))


def wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if draw.textlength(shape(trial), font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_text(img, text, y, size, fill, max_w=W - 140, max_lines=1, stroke=0, min_size=44):
    """يرسم نصاً عربياً في وسط الصورة مع تصغير تلقائي ليتسع."""
    global FONT_PATH
    if FONT_PATH is None:
        FONT_PATH = find_font()
    draw = ImageDraw.Draw(img)
    while True:
        font = ImageFont.truetype(FONT_PATH, size)
        lines = wrap(draw, text, font, max_w)
        widest = max(draw.textlength(shape(l), font=font) for l in lines)
        if (len(lines) <= max_lines and widest <= max_w) or size <= min_size:
            break
        size -= 4
    lh = int(size * 1.6)
    top = y - (len(lines) * lh) / 2 + lh / 2
    for i, line in enumerate(lines):
        draw.text(
            (W / 2, top + i * lh),
            shape(line),
            font=font,
            fill=fill,
            anchor="mm",
            stroke_width=stroke,
            stroke_fill=(0, 0, 0),
        )


def gradient(c1, c2):
    g = Image.linear_gradient("L").resize((W, H))
    return ImageOps.colorize(g, c1, c2)


def make_frames(word, explanation, photo, outdir):
    c1, c2 = random.choice(PALETTES)
    accent = random.choice(ACCENTS)

    # المشهد 1: الكلمة
    a = gradient(c1, c2)
    draw_text(a, "كلمة اليوم", 380, 80, accent)
    draw_text(a, word, H // 2, 300, (255, 255, 255), max_lines=2, min_size=120)
    a.save(outdir / "a.png")

    # المشهد 2: الصورة مع الكلمة
    cover = ImageOps.fit(photo, (W, H), Image.LANCZOS)
    shade = gradient((0, 0, 0), (0, 0, 0))
    mask = Image.linear_gradient("L").resize((W, H)).point(lambda v: int(max(0, v - 90) * 1.2))
    b = Image.composite(shade, cover, mask)
    draw_text(b, word, H - 330, 230, (255, 255, 255), max_lines=2, stroke=6, min_size=110)
    b.save(outdir / "b.png")

    # المشهد 3: الشرح فوق الصورة المموّهة
    blurred = cover.filter(ImageFilter.GaussianBlur(28))
    c = Image.blend(blurred, Image.new("RGB", (W, H), (0, 0, 0)), 0.58)
    draw_text(c, word, 330, 150, accent, max_lines=1, stroke=3)
    draw_text(c, explanation, H // 2 + 60, 86, (255, 255, 255), max_w=W - 160, max_lines=8)
    c.save(outdir / "c.png")


# ---------------------------------------------------------------- الصوت والفيديو
async def _edge(text, out):
    await edge_tts.Communicate(text, VOICE, rate="-8%").save(out)


def tts(text: str, out: str):
    for _ in range(3):
        try:
            asyncio.run(_edge(text, out))
            if os.path.getsize(out) > 1000:
                return
        except Exception as e:  # noqa: BLE001
            print("edge-tts:", e)
            time.sleep(4)
    print("استعمال gTTS كبديل")
    from gtts import gTTS

    gTTS(text, lang="ar").save(out)


def duration(path: str) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", path])
    return float(out.strip())


def build_video(word, explanation, photo, out_path):
    tmp = Path(tempfile.mkdtemp())
    make_frames(word, explanation, photo, tmp)

    tts(word, str(tmp / "w.mp3"))
    tts(explanation, str(tmp / "e.mp3"))
    dw, de = duration(str(tmp / "w.mp3")), duration(str(tmp / "e.mp3"))

    gap, tail = 2.5, 0.9
    fmt = "aformat=sample_rates=24000:channel_layouts=mono"
    run([
        "ffmpeg", "-y",
        "-i", str(tmp / "w.mp3"),
        "-f", "lavfi", "-t", str(gap), "-i", "anullsrc=r=24000:cl=mono",
        "-i", str(tmp / "e.mp3"),
        "-f", "lavfi", "-t", str(tail), "-i", "anullsrc=r=24000:cl=mono",
        "-filter_complex",
        f"[0:a]{fmt}[a0];[1:a]{fmt}[a1];[2:a]{fmt}[a2];[3:a]{fmt}[a3];"
        "[a0][a1][a2][a3]concat=n=4:v=0:a=1[a]",
        "-map", "[a]", str(tmp / "audio.mp3"),
    ])

    t_a = dw + 0.3
    t_b = gap - 0.3
    t_c = de + tail
    (tmp / "list.txt").write_text(
        f"file '{tmp}/a.png'\nduration {t_a:.2f}\n"
        f"file '{tmp}/b.png'\nduration {t_b:.2f}\n"
        f"file '{tmp}/c.png'\nduration {t_c:.2f}\n"
        f"file '{tmp}/c.png'\n"
    )
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
        "-i", str(tmp / "audio.mp3"),
        "-vf", "fps=30,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart",
        out_path,
    ])
    print(f"مدة الفيديو ≈ {t_a + t_b + t_c:.1f} ثانية")


# ---------------------------------------------------------------- يوتيوب
def upload_youtube(path, title, description, tags):
    creds = Credentials(
        None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    creds.refresh(Request())
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:4900],
            "tags": tags,
            "categoryId": "27",  # Education
            "defaultLanguage": "ar",
            "defaultAudioLanguage": "ar",
        },
        "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(path, mimetype="video/mp4", resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    return resp["id"]


# ---------------------------------------------------------------- الرئيسي
def main():
    words = load_json(WORDS_FILE, [])
    used = load_json(USED_FILE, [])
    used_norm = {norm(w) for w in used}

    # إزالة التكرار داخل القائمة نفسها
    seen, clean = set(), []
    for w in words:
        n = norm(w)
        if n and n not in seen:
            seen.add(n)
            clean.append(w.strip())
    words = clean

    available = [w for w in words if norm(w) not in used_norm]
    if len(available) < REFILL_THRESHOLD:
        extra = refill_words(words, used)
        words += extra
        available += extra
        save_json(WORDS_FILE, words)
    if not available:
        sys.exit("لا توجد كلمات متاحة")

    random.shuffle(available)
    chosen = None
    out_path = str(ROOT / "out.mp4")
    for word in available[:8]:
        try:
            print("المحاولة مع الكلمة:", word)
            info = describe_word(word)
            photo, author = fetch_image(info["english"])
            if photo is None:
                print("لا توجد صورة، كلمة أخرى")
                continue
            build_video(word, info["explanation"], photo, out_path)
            chosen = (word, info, author)
            break
        except Exception as e:  # noqa: BLE001
            print("فشلت الكلمة:", word, "→", e)
    if not chosen:
        sys.exit("فشلت كل المحاولات")

    word, info, author = chosen
    title = f"{word} | معنى كلمة {word} #shorts"
    desc = (
        f"{word}: {info['explanation']}\n\n"
        "كلمة جديدة كل يوم، تعلّم العربية بسهولة.\n"
        f"الصورة: {author} / Pexels\n\n"
        "#shorts #تعلم_العربية #كلمة_اليوم #اللغة_العربية"
    )
    tags = ["shorts", "تعلم العربية", "معنى كلمة", word, "اللغة العربية", "كلمة اليوم"]
    vid = upload_youtube(out_path, title, desc, tags)
    print("تم الرفع: https://youtu.be/" + vid)

    # تسجيل الكلمة فوراً بعد نجاح الرفع
    used.append(word)
    save_json(USED_FILE, sorted(set(used)))


if __name__ == "__main__":
    main()
