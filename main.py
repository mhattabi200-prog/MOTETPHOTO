"""قناة "كلمة وصورة": كلمة عربية + ملصق (إيموجي) + شرح بسيط، تُرفع تلقائياً على يوتيوب.
- لا بنك كلمات: Gemini يقترح كلمات جديدة في كل مرة.
- لا تتكرر كلمة أبداً: كل كلمة تُقارن بـ used.json (بعد توحيد الكتابة).
- لا صور خارجية: الملصق إيموجي من Noto Emoji.
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
from PIL import Image, ImageDraw, ImageFont, ImageOps

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


ROOT = Path(__file__).parent
USED_FILE = ROOT / "used.json"

W, H = 1080, 1920
VOICE = os.getenv("TTS_VOICE", "ar-SA-HamedNeural")
GEMINI_PREFERRED = [
    m for m in (
        os.getenv("GEMINI_MODEL", ""),
        "gemini-flash-latest",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
    ) if m
]
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

LETTERS = list("أبتثجحخدذرزسشصضطظعغفقكلمنهوي")
CATEGORIES = [
    "حيوانات", "طيور", "حشرات", "أسماك وحيوانات البحر", "نباتات وأشجار", "فواكه", "خضروات",
    "طعام وحلويات", "مشروبات", "ملابس وإكسسوارات", "أدوات المنزل", "أدوات العمل والحِرف",
    "مهن", "وسائل النقل", "أماكن وبنايات", "ظواهر الطبيعة", "أجزاء الجسم", "مشاعر",
    "صفات (كبير، سريع...)", "أفعال يومية", "ألوان وأشكال", "الفصول والوقت", "رياضة وألعاب",
    "موسيقى وفن", "الفضاء والكواكب", "المدرسة والدراسة", "السفر والعطلات", "الصحة والطب",
    "التقنية والأجهزة", "الأسرة والمجتمع",
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

_TASHKEEL = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_ARABIC_ONLY = re.compile(r"^[\u0621-\u064A ]{2,25}$")


# ---------------------------------------------------------------- أدوات عامة
def norm(word: str) -> str:
    """توحيد الكلمة لمنع التكرار (إزالة التشكيل وتوحيد الهمزات والتاء المربوطة)."""
    w = _TASHKEEL.sub("", word.strip())
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")):
        w = w.replace(a, b)
    w = re.sub(r"^ال", "", w)  # "الأسد" و"أسد" نفس الكلمة
    return w


def load_used():
    if USED_FILE.exists():
        return json.loads(USED_FILE.read_text(encoding="utf-8"))
    return []


def save_used(used):
    USED_FILE.write_text(
        json.dumps(sorted(set(used)), ensure_ascii=False, indent=1), encoding="utf-8"
    )


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[-2000:])
        raise RuntimeError(f"فشل الأمر: {cmd[0]}")
    return p.stdout


# ---------------------------------------------------------------- Gemini
def discover_models(key):
    """يسأل Google عن النماذج المتاحة فعلاً (حتى لا يتعطل المشروع عند إيقاف نموذج قديم)."""
    try:
        r = requests.get(f"{GEMINI_BASE}/models?pageSize=200",
                         headers={"x-goog-api-key": key}, timeout=20)
        r.raise_for_status()
        names = []
        for m in r.json().get("models", []):
            n = m["name"].split("/", 1)[1]
            if "generateContent" not in m.get("supportedGenerationMethods", []):
                continue
            if "flash" not in n or not n.startswith("gemini"):
                continue
            if any(x in n for x in ("image", "tts", "audio", "live", "embedding",
                                    "robotics", "computer-use", "thinking")):
                continue
            names.append(n)

        def rank(n):
            v = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
            return ("lite" in n, "preview" in n or "exp" in n,
                    -(float(v.group(1)) if v else 0), n)

        found = sorted(names, key=rank)
        log(f"نماذج متاحة: {found[:8]}")
        return found
    except Exception as e:  # noqa: BLE001
        log(f"تعذّر جلب قائمة النماذج: {e}")
        return []


_DEAD_MODELS = set()


def _try_models(key, models, prompt):
    last = None
    for model in models:
        if model in _DEAD_MODELS:
            continue
        for attempt in range(2):
            try:
                r = requests.post(
                    f"{GEMINI_BASE}/models/{model}:generateContent",
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "temperature": 1.0,
                        },
                    },
                    timeout=40,
                )
                if r.status_code == 404:           # النموذج غير موجود/أُوقف: لا فائدة من التكرار
                    _DEAD_MODELS.add(model)
                    last = f"{model}: 404 (غير متاح)"
                    log(last)
                    break
                if r.status_code == 429:           # تجاوز الحصة المؤقتة
                    last = f"{model}: 429 {r.text[:150]}"
                    log(last)
                    time.sleep(8)
                    continue
                r.raise_for_status()
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
                log(f"Gemini ✓ ({model})")
                return json.loads(text)
            except Exception as e:  # noqa: BLE001
                last = f"{model}: {e}"
                log(last)
                time.sleep(2)
    raise RuntimeError(last or "لا توجد نماذج")


def gemini_json(prompt: str):
    key = os.environ["GEMINI_API_KEY"]
    try:
        return _try_models(key, GEMINI_PREFERRED, prompt)
    except RuntimeError as first:
        log(f"فشلت النماذج المفضلة ({first})، البحث عن نماذج متاحة...")
        extra = [m for m in discover_models(key) if m not in GEMINI_PREFERRED]
        return _try_models(key, extra[:5], prompt)


def is_emoji(s: str) -> bool:
    return 0 < len(s) <= 12 and any(ord(c) > 0x2000 for c in s) and not s.isascii()


def propose_candidates(used):
    """يطلب من Gemini كلمات جديدة (موضوع وحرف عشوائيان لتنويع النتائج)."""
    category = random.choice(CATEGORIES)
    letter = random.choice(LETTERS)
    recent = random.sample(used, min(80, len(used)))
    data = gemini_json(
        f"""اقترح 15 كلمة عربية فصيحة بسيطة ومعروفة، من فئة: {category}،
ويفضّل أن تبدأ بحرف "{letter}" (إن لم تكفِ فمن حروف أخرى).
لكل كلمة:
- "word": الكلمة دون تشكيل ودون "ال" التعريف، كلمة واحدة أو كلمتان.
- "emoji": إيموجي واحد فقط هو الأنسب لتمثيل معناها كملصق.
- "explanation": جملة عربية واحدة بسيطة وواضحة (10 إلى 18 كلمة) تشرح المعنى لطفل أو مبتدئ، دون تشكيل ودون أن تبدأ بالكلمة نفسها.
لا تذكر أياً من هذه الكلمات: {', '.join(recent) if recent else 'لا يوجد'}
أعد JSON فقط بالشكل: {{"items": [{{"word": "", "emoji": "", "explanation": ""}}]}}"""
    )
    return data["items"]


def pick_new_word(used):
    used_norm = {norm(w) for w in used}
    for attempt in range(5):
        try:
            items = propose_candidates(used)
        except Exception as e:  # noqa: BLE001
            print("خطأ في الاقتراح:", e)
            continue
        random.shuffle(items)
        for it in items:
            word = str(it.get("word", "")).strip()
            word = _TASHKEEL.sub("", word)
            emoji = str(it.get("emoji", "")).strip()
            expl = str(it.get("explanation", "")).strip()
            if not _ARABIC_ONLY.match(word) or not is_emoji(emoji) or len(expl) < 8:
                continue
            if norm(word) in used_norm:
                continue
            return {"word": word, "emoji": emoji, "explanation": expl}
        print(f"المحاولة {attempt + 1}: كل المقترحات مكررة أو غير صالحة")
    raise RuntimeError("لم نجد كلمة جديدة")


# ---------------------------------------------------------------- الملصق (إيموجي)
def emoji_image(e: str) -> Image.Image:
    no_vs = "_".join(f"{ord(c):x}" for c in e if ord(c) != 0xFE0F)
    with_vs = "_".join(f"{ord(c):x}" for c in e)
    for name in dict.fromkeys([no_vs, with_vs]):
        url = ("https://raw.githubusercontent.com/googlefonts/noto-emoji/main/"
               f"png/512/emoji_u{name}.png")
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return Image.open(io.BytesIO(r.content)).convert("RGBA")
        except requests.RequestException:
            pass
    # بديل: الخط المحلي
    for p in glob.glob("/usr/share/fonts/**/NotoColorEmoji.ttf", recursive=True):
        font = ImageFont.truetype(p, 109)
        canvas = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(canvas).text((10, 10), e, font=font, embedded_color=True)
        bbox = canvas.getbbox()
        if bbox:
            return canvas.crop(bbox).resize((512, 512), Image.LANCZOS)
    raise RuntimeError(f"تعذّر الحصول على الإيموجي {e}")


def add_sticker(canvas: Image.Image, emoji_img: Image.Image, cy: int, circle: int, size: int):
    """دائرة بيضاء مع ظل خفيف والإيموجي في وسطها (شكل ملصق)."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, r = W // 2, circle // 2
    d.ellipse((cx - r, cy - r + 16, cx + r, cy + r + 16), fill=(0, 0, 0, 70))
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, 240))
    em = emoji_img.resize((size, size), Image.LANCZOS)
    layer.alpha_composite(em, (cx - size // 2, cy - size // 2))
    return Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")


# ---------------------------------------------------------------- الرسم
FONT_PATH = None


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
        draw.text((W / 2, top + i * lh), shape(line), font=font, fill=fill,
                  anchor="mm", stroke_width=stroke, stroke_fill=(0, 0, 0))


def gradient(c1, c2):
    g = Image.linear_gradient("L").resize((W, H))
    return ImageOps.colorize(g, c1, c2)


def make_frames(word, explanation, emoji_img, outdir):
    c1, c2 = random.choice(PALETTES)
    accent = random.choice(ACCENTS)

    # 1) الكلمة
    a = gradient(c1, c2)
    draw_text(a, "كلمة اليوم", 380, 80, accent)
    draw_text(a, word, H // 2, 300, (255, 255, 255), max_lines=2, min_size=120)
    a.save(outdir / "a.png")

    # 2) الملصق مع الكلمة
    b = gradient(c1, c2)
    b = add_sticker(b, emoji_img, cy=800, circle=860, size=580)
    draw_text(b, word, 1500, 240, (255, 255, 255), max_lines=2, stroke=5, min_size=110)
    b.save(outdir / "b.png")

    # 3) الشرح
    c = gradient(c1, c2)
    c = add_sticker(c, emoji_img, cy=430, circle=420, size=290)
    draw_text(c, word, 800, 150, accent, max_lines=1)
    draw_text(c, explanation, 1200, 86, (255, 255, 255), max_w=W - 160, max_lines=7)
    c.save(outdir / "c.png")


# ---------------------------------------------------------------- الصوت والفيديو
async def _edge(text, out):
    await asyncio.wait_for(edge_tts.Communicate(text, VOICE, rate="-8%").save(out), 40)


def tts(text: str, out: str):
    for _ in range(3):
        try:
            asyncio.run(_edge(text, out))
            if os.path.getsize(out) > 1000:
                return
        except Exception as e:  # noqa: BLE001
            log(f"edge-tts: {e!r}")
            time.sleep(2)
    print("استعمال gTTS كبديل")
    from gtts import gTTS

    gTTS(text, lang="ar").save(out)


def duration(path: str) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", path])
    return float(out.strip())


def build_video(word, explanation, emoji_img, out_path):
    tmp = Path(tempfile.mkdtemp())
    make_frames(word, explanation, emoji_img, tmp)
    log("تم رسم الصور، بدء الصوت...")

    tts(word, str(tmp / "w.mp3"))
    tts(explanation, str(tmp / "e.mp3"))
    dw, de = duration(str(tmp / "w.mp3")), duration(str(tmp / "e.mp3"))
    log("تم الصوت، بدء ffmpeg...")

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

    t_a, t_b, t_c = dw + 0.3, gap - 0.3, de + tail
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
    log(f"انتهى ffmpeg، مدة الفيديو ≈ {t_a + t_b + t_c:.1f} ثانية")


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
            "categoryId": "27",
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
    used = load_used()
    out_path = str(ROOT / "out.mp4")

    built = None
    for _ in range(4):
        try:
            log("طلب كلمة من Gemini...")
            item = pick_new_word(used)
            log(f"الكلمة: {item['word']} {item['emoji']}")
            emoji_img = emoji_image(item["emoji"])
            log("تم تنزيل الملصق، بدء تركيب الفيديو...")
            build_video(item["word"], item["explanation"], emoji_img, out_path)
            built = item
            break
        except Exception as e:  # noqa: BLE001
            print("فشلت المحاولة:", e)
    if not built:
        sys.exit("فشلت كل المحاولات")

    word = built["word"]
    title = f"{word} {built['emoji']} | معنى كلمة {word} #shorts"
    desc = (
        f"{word}: {built['explanation']}\n\n"
        "كلمة عربية جديدة كل يوم بالصورة والشرح البسيط.\n\n"
        "#shorts #تعلم_العربية #كلمة_اليوم #اللغة_العربية"
    )
    tags = ["shorts", "تعلم العربية", "معنى كلمة", word, "اللغة العربية", "كلمة اليوم"]
    log("بدء الرفع إلى يوتيوب...")
    vid = upload_youtube(out_path, title, desc, tags)
    log("تم الرفع: https://youtu.be/" + vid)

    # تسجيل الكلمة فوراً بعد نجاح الرفع
    used.append(word)
    save_used(used)


if __name__ == "__main__":
    main()
