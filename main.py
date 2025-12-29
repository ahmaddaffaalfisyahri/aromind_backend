from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import os
import base64
import io

# --- untuk Groq Vision AI OCR ---
from groq import Groq

# Konfigurasi Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI(
    title="Aromind AI API",
    description="API rekomendasi parfum Aromind AI",
    version="0.1.0"
)

# CORS supaya bisa diakses dari Flutter Web / mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # DEV: izinkan semua origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load dataset sekali di awal ----
DF = pd.read_csv("data/Parfume Lokal Indonesian.csv")


# ---- Pydantic models ----
class RecommendRequest(BaseModel):
    age: int
    activity: str           # "kerja", "kuliah", "date", dll
    weather: str            # "panas", "hujan", "mendung"
    budget_min: Optional[float] = 0
    budget_max: Optional[float] = 999999999
    preference: Optional[str] = None   # "fresh", "woody", dll


class PerfumeOut(BaseModel):
    perfume: str
    brand: str
    price: Optional[float] = None
    notes: Optional[str] = None
    score: float


class RecommendResponse(BaseModel):
    total_results: int
    recommendations: List[PerfumeOut]


class RecognizeResponse(BaseModel):
    recognized_text: str
    matched: Optional[PerfumeOut] = None
    debug_score: Optional[float] = None  # Debug: skor tertinggi


# ---- Endpoints ----
@app.get("/")
async def root():
    return {"message": "Halo dari Aromind AI 🚀", "version": "1.3"}


# Debug endpoint untuk test Groq langsung
@app.post("/debug-ocr")
async def debug_ocr(file: UploadFile = File(...)):
    """Debug endpoint - lihat raw response dari Groq Vision"""
    content = await file.read()
    image_base64 = base64.b64encode(content).decode("utf-8")
    mime_type = file.content_type or "image/jpeg"
    
    if not groq_client:
        return {"error": "GROQ_API_KEY tidak dikonfigurasi"}
    
    try:
        prompt = """Baca semua teks yang terlihat di gambar parfum ini.
Tuliskan semua teks yang kamu lihat, satu per baris.
Fokus pada: nama parfum, nama brand, ukuran (ml), dan teks lainnya.
Jangan tambahkan penjelasan, cukup tulis teks yang terlihat."""

        response = groq_client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024
        )
        
        text = response.choices[0].message.content if response.choices else "(kosong)"
        
        # Cari match dari database
        text_lower = text.lower()
        matches_found = []
        for _, row in DF.iterrows():
            name = str(row.get("perfume", "")).lower().strip()
            brand = str(row.get("brand", "")).lower().strip()
            if name and len(name) >= 4 and name in text_lower:
                matches_found.append({"type": "perfume", "value": name})
            if brand and len(brand) >= 3 and brand in text_lower:
                matches_found.append({"type": "brand", "value": brand})
        
        return {
            "groq_response": text,
            "text_lowercase": text_lower,
            "matches_found": matches_found[:10]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/sample")
async def sample():
    sample_df = DF.head(5)
    return {
        "count": len(sample_df),
        "items": sample_df.to_dict(orient="records"),
    }


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(payload: RecommendRequest):
    age = payload.age
    activity = payload.activity.lower()
    weather = payload.weather.lower()
    budget_min = payload.budget_min or 0
    budget_max = payload.budget_max or 999999999
    preference = (payload.preference or "").lower()

    # ---- Tentukan tag aroma target dari cuaca & aktivitas ----
    target_tags: List[str] = []

    # Cuaca
    if weather in ["panas", "terik"]:
        target_tags += ["citrus", "fresh", "aquatic"]
    elif weather in ["hujan", "mendung"]:
        target_tags += ["woody", "warm", "amber", "sweet"]

    # Aktivitas
    if activity in ["kerja", "kantor", "kuliah"]:
        target_tags += ["clean", "fresh", "soft"]
    elif activity in ["date", "malam", "dinner"]:
        target_tags += ["sweet", "oriental", "woody", "amber"]

    # Preferensi user
    if preference:
        target_tags.append(preference)

    # Hilangkan duplikat
    target_tags = list(dict.fromkeys(target_tags))

    results = []

    for _, row in DF.iterrows():
        score = 0.0

        # gabungkan semua notes agar mudah dicari
        notes = " ".join([
            str(row.get("top notes", "")),
            str(row.get("mid notes", "")),
            str(row.get("base notes", "")),
        ]).lower()

        # kesesuaian tag aroma
        for tag in target_tags:
            if tag and tag in notes:
                score += 2

        # budget
        price_val = row.get("price", None)
        try:
            price = float(price_val)
        except (TypeError, ValueError):
            price = None

        if price is not None:
            if budget_min <= price <= budget_max:
                score += 1
            # kalau harga terlalu jauh di atas budget max, sedikit penalti
            if price > budget_max * 1.5:
                score -= 1

        if score <= 0:
            continue

        results.append({
            "perfume": row.get("perfume", ""),
            "brand": row.get("brand", ""),
            "price": price,
            "notes": notes,
            "score": score,
        })

    # urutkan dari skor tertinggi
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    # siapkan output
    recs = [PerfumeOut(**r) for r in results[:10]]

    return RecommendResponse(
        total_results=len(results),
        recommendations=recs
    )


# ---- Helper: Fuzzy/Similarity Search ----
def calculate_similarity(str1: str, str2: str) -> float:
    """Hitung similarity antara 2 string (0.0 - 1.0)"""
    if not str1 or not str2:
        return 0.0
    
    str1, str2 = str1.lower(), str2.lower()
    
    # Exact match
    if str1 == str2:
        return 1.0
    
    # Contains match
    if str1 in str2 or str2 in str1:
        return 0.8
    
    # Word overlap
    words1 = set(str1.split())
    words2 = set(str2.split())
    if words1 and words2:
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        return overlap / total if total > 0 else 0.0
    
    return 0.0


def extract_keywords_from_text(text: str) -> dict:
    """Ekstrak keyword dari hasil OCR Gemini"""
    import re
    
    result = {
        "nama": None,
        "brand": None,
        "keywords": [],
        "raw_text": text
    }
    
    lines = text.strip().split('\n')
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Cari format NAMA: xxx atau BRAND: xxx
        if 'nama:' in line_lower:
            result["nama"] = line.split(':', 1)[-1].strip()
        elif 'brand:' in line_lower or 'merek:' in line_lower:
            result["brand"] = line.split(':', 1)[-1].strip()
    
    # Ekstrak semua kata penting sebagai keywords
    text_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    words = text_clean.split()
    
    # Filter kata umum
    stopwords = {'the', 'of', 'for', 'and', 'eau', 'de', 'parfum', 'edp', 'edt', 
                 'ml', 'nama', 'brand', 'merek', 'ukuran', 'jika', 'tidak', 'ada',
                 'yang', 'ini', 'itu', 'dengan', 'untuk', 'atau', 'dan'}
    
    result["keywords"] = [w for w in words if len(w) >= 3 and w not in stopwords]
    
    return result


def search_perfume_by_keywords(keywords_data: dict, df) -> tuple:
    """Search parfum berdasarkan keywords - return (best_row, best_score, match_reason)"""
    
    best_row = None
    best_score = 0
    match_reason = ""
    
    nama = (keywords_data.get("nama", "") or "").lower().strip()
    brand = (keywords_data.get("brand", "") or "").lower().strip()
    keywords = keywords_data.get("keywords", [])
    raw_text = keywords_data.get("raw_text", "").lower()
    
    # Validasi - jika tidak ada data dari OCR, langsung return
    if not nama and not brand and not keywords:
        return None, 0, "No keywords extracted"
    
    for _, row in df.iterrows():
        db_name = str(row.get("perfume", "")).strip()
        db_brand = str(row.get("brand", "")).strip()
        db_name_lower = db_name.lower()
        db_brand_lower = db_brand.lower()
        
        score = 0
        reason = []
        
        # === PRIORITAS 1: EXACT MATCH nama parfum ===
        # Nama harus cukup panjang (min 5 char) dan match persis
        if nama and len(nama) >= 5:
            if nama == db_name_lower:
                score += 100  # Exact match = pasti ini
                reason.append(f"Exact name: {db_name}")
            elif nama in db_name_lower or db_name_lower in nama:
                # Partial match - tapi harus signifikan
                if len(nama) >= 6 or len(db_name_lower) >= 6:
                    score += 60
                    reason.append(f"Partial name: {db_name}")
        
        # === PRIORITAS 2: EXACT MATCH brand ===
        if brand and len(brand) >= 3:
            if brand == db_brand_lower:
                score += 50  # Brand exact match
                reason.append(f"Exact brand: {db_brand}")
            elif brand in db_brand_lower or db_brand_lower in brand:
                score += 25
                reason.append(f"Partial brand: {db_brand}")
        
        # === PRIORITAS 3: Full name found in raw text ===
        # Hanya match jika nama parfum LENGKAP ditemukan (bukan parsial)
        if len(db_name_lower) >= 6 and db_name_lower in raw_text:
            score += 80
            reason.append(f"Full name in text: {db_name}")
        
        # === PRIORITAS 4: Brand found in raw text ===
        if len(db_brand_lower) >= 4 and db_brand_lower in raw_text:
            score += 40
            reason.append(f"Brand in text: {db_brand}")
        
        # === PRIORITAS 5: Significant keyword matching ===
        # Hanya match keyword yang panjang dan unik
        matched_keywords = []
        for kw in keywords:
            if len(kw) >= 5:  # Keyword harus minimal 5 karakter
                if kw in db_name_lower:
                    score += 15
                    matched_keywords.append(kw)
                elif kw in db_brand_lower:
                    score += 10
                    matched_keywords.append(kw)
        
        if matched_keywords:
            reason.append(f"Keywords: {matched_keywords}")
        
        if score > best_score:
            best_score = score
            best_row = row
            match_reason = " | ".join(reason)
    
    return best_row, best_score, match_reason


# ---- Endpoint OCR gambar parfum dengan Groq Vision AI ----
@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(file: UploadFile = File(...)):
    """
    OCR parfum dengan kombinasi approach:
    1. Groq Vision AI ekstrak keyword dari gambar
    2. Fuzzy search di database parfum
    """
    
    # 1. Baca file gambar dan konversi ke base64
    content = await file.read()
    image_base64 = base64.b64encode(content).decode("utf-8")
    mime_type = file.content_type or "image/jpeg"
    
    # 2. Jalankan OCR dengan Groq Vision AI
    if not groq_client:
        return RecognizeResponse(
            recognized_text="Error: GROQ_API_KEY tidak dikonfigurasi",
            matched=None,
        )
    
    try:
        # Prompt yang lebih spesifik untuk ekstrak keyword
        prompt = """Analisis gambar parfum/cologne ini. Ekstrak informasi berikut:

NAMA: [tulis nama produk parfum yang terlihat, biasanya teks paling besar/menonjol]
BRAND: [tulis nama brand/merek pembuat parfum]

Panduan:
- Untuk parfum Indonesia, brand umum: HMNS, Mykonos, Jayrosse, Evangeline, Brasov, Casablanca, Gatsby, Axe, Implora, Wardah, dll.
- Baca SEMUA teks yang terlihat pada kemasan
- Jika ragu, tulis semua teks yang terbaca

Contoh output:
NAMA: Bukan Parfum Biasa
BRAND: HMNS"""

        response = groq_client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024
        )
        
        text = response.choices[0].message.content if response.choices else ""
    except Exception as e:
        text = f"Error OCR: {str(e)}"
    
    # 3. Ekstrak keywords dari hasil OCR
    keywords_data = extract_keywords_from_text(text)
    
    # 4. Search di database dengan fuzzy matching
    best_row, best_score, match_reason = search_perfume_by_keywords(keywords_data, DF)
    
    # 5. Build response
    # THRESHOLD = 40 (harus ada match yang cukup kuat)
    matched = None
    if best_row is not None and best_score >= 40:
        price_val = best_row.get("price", None)
        try:
            price = float(price_val) if price_val is not None else None
        except (TypeError, ValueError):
            price = None

        notes = " ".join([
            str(best_row.get("top notes", "")),
            str(best_row.get("mid notes", "")),
            str(best_row.get("base notes", "")),
        ]).strip()

        matched = PerfumeOut(
            perfume=best_row.get("perfume", ""),
            brand=best_row.get("brand", ""),
            price=price,
            notes=notes,
            score=float(best_score),
        )

    # Tambahkan debug info di recognized_text jika tidak match
    debug_info = text
    if not matched and best_score > 0:
        debug_info += f"\n\n[DEBUG] Best score: {best_score}, Reason: {match_reason}"

    return RecognizeResponse(
        recognized_text=debug_info,
        matched=matched,
        debug_score=float(best_score),
    )

