from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import os
import base64
import io

# --- untuk Gemini AI OCR ---
import google.generativeai as genai

# Konfigurasi Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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


# Debug endpoint untuk test Gemini langsung
@app.post("/debug-ocr")
async def debug_ocr(file: UploadFile = File(...)):
    """Debug endpoint - lihat raw response dari Gemini"""
    content = await file.read()
    image_base64 = base64.b64encode(content).decode("utf-8")
    mime_type = file.content_type or "image/jpeg"
    
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY tidak dikonfigurasi"}
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = """Baca semua teks yang terlihat di gambar parfum ini.
Tuliskan semua teks yang kamu lihat, satu per baris.
Fokus pada: nama parfum, nama brand, ukuran (ml), dan teks lainnya.
Jangan tambahkan penjelasan, cukup tulis teks yang terlihat."""

        response = model.generate_content([
            prompt,
            {"mime_type": mime_type, "data": image_base64}
        ])
        
        text = response.text if response.text else "(kosong)"
        
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
            "gemini_response": text,
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


# ---- Endpoint OCR gambar parfum dengan Gemini AI ----
@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(file: UploadFile = File(...)):
    import re
    
    # 1. Baca file gambar dan konversi ke base64
    content = await file.read()
    image_base64 = base64.b64encode(content).decode("utf-8")
    
    # Tentukan mime type
    mime_type = file.content_type or "image/jpeg"
    
    # 2. Jalankan OCR dengan Gemini AI
    if not GEMINI_API_KEY:
        return RecognizeResponse(
            recognized_text="Error: GEMINI_API_KEY tidak dikonfigurasi",
            matched=None,
        )
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Prompt yang lebih spesifik untuk parfum
        prompt = """Lihat gambar parfum ini dengan teliti.
Identifikasi dan tulis:
1. Nama parfum (biasanya huruf besar/menonjol)
2. Nama brand/merek
3. Ukuran ml jika ada

Tulis hasilnya dalam format:
NAMA: [nama parfum]
BRAND: [nama brand]

Jika tidak bisa membaca, tulis teks apapun yang terlihat."""

        response = model.generate_content([
            prompt,
            {
                "mime_type": mime_type,
                "data": image_base64
            }
        ])
        
        text = response.text if response.text else ""
    except Exception as e:
        text = f"Error OCR: {str(e)}"
    
    text_lower = text.lower()
    
    # Bersihkan teks - hapus karakter khusus
    text_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text_lower)
    
    # Pecah teks menjadi kata-kata untuk matching
    text_words = set(text_clean.split())

    best_row = None
    best_score = 0

    # Cari parfum yang cocok dengan pendekatan yang lebih toleran
    for _, row in DF.iterrows():
        name = str(row.get("perfume", "")).lower().strip()
        brand = str(row.get("brand", "")).lower().strip()

        score = 0
        
        # === Strategi 1: Exact match nama parfum ===
        if name and len(name) >= 3:
            if name in text_lower or name in text_clean:
                score += 10  # Bonus besar untuk exact match
                
        # === Strategi 2: Partial word match untuk nama parfum ===
        if name and len(name) >= 3:
            name_words = name.split()
            for word in name_words:
                if len(word) >= 3:
                    # Cek apakah kata ada di teks
                    if word in text_lower or word in text_words:
                        score += 3
                    # Cek partial match (kata mengandung atau terkandung)
                    for text_word in text_words:
                        if len(text_word) >= 3:
                            if word in text_word or text_word in word:
                                score += 2
                                break
                
        # === Strategi 3: Brand matching ===
        if brand and len(brand) >= 3:
            if brand in text_lower or brand in text_clean:
                score += 5  # Brand match
            # Partial brand match
            brand_words = brand.split()
            for word in brand_words:
                if len(word) >= 3 and word in text_words:
                    score += 2

        if score > best_score:
            best_score = score
            best_row = row

    matched = None
    # Threshold = 3 (diturunkan agar lebih mudah match)
    if best_row is not None and best_score >= 3:
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

    return RecognizeResponse(
        recognized_text=text,
        matched=matched,
        debug_score=float(best_score),
    )

