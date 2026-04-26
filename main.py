from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import httpx
from collections import Counter
import re
import os
import json
from datetime import datetime
from typing import Optional

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io

app = FastAPI(title="MenuPulse API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERPAPI_KEY    = os.environ.get("SERPAPI_KEY", "")
OUTSCRAPER_KEY = os.environ.get("OUTSCRAPER_KEY", "")
ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_KEY", "")

# ─── Review Fetching ──────────────────────────────────────────────────────────

def fetch_outscraper_reviews(address: str, reviews_limit: int = 100):
    print(f"Outscraper fetching: {address}")
    url = "https://api.app.outscraper.com/maps/reviews-v3"
    params = {
        "query": address,
        "reviewsLimit": reviews_limit,
        "language": "en",
        "async": False
    }
    headers = {"X-API-KEY": OUTSCRAPER_KEY}
    try:
        res = requests.get(url, params=params, headers=headers, timeout=60)
        data = res.json()
        if res.status_code != 200:
            print(f"Outscraper error: {data}")
            return [], address
        reviews = []
        business_name = None
        for place in data.get("data", []):
            if not business_name:
                business_name = place.get("name", address)
            for review in place.get("reviews_data", []):
                text = review.get("review_text", "")
                rating = review.get("review_rating", None)
                if text and len(text) > 40:
                    reviews.append({"text": text, "rating": rating, "source": "google_maps"})
        print(f"Outscraper: {len(reviews)} reviews for {address}")
        return reviews, business_name
    except Exception as e:
        print(f"Outscraper error: {e}")
        return [], address


def fetch_reviews_for_location(address: str):
    reviews, business_name = fetch_outscraper_reviews(address)
    seen = set()
    unique = []
    for r in reviews:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique.append(r)
    return unique, business_name or address


# ─── Sentiment — Claude-based (no PyTorch needed) ────────────────────────────

def analyze_sentiment_from_ratings(reviews: list) -> dict:
    """Score sentiment from star ratings alone — fast and reliable."""
    if not reviews:
        return {"overall": "Unknown", "score": 0}
    scores = []
    for r in reviews:
        rating = r.get("rating")
        if rating:
            scores.append(2 if rating >= 4 else (1 if rating == 3 else 0))
    if not scores:
        return {"overall": "Unknown", "score": 5.0}
    avg = sum(scores) / len(scores)
    score_out_of_10 = round((avg / 2) * 10, 1)
    if avg >= 1.5:
        overall = "Positive"
    elif avg <= 0.5:
        overall = "Negative"
    else:
        overall = "Mixed"
    return {"overall": overall, "score": score_out_of_10}


def analyze_location(address: str) -> dict:
    reviews, business_name = fetch_reviews_for_location(address)
    if not reviews:
        return {
            "address": address,
            "business_name": business_name,
            "count": 0,
            "overall": "No data",
            "score": 0,
            "reviews": []
        }
    sentiment = analyze_sentiment_from_ratings(reviews)
    return {
        "address": address,
        "business_name": business_name,
        "count": len(reviews),
        "overall": sentiment["overall"],
        "score": sentiment["score"],
        "reviews": [r["text"] for r in reviews]  # all reviews sent to Claude in frontend
    }


# ─── PDF Generation ───────────────────────────────────────────────────────────

def generate_location_pdf(location_data: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                  fontSize=22, spaceAfter=6, textColor=colors.HexColor("#111111"))
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
                                     fontSize=11, textColor=colors.HexColor("#666666"), spaceAfter=20)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"],
                                    fontSize=13, textColor=colors.HexColor("#111111"),
                                    spaceBefore=16, spaceAfter=8)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=10, leading=16, textColor=colors.HexColor("#333333"))
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"],
                                   fontSize=8, textColor=colors.HexColor("#aaaaaa"),
                                   alignment=TA_CENTER)
    score_color = colors.HexColor(
        "#16a34a" if location_data.get("overall") == "Positive"
        else "#d97706" if location_data.get("overall") == "Mixed"
        else "#dc2626"
    )
    story = []
    story.append(Paragraph(location_data.get("business_name", ""), title_style))
    story.append(Paragraph(location_data.get("address", ""), subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#eeeeee")))
    story.append(Spacer(1, 16))

    score_data = [
        ["Overall Sentiment", "Score", "Reviews Analyzed"],
        [location_data.get("overall",""), f"{location_data.get('score',0)}/10", str(location_data.get("count",0))]
    ]
    score_table = Table(score_data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f5f5f5")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#888888")),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("BACKGROUND", (0,1), (-1,1), colors.white),
        ("TEXTCOLOR", (0,1), (0,1), score_color),
        ("TEXTCOLOR", (1,1), (-1,1), colors.HexColor("#111111")),
        ("FONTSIZE", (0,1), (-1,1), 18),
        ("FONTNAME", (0,1), (-1,1), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#eeeeee")),
        ("INNERGRID", (0,0), (-1,-1), 0.5, colors.HexColor("#eeeeee")),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 20))
    story.append(Paragraph("Sample Reviews", heading_style))
    for i, review in enumerate(location_data.get("reviews", [])[:10]):
        story.append(Paragraph(f"{i+1}. {review}", body_style))
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#eeeeee")))
        story.append(Spacer(1, 8))
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#eeeeee")))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y')} | MenuPulse", footer_style))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ─── Routes ───────────────────────────────────────────────────────────────────

class SingleLocationRequest(BaseModel):
    address: str
    reviews_limit: int = 100

class MultiLocationRequest(BaseModel):
    chain_name: str
    addresses: list
    reviews_limit: int = 100

@app.get("/")
def root():
    return {
        "service": "MenuPulse API",
        "version": "2.0.0",
        "status": "running"
    }

@app.post("/analyze/single")
def analyze_single(request: SingleLocationRequest):
    result = analyze_location(request.address)
    if result["count"] == 0:
        raise HTTPException(status_code=404, detail="No reviews found for this address.")
    return result

@app.post("/analyze/multi")
def analyze_multi(request: MultiLocationRequest):
    if not 2 <= len(request.addresses) <= 20:
        raise HTTPException(status_code=400, detail="Please provide between 2 and 20 addresses.")
    locations = [analyze_location(a.strip()) for a in request.addresses]
    valid = [l for l in locations if l["score"] > 0]
    avg_score = round(sum(l["score"] for l in valid) / len(valid), 1) if valid else 0
    best  = max(valid, key=lambda x: x["score"]) if valid else None
    worst = min(valid, key=lambda x: x["score"]) if valid else None
    return {
        "chain_name": request.chain_name,
        "locations": locations,
        "summary": {
            "total_locations": len(locations),
            "avg_score": avg_score,
            "total_reviews": sum(l["count"] for l in locations),
            "best_location":  best["business_name"]  if best  else None,
            "worst_location": worst["business_name"] if worst else None,
        }
    }

@app.post("/report/single")
def report_single(request: SingleLocationRequest):
    result = analyze_location(request.address)
    if result["count"] == 0:
        raise HTTPException(status_code=404, detail="No reviews found.")
    pdf_bytes = generate_location_pdf(result)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=menupulse_report.pdf"}
    )

@app.get("/model/status")
def model_status():
    return {"status": "Claude-based analysis — no local model required"}


# ─── Anthropic Proxy ──────────────────────────────────────────────────────────

class ClaudeRequest(BaseModel):
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2000
    system: str = ""
    messages: list

@app.post("/claude")
async def claude_proxy(request: ClaudeRequest):
    if not ANTHROPIC_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured on server.")

    safe_tokens = min(request.max_tokens, 8000)

    payload = {
        "model": request.model,
        "max_tokens": safe_tokens,
        "messages": request.messages,
    }
    if request.system:
        payload["system"] = request.system

    headers = {
        "x-api-key": ANTHROPIC_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers
            )
        return resp.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to Claude timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
