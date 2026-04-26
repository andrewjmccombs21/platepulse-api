from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io

app = FastAPI(title="Plate Pulse API", version="1.0.0")

# CORS — allow requests from any origin (browser-based SaaS tool)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://menupulse-beta.vercel.app",
        "https://menupulse.vercel.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "null",  # local file:// access
        "*"      # keep open during beta
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files — only mount if directory exists (not needed on Railway)
import os as _os
if _os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "your-serpapi-key-here")
OUTSCRAPER_KEY = os.environ.get("OUTSCRAPER_KEY", "your-outscraper-key-here")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_KEY", "")

MODEL_DIR = "model"
TRAINING_FILE = "training_data.json"

# ─── Model Setup ─────────────────────────────────────────────────────────────

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def load_model():
    if os.path.exists(MODEL_DIR):
        print("Loading fine-tuned local model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            print("Fine-tuned model loaded.")
            return tokenizer, model
        except Exception as e:
            print(f"Could not load fine-tuned model ({e}), falling back to pretrained.")
    print("Loading pretrained model from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    print("Pretrained model loaded.")
    return tokenizer, model


# Lazy load model — only when first review request comes in
tokenizer, model, sentiment_pipeline = None, None, None

def get_pipeline():
    global tokenizer, model, sentiment_pipeline
    if sentiment_pipeline is None:
        try:
            print("Loading sentiment model...")
            tokenizer, model = load_model()
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                truncation=True,
                max_length=512
            )
            print("Model ready!")
        except Exception as e:
            print(f"Model load failed: {e} — sentiment analysis unavailable")
            sentiment_pipeline = None
    return sentiment_pipeline

# ─── Dataset & Training ───────────────────────────────────────────────────────

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def retrain_model():
    if not os.path.exists(TRAINING_FILE):
        return
    with open(TRAINING_FILE, "r") as f:
        data = json.load(f)
    if len(data) < 10:
        return

    print(f"Retraining on {len(data)} samples...")
    label_map = {"Negative": 0, "Mixed": 1, "Positive": 2}
    texts = [d["text"] for d in data]
    labels = [label_map.get(d["label"], 1) for d in data]

    dataset = ReviewDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} — Loss: {total_loss:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Model saved!")

    global sentiment_pipeline
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512
        )
    except Exception as e:
        print(f"Pipeline reload failed: {e}")

# ─── Review Fetching ──────────────────────────────────────────────────────────

def fetch_outscraper_reviews(address: str, reviews_limit: int = 100) -> list[dict]:
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
            return []

        reviews = []
        business_name = None
        for place in data.get("data", []):
            if not business_name:
                business_name = place.get("name", address)
            for review in place.get("reviews_data", []):
                text = review.get("review_text", "")
                rating = review.get("review_rating", None)
                if text and len(text) > 40:
                    reviews.append({
                        "text": text,
                        "rating": rating,
                        "source": "google_maps"
                    })

        print(f"Outscraper: {len(reviews)} reviews for {address}")
        return reviews, business_name

    except Exception as e:
        print(f"Outscraper error: {e}")
        return [], address


def fetch_serpapi_reviews(query: str) -> list[dict]:
    reviews = []
    url = "https://serpapi.com/search"
    for start in range(0, 30, 10):
        params = {
            "api_key": SERPAPI_KEY,
            "engine": "google",
            "q": f"{query} restaurant customer reviews",
            "num": 10,
            "start": start
        }
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            if "error" in data:
                break
            results = data.get("organic_results", [])
            if not results:
                break
            for result in results:
                snippet = result.get("snippet", "")
                if len(snippet) > 40:
                    reviews.append({"text": snippet, "rating": None, "source": "serp"})
        except Exception as e:
            print(f"SerpApi error: {e}")
            break
    return reviews


def fetch_reviews_for_location(address: str) -> tuple[list[dict], str]:
    reviews, business_name = fetch_outscraper_reviews(address)
    if len(reviews) < 5:
        print(f"Falling back to SerpApi for {address}")
        reviews.extend(fetch_serpapi_reviews(address))

    seen = set()
    unique = []
    for r in reviews:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique.append(r)
    return unique, business_name or address

# ─── Analysis ─────────────────────────────────────────────────────────────────

def analyze_sentiment(reviews: list[dict]) -> dict:
    if not reviews:
        return {"overall": "Unknown", "score": 0}

    scores = []
    for review in reviews:
        rating = review.get("rating")
        if rating:
            scores.append(2 if rating >= 4 else (1 if rating == 3 else 0))
        else:
            try:
                pipe = get_pipeline()
                if pipe is None:
                    scores.append(1)
                    continue
                result = pipe(review["text"][:512])[0]
                label = result["label"]
                if "positive" in label.lower() or label == "LABEL_2":
                    scores.append(2)
                elif "negative" in label.lower() or label == "LABEL_0":
                    scores.append(0)
                else:
                    scores.append(1)
            except Exception:
                scores.append(1)

    avg = sum(scores) / len(scores)
    score_out_of_10 = round((avg / 2) * 10, 1)

    if avg >= 1.5:
        overall = "Positive"
    elif avg <= 0.5:
        overall = "Negative"
    else:
        overall = "Mixed"

    return {"overall": overall, "score": score_out_of_10}


def extract_themes(reviews: list[dict], polarity: str) -> list[str]:
    positive_keywords = [
        "food", "service", "staff", "atmosphere", "price", "portions",
        "flavour", "fresh", "friendly", "quick", "clean", "cozy", "delicious"
    ]
    negative_keywords = [
        "slow", "cold", "rude", "expensive", "wait", "dirty", "small",
        "bland", "noisy", "overpriced", "disappointing", "undercooked"
    ]
    keywords = positive_keywords if polarity == "Positive" else negative_keywords
    text = " ".join([r["text"] for r in reviews]).lower()
    counts = Counter()
    for word in keywords:
        count = len(re.findall(r'\b' + word + r'\b', text))
        if count > 0:
            counts[word] = count

    labels = {
        "food": "Quality of food praised", "service": "Service quality mentioned",
        "staff": "Friendly staff highlighted", "atmosphere": "Great atmosphere noted",
        "price": "Good value for money", "portions": "Generous portions",
        "flavour": "Strong flavours", "fresh": "Fresh ingredients",
        "friendly": "Friendly and welcoming", "quick": "Fast service",
        "clean": "Clean environment", "cozy": "Cozy and comfortable",
        "delicious": "Delicious food", "slow": "Slow service reported",
        "cold": "Food served cold", "rude": "Rude staff reported",
        "expensive": "Considered overpriced", "wait": "Long wait times",
        "dirty": "Cleanliness concerns", "small": "Small portion sizes",
        "bland": "Food described as bland", "noisy": "Noisy environment",
        "overpriced": "Considered overpriced", "disappointing": "Disappointing experience",
        "undercooked": "Undercooked food reported"
    }
    top = [word for word, _ in counts.most_common(5)]
    return [labels.get(w, w.capitalize()) for w in top] or ["Not enough data"]


def generate_recommendations(sentiment_result: dict, positives: list[str], negatives: list[str]) -> list[str]:
    recs = []
    score = sentiment_result["score"]
    if score < 5:
        recs.append("Urgently review core operations — customer satisfaction is critically low")
    elif score < 7:
        recs.append("Focus on consistency — customers are having mixed experiences")
    else:
        recs.append("Maintain current standards — customers are largely satisfied")

    neg_text = " ".join(negatives).lower()
    if "service" in neg_text or "staff" in neg_text or "rude" in neg_text:
        recs.append("Invest in staff training to improve service quality")
    if "wait" in neg_text or "slow" in neg_text:
        recs.append("Review kitchen and front-of-house workflow to reduce wait times")
    if "price" in neg_text or "expensive" in neg_text or "overpriced" in neg_text:
        recs.append("Consider introducing value meal options or a loyalty programme")
    if "clean" in neg_text or "dirty" in neg_text:
        recs.append("Implement a stricter cleaning schedule and hygiene checks")
    if "cold" in neg_text or "undercooked" in neg_text:
        recs.append("Review food preparation and delivery processes")

    pos_text = " ".join(positives).lower()
    if "food" in pos_text or "delicious" in pos_text:
        recs.append("Leverage strong food quality in marketing and social media")
    if "friendly" in pos_text or "staff" in pos_text:
        recs.append("Highlight excellent customer service in your online profiles")

    return recs[:4] or ["Gather more reviews to generate targeted recommendations"]


def analyze_location(address: str) -> dict:
    reviews, business_name = fetch_reviews_for_location(address)
    if not reviews:
        return {
            "address": address,
            "business_name": business_name,
            "count": 0,
            "overall": "No data",
            "score": 0,
            "positives": [],
            "negatives": [],
            "recommendations": [],
            "reviews": []
        }

    sentiment_result = analyze_sentiment(reviews)
    positives = extract_themes(reviews, "Positive")
    negatives = extract_themes(reviews, "Negative")
    recommendations = generate_recommendations(sentiment_result, positives, negatives)

    return {
        "address": address,
        "business_name": business_name,
        "count": len(reviews),
        "overall": sentiment_result["overall"],
        "score": sentiment_result["score"],
        "positives": positives,
        "negatives": negatives,
        "recommendations": recommendations,
        "reviews": [r["text"] for r in reviews]  # all reviews for Claude analysis
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
    score_color = colors.HexColor(
        "#16a34a" if location_data["overall"] == "Positive"
        else "#d97706" if location_data["overall"] == "Mixed"
        else "#dc2626"
    )

    story = []

    # Header
    story.append(Paragraph(location_data["business_name"], title_style))
    story.append(Paragraph(location_data["address"], subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#eeeeee")))
    story.append(Spacer(1, 16))

    # Score cards
    score_data = [
        ["Overall Sentiment", "Score", "Reviews Analyzed"],
        [
            location_data["overall"],
            f"{location_data['score']}/10",
            str(location_data["count"])
        ]
    ]
    score_table = Table(score_data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#888888")),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
        ("BACKGROUND", (0, 1), (-1, 1), colors.white),
        ("TEXTCOLOR", (0, 1), (0, 1), score_color),
        ("TEXTCOLOR", (1, 1), (-1, 1), colors.HexColor("#111111")),
        ("FONTSIZE", (0, 1), (-1, 1), 18),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 20))

    # Positives and negatives
    theme_data = [["What Customers Love", "Areas to Improve"]]
    pos_text = "\n".join([f"+ {p}" for p in location_data["positives"]]) or "Not enough data"
    neg_text = "\n".join([f"- {n}" for n in location_data["negatives"]]) or "Not enough data"
    theme_data.append([
        Paragraph(pos_text.replace("\n", "<br/>"), body_style),
        Paragraph(neg_text.replace("\n", "<br/>"), body_style)
    ])
    theme_table = Table(theme_data, colWidths=[3.3*inch, 3.3*inch])
    theme_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#888888")),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
        ("BACKGROUND", (0, 1), (-1, 1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(theme_table)
    story.append(Spacer(1, 20))

    # Recommendations
    story.append(Paragraph("Recommendations for the Owner", heading_style))
    for i, rec in enumerate(location_data["recommendations"]):
        story.append(Paragraph(f"{i+1}. {rec}", body_style))
        story.append(Spacer(1, 4))

    # Sample reviews
    story.append(Spacer(1, 8))
    story.append(Paragraph("Sample Reviews", heading_style))
    for i, review in enumerate(location_data["reviews"]):
        story.append(Paragraph(f"{i+1}. {review}", body_style))
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#eeeeee")))
        story.append(Spacer(1, 8))

    # Footer
    story.append(Spacer(1, 16))
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"],
                                   fontSize=8, textColor=colors.HexColor("#aaaaaa"),
                                   alignment=TA_CENTER)
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#eeeeee")))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y')} | Restaurant Review Analyzer", footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def generate_combined_pdf(chain_name: str, locations: list[dict]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                  fontSize=24, spaceAfter=6, textColor=colors.HexColor("#111111"))
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
                                     fontSize=11, textColor=colors.HexColor("#666666"), spaceAfter=20)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"],
                                    fontSize=14, textColor=colors.HexColor("#111111"),
                                    spaceBefore=20, spaceAfter=10)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=10, leading=16, textColor=colors.HexColor("#333333"))
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"],
                                   fontSize=8, textColor=colors.HexColor("#aaaaaa"),
                                   alignment=TA_CENTER)

    story = []

    # Cover page
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph(chain_name, title_style))
    story.append(Paragraph("Multi-Location Sentiment Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#eeeeee")))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y')}", footer_style))
    story.append(Spacer(1, 0.5*inch))

    # Overall summary metrics
    all_reviews = [r for loc in locations for r in loc.get("reviews", [])]
    valid = [l for l in locations if l["score"] > 0]
    avg_score = round(sum(l["score"] for l in valid) / len(valid), 1) if valid else 0
    best = max(valid, key=lambda x: x["score"]) if valid else None
    worst = min(valid, key=lambda x: x["score"]) if valid else None
    total_reviews = sum(l["count"] for l in locations)

    summary_data = [
        ["Locations Analyzed", "Avg Score", "Total Reviews", "Best Location"],
        [
            str(len(locations)),
            f"{avg_score}/10",
            str(total_reviews),
            best["business_name"][:20] if best else "N/A"
        ]
    ]
    summary_table = Table(summary_data, colWidths=[1.65*inch, 1.65*inch, 1.65*inch, 1.65*inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#888888")),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
        ("BACKGROUND", (0, 1), (-1, 1), colors.white),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#111111")),
        ("FONTSIZE", (0, 1), (-1, 1), 16),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 24))

    # Side by side comparison table
    story.append(Paragraph("Location Comparison", heading_style))
    comp_header = ["Location", "Sentiment", "Score", "Reviews"]
    comp_data = [comp_header]
    for loc in locations:
        comp_data.append([
            Paragraph(loc["business_name"][:30], body_style),
            loc["overall"],
            f"{loc['score']}/10",
            str(loc["count"])
        ])

    comp_table = Table(comp_data, colWidths=[3*inch, 1.2*inch, 1*inch, 1.2*inch])
    row_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#888888")),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]
    for i, loc in enumerate(locations, start=1):
        color = (
            colors.HexColor("#f0fdf4") if loc["overall"] == "Positive"
            else colors.HexColor("#fffbeb") if loc["overall"] == "Mixed"
            else colors.HexColor("#fef2f2")
        )
        row_styles.append(("BACKGROUND", (0, i), (-1, i), color))
        text_color = (
            colors.HexColor("#16a34a") if loc["overall"] == "Positive"
            else colors.HexColor("#d97706") if loc["overall"] == "Mixed"
            else colors.HexColor("#dc2626")
        )
        row_styles.append(("TEXTCOLOR", (1, i), (1, i), text_color))

    comp_table.setStyle(TableStyle(row_styles))
    story.append(comp_table)
    story.append(PageBreak())

    # Individual location pages
    for loc in locations:
        story.append(Paragraph(loc["business_name"], title_style))
        story.append(Paragraph(loc["address"], subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#eeeeee")))
        story.append(Spacer(1, 16))

        score_color = colors.HexColor(
            "#16a34a" if loc["overall"] == "Positive"
            else "#d97706" if loc["overall"] == "Mixed"
            else "#dc2626"
        )
        score_data = [
            ["Overall Sentiment", "Score", "Reviews Analyzed"],
            [loc["overall"], f"{loc['score']}/10", str(loc["count"])]
        ]
        score_table = Table(score_data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
        score_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#888888")),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
            ("BACKGROUND", (0, 1), (-1, 1), colors.white),
            ("TEXTCOLOR", (0, 1), (0, 1), score_color),
            ("TEXTCOLOR", (1, 1), (-1, 1), colors.HexColor("#111111")),
            ("FONTSIZE", (0, 1), (-1, 1), 18),
            ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 16))

        pos_text = "\n".join([f"+ {p}" for p in loc["positives"]]) or "Not enough data"
        neg_text = "\n".join([f"- {n}" for n in loc["negatives"]]) or "Not enough data"
        theme_data = [
            ["What Customers Love", "Areas to Improve"],
            [
                Paragraph(pos_text.replace("\n", "<br/>"), body_style),
                Paragraph(neg_text.replace("\n", "<br/>"), body_style)
            ]
        ]
        theme_table = Table(theme_data, colWidths=[3.3*inch, 3.3*inch])
        theme_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#888888")),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
            ("BACKGROUND", (0, 1), (-1, 1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(theme_table)
        story.append(Spacer(1, 16))

        story.append(Paragraph("Recommendations", heading_style))
        for i, rec in enumerate(loc["recommendations"]):
            story.append(Paragraph(f"{i+1}. {rec}", body_style))
            story.append(Spacer(1, 4))

        story.append(PageBreak())

    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# ─── Routes ───────────────────────────────────────────────────────────────────

class MultiLocationRequest(BaseModel):
    chain_name: str
    addresses: list[str]
    reviews_limit: int = 100


class SingleLocationRequest(BaseModel):
    address: str
    reviews_limit: int = 100


class FeedbackRequest(BaseModel):
    review: str
    correct_label: str


@app.get("/")
def root():
    return {
        "service": "Plate Pulse API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "POST /analyze/single",
            "POST /analyze/multi",
            "POST /report/single",
            "GET  /model/status"
        ]
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

    locations = []
    for address in request.addresses:
        result = analyze_location(address.strip())
        locations.append(result)

    valid = [l for l in locations if l["score"] > 0]
    avg_score = round(sum(l["score"] for l in valid) / len(valid), 1) if valid else 0
    best = max(valid, key=lambda x: x["score"]) if valid else None
    worst = min(valid, key=lambda x: x["score"]) if valid else None

    return {
        "chain_name": request.chain_name,
        "locations": locations,
        "summary": {
            "total_locations": len(locations),
            "avg_score": avg_score,
            "total_reviews": sum(l["count"] for l in locations),
            "best_location": best["business_name"] if best else None,
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
        headers={"Content-Disposition": f"attachment; filename=report_{result['business_name'][:20]}.pdf"}
    )


@app.post("/report/multi")
def report_multi(request: MultiLocationRequest):
    if not 2 <= len(request.addresses) <= 20:
        raise HTTPException(status_code=400, detail="Please provide between 2 and 20 addresses.")

    locations = []
    for address in request.addresses:
        result = analyze_location(address.strip())
        locations.append(result)

    pdf_bytes = generate_combined_pdf(request.chain_name, locations)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={request.chain_name}_report.pdf"}
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    if request.correct_label not in ["Positive", "Mixed", "Negative"]:
        raise HTTPException(status_code=400, detail="Label must be Positive, Mixed, or Negative")

    data = []
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "r") as f:
            data = json.load(f)

    data.append({
        "text": request.review,
        "label": request.correct_label,
        "timestamp": datetime.now().isoformat()
    })

    with open(TRAINING_FILE, "w") as f:
        json.dump(data, f, indent=2)

    if len(data) % 10 == 0:
        background_tasks.add_task(retrain_model)
        return {"message": f"Feedback saved! Retraining on {len(data)} samples."}

    return {"message": f"Feedback saved! ({len(data)} samples collected)"}

class NotReviewRequest(BaseModel):
    text: str

@app.post("/feedback/not-review")
def mark_not_review(request: NotReviewRequest):
    not_reviews_file = "not_reviews.json"
    data = []
    if os.path.exists(not_reviews_file):
        with open(not_reviews_file, "r") as f:
            data = json.load(f)

    if request.text not in data:
        data.append(request.text)
        with open(not_reviews_file, "w") as f:
            json.dump(data, f, indent=2)

    return {"message": f"Saved. {len(data)} non-reviews excluded."}



@app.get("/model/status")
def model_status():
    sample_count = 0
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "r") as f:
            sample_count = len(json.load(f))

    excluded_count = 0
    if os.path.exists("not_reviews.json"):
        with open("not_reviews.json", "r") as f:
            excluded_count = len(json.load(f))

    return {
        "fine_tuned": os.path.exists(MODEL_DIR),
        "training_samples": sample_count,
        "next_retrain_at": ((sample_count // 10) + 1) * 10,
        "excluded_count": excluded_count
    }


# ─── Anthropic Proxy ─────────────────────────────────────────────────────────
# Keeps ANTHROPIC_KEY server-side — never exposed to the browser

class ClaudeRequest(BaseModel):
    model: str = "claude-sonnet-4-20250514"
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
