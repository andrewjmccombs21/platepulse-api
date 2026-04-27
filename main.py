from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Header
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import httpx
from collections import Counter
import re
import os
import json
from datetime import datetime, timezone
from typing import Optional, List
import hashlib
import hmac

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io

app = FastAPI(title="MenuPulse API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERPAPI_KEY        = os.environ.get("SERPAPI_KEY", "")
OUTSCRAPER_KEY     = os.environ.get("OUTSCRAPER_KEY", "")
ANTHROPIC_KEY      = os.environ.get("ANTHROPIC_KEY", "")
SUPABASE_URL       = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")  # service role key for server-side writes
EDI_WEBHOOK_SECRET = os.environ.get("EDI_WEBHOOK_SECRET", "menupulse-edi-secret")  # change in Railway vars

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


# ─── Supabase Helper ──────────────────────────────────────────────────────────

def supabase_insert(table: str, rows: list):
    """Server-side Supabase insert using service role key (bypasses RLS)."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("Supabase not configured on server")
        return False
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    res = requests.post(url, json=rows, headers=headers, timeout=10)
    if res.status_code not in (200, 201):
        print(f"Supabase insert error: {res.status_code} {res.text}")
        return False
    return True


def supabase_upsert(table: str, rows: list, on_conflict: str):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return False
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": f"resolution=merge-duplicates,return=minimal"
    }
    params = {"on_conflict": on_conflict}
    res = requests.post(url, json=rows, headers=headers, params=params, timeout=10)
    return res.status_code in (200, 201)


# ─── Toast POS Integration ────────────────────────────────────────────────────

class ToastConnectRequest(BaseModel):
    user_id: str
    client_id: str        # Toast OAuth client ID
    client_secret: str    # Toast OAuth client secret
    restaurant_guid: str  # Toast restaurant GUID

class ToastSyncRequest(BaseModel):
    user_id: str
    access_token: str
    restaurant_guid: str

@app.post("/integrations/toast/connect")
async def toast_connect(req: ToastConnectRequest):
    """Exchange Toast credentials for an access token and verify connection."""
    try:
        # Toast OAuth token endpoint
        token_url = "https://ws-api.toasttab.com/authentication/v1/authentication/login"
        payload = {
            "clientId": req.client_id,
            "clientSecret": req.client_secret,
            "userAccessType": "TOAST_MACHINE_CLIENT"
        }
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(token_url, json=payload)

        if res.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Toast auth failed: {res.text}")

        data = res.json()
        token = data.get("token", {}).get("accessToken")
        if not token:
            raise HTTPException(status_code=400, detail="No access token in Toast response")

        # Store connection in Supabase integrations table
        supabase_upsert("integrations", [{
            "user_id": req.user_id,
            "provider": "toast",
            "restaurant_guid": req.restaurant_guid,
            "access_token": token,  # In production: encrypt this
            "connected_at": datetime.now(timezone.utc).isoformat(),
            "status": "active"
        }], on_conflict="user_id,provider")

        return {"success": True, "message": "Toast connected successfully", "token": token}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/integrations/toast/sync")
async def toast_sync(req: ToastSyncRequest):
    """Pull menu items and recent sales from Toast."""
    headers = {
        "Authorization": f"Bearer {req.access_token}",
        "Toast-Restaurant-External-ID": req.restaurant_guid
    }
    base = "https://ws-api.toasttab.com"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Fetch menu items
            menu_res = await client.get(f"{base}/menus/v2/menus", headers=headers)
            # Fetch recent orders (last 30 days)
            from_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00.000+0000")
            orders_res = await client.get(
                f"{base}/orders/v2/ordersBulk",
                headers=headers,
                params={"startDate": from_date, "pageSize": 100}
            )

        menu_data   = menu_res.json()   if menu_res.status_code   == 200 else {}
        orders_data = orders_res.json() if orders_res.status_code == 200 else []

        # Parse menu items
        items = []
        for menu in menu_data.get("menus", []):
            for group in menu.get("menuGroups", []):
                for item in group.get("menuItems", []):
                    items.append({
                        "guid":     item.get("guid"),
                        "name":     item.get("name"),
                        "price":    item.get("price", 0),
                        "category": group.get("name", ""),
                    })

        # Parse sales by item
        sales = {}
        for order in (orders_data if isinstance(orders_data, list) else []):
            for check in order.get("checks", []):
                for sel in check.get("selections", []):
                    guid = sel.get("itemGuid", "")
                    if guid:
                        sales[guid] = sales.get(guid, 0) + 1

        # Combine
        result = []
        for item in items:
            result.append({
                **item,
                "units_sold_30d": sales.get(item["guid"], 0)
            })

        # Store in Supabase pos_items table
        if result and req.user_id:
            rows = [{
                "user_id":        req.user_id,
                "provider":       "toast",
                "item_guid":      i["guid"],
                "name":           i["name"],
                "price":          i["price"],
                "category":       i["category"],
                "units_sold_30d": i["units_sold_30d"],
                "synced_at":      datetime.now(timezone.utc).isoformat()
            } for i in result]
            supabase_upsert("pos_items", rows, on_conflict="user_id,provider,item_guid")

        return {
            "success": True,
            "items_synced": len(result),
            "items": result[:50]  # return first 50 for display
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/integrations/toast/status/{user_id}")
async def toast_status(user_id: str):
    """Check if user has an active Toast connection."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return {"connected": False}
    url = f"{SUPABASE_URL}/rest/v1/integrations"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
    }
    params = {"user_id": f"eq.{user_id}", "provider": "eq.toast", "select": "status,connected_at,restaurant_guid"}
    res = requests.get(url, headers=headers, params=params, timeout=10)
    data = res.json()
    if data:
        return {"connected": True, **data[0]}
    return {"connected": False}


# ─── EDI Webhook Receiver ─────────────────────────────────────────────────────

def parse_edi_832(edi_text: str) -> list:
    """
    Parse EDI 832 (Price/Sales Catalog) transaction set.
    Returns list of ingredient price records.
    EDI 832 key segments:
      BEG = beginning of transaction
      CTP = price info (unit price)
      PID = product description
      LIN = line item ID (SKU)
      CTT = transaction totals
    """
    items = []
    current = {}
    lines = edi_text.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # EDI segments are delimited by * and terminated by ~
        seg = line.rstrip('~').split('*')
        if not seg:
            continue

        seg_id = seg[0].upper()

        if seg_id == 'LIN':
            # Save previous item if complete
            if current.get('name') and current.get('cost'):
                items.append(current)
            current = {}
            # LIN*1*IN*SKU123 — element 3 is qualifier, 4 is SKU
            if len(seg) >= 4:
                current['sku'] = seg[3]

        elif seg_id == 'PID':
            # PID*F****Product Description
            if len(seg) >= 6 and seg[5]:
                current['name'] = seg[5].strip()

        elif seg_id == 'CTP':
            # CTP**CON*12.50*1*EA — price per unit
            # CTP qualifier, price type, price, quantity, unit
            if len(seg) >= 4:
                try:
                    price_str = seg[3].replace('$', '').replace(',', '').strip()
                    current['cost'] = float(price_str)
                    current['unit'] = seg[5].strip() if len(seg) >= 6 else 'each'
                    current['pack_size'] = seg[4].strip() if len(seg) >= 5 else ''
                except (ValueError, IndexError):
                    pass

        elif seg_id == 'ITD':
            # ITD = Terms of Sale — skip but note distributor terms
            pass

        elif seg_id == 'N1':
            # N1*SE*Sysco Foods — seller/distributor name
            if len(seg) >= 3 and seg[1].upper() == 'SE':
                current['distributor'] = seg[2].strip()

    # Don't forget the last item
    if current.get('name') and current.get('cost'):
        items.append(current)

    return items


class EdiWebhookRequest(BaseModel):
    user_id: str
    edi_content: str          # raw EDI 832 text
    distributor: Optional[str] = None
    transaction_type: str = "832"  # 832 = price catalog, 855 = PO ack, 810 = invoice

@app.post("/integrations/edi/webhook")
async def edi_webhook(req: EdiWebhookRequest):
    """
    Receive EDI files from middleware (SPS Commerce, TrueCommerce, Orderful).
    For now accepts raw EDI text via POST — middleware will call this endpoint
    when a new EDI file arrives from the distributor.
    """
    if req.transaction_type != "832":
        # Log other transaction types but don't process yet
        return {"received": True, "processed": False, "note": f"Transaction type {req.transaction_type} logged but not yet processed"}

    # Parse EDI 832
    items = parse_edi_832(req.edi_content)
    if not items:
        raise HTTPException(status_code=400, detail="No valid line items found in EDI 832 content")

    # Normalize units
    unit_map = {
        'EA': 'each', 'CS': 'each', 'LB': 'lb', 'OZ': 'oz',
        'KG': 'kg', 'GA': 'fl oz', 'LT': 'l', 'DZ': 'each'
    }

    # Write to price_history in Supabase
    rows = [{
        "user_id":     req.user_id,
        "sku":         item.get('sku', f'EDI-{i}'),
        "name":        item.get('name', 'Unknown'),
        "cost_per":    item.get('cost', 0),
        "unit":        unit_map.get(item.get('unit', 'EA').upper(), 'each'),
        "pack_size":   item.get('pack_size', ''),
        "distributor": req.distributor or item.get('distributor', 'EDI Import'),
        "recorded_at": datetime.now(timezone.utc).isoformat()
    } for i, item in enumerate(items)]

    success = supabase_insert("price_history", rows)

    return {
        "success": success,
        "items_processed": len(items),
        "distributor": req.distributor,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/integrations/edi/test")
async def edi_test(req: EdiWebhookRequest):
    """Parse EDI content and return what would be imported — without saving."""
    items = parse_edi_832(req.edi_content)
    return {
        "items_found": len(items),
        "preview": items[:10],
        "message": f"Found {len(items)} items. Use /integrations/edi/webhook to import."
    }


# ─── Integration Status ───────────────────────────────────────────────────────

@app.get("/integrations/status/{user_id}")
async def integrations_status(user_id: str):
    """Return all active integrations for a user."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return {"integrations": []}
    url = f"{SUPABASE_URL}/rest/v1/integrations"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
    }
    params = {"user_id": f"eq.{user_id}", "select": "provider,status,connected_at,restaurant_guid"}
    res = requests.get(url, headers=headers, params=params, timeout=10)
    return {"integrations": res.json() if res.status_code == 200 else []}


@app.delete("/integrations/{user_id}/{provider}")
async def disconnect_integration(user_id: str, provider: str):
    """Disconnect an integration."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    url = f"{SUPABASE_URL}/rest/v1/integrations"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
    }
    params = {"user_id": f"eq.{user_id}", "provider": f"eq.{provider}"}
    res = requests.delete(url, headers=headers, params=params, timeout=10)
    return {"success": res.status_code in (200, 204)}


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
