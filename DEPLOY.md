# Plate Pulse API — Railway Deploy Guide

## What this is
The Plate Pulse backend — a FastAPI server that:
- Fetches Google Maps reviews via Outscraper
- Runs RoBERTa ML sentiment analysis on every review
- Returns structured data to the Plate Pulse frontend (reviews.html)

## Files needed for deployment
```
main.py          ← the FastAPI app
requirements.txt ← Python dependencies
railway.json     ← Railway config
nixpacks.toml    ← Build instructions (installs CPU-only torch)
Procfile         ← Start command fallback
runtime.txt      ← Python version
.gitignore       ← Excludes model files and secrets
```

## Step 1 — Create a GitHub repo
1. Go to github.com and create a new repository (e.g. `platepulse-api`)
2. Make it private if you don't want the code public

## Step 2 — Push these files to GitHub
On your machine, in a folder containing these files:
```bash
git init
git add main.py requirements.txt railway.json nixpacks.toml Procfile runtime.txt .gitignore
git commit -m "Initial deploy"
git remote add origin https://github.com/YOUR_USERNAME/platepulse-api.git
git push -u origin main
```

## Step 3 — Deploy on Railway
1. Go to railway.app and sign up (free)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your `platepulse-api` repo
4. Railway will detect the config and start building

## Step 4 — Set environment variables
In Railway dashboard → your project → "Variables" tab, add:
```
OUTSCRAPER_KEY   = your_outscraper_api_key_here
SERPAPI_KEY      = your_serpapi_key_here
```

## Step 5 — Get your deployment URL
In Railway dashboard → your project → "Settings" → "Domains"
Click "Generate Domain" — you'll get a URL like:
```
https://platepulse-api-production.up.railway.app
```

## Step 6 — Update reviews.html
Open reviews.html and find this line near the top of the script:
```javascript
const BACKEND = 'http://localhost:8000';
```
Change it to your Railway URL:
```javascript
const BACKEND = 'https://platepulse-api-production.up.railway.app';
```

## Step 7 — Test it
Visit your Railway URL in the browser. You should see:
```json
{
  "service": "Plate Pulse API",
  "version": "1.0.0",
  "status": "running"
}
```

Then try the Google Fetch tab in reviews.html with a restaurant address.

## Cost
- Railway Hobby plan: $5/month
- Includes 512MB RAM, 1 vCPU, enough for this workload
- Outscraper: pay-per-use (~$0.002 per review fetched)

## Notes
- The RoBERTa model (~500MB) downloads from HuggingFace on first boot
  First startup takes 2-3 minutes. After that it's cached.
- Fine-tuned model weights (if you collect feedback) are stored on the
  Railway volume — they won't persist between deploys unless you add
  a Railway Volume. For now the base model reloads each deploy.
- For production with persistent fine-tuning, add a Railway Volume
  mounted at `/app/model`

## Troubleshooting
- Build fails: check Railway logs for pip install errors
- 502 errors: model is still loading, wait 2-3 min on first boot
- No reviews found: make sure address includes city and state
  e.g. "Haymaker Goodyear, 1234 N Litchfield Rd, Goodyear AZ 85395"
