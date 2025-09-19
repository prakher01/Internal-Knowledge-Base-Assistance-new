import os
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from retriever import retrieve_top_k, answer_query


load_dotenv()

app = FastAPI(title="AI Knowledge Base Assistant")

# Mount templates (package-relative)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


class SearchRequest(BaseModel):
	query: str
	top_k: Optional[int] = 5


class AskRequest(BaseModel):
	query: str
	top_k: Optional[int] = 5


@app.get("/")
async def index(request: Request):
	return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search")
async def search(req: SearchRequest):
	try:
		matches = await retrieve_top_k(query=req.query, top_k=req.top_k or 5)
		return JSONResponse({"matches": matches})
	except Exception as exc:  # noqa: BLE001
		return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/ask")
async def ask(req: AskRequest):
	try:
		result = await answer_query(query=req.query, top_k=req.top_k or 5)
		return JSONResponse(result)
	except Exception as exc:  # noqa: BLE001
		return JSONResponse({"error": str(exc)}, status_code=500)


# Optional: static directory if needed in template
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
	app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# To run: uvicorn ai_knowledge_base.main:app --reload
