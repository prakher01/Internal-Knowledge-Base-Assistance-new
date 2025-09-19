import os
import uuid
import json
from typing import List, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# Always load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

PROJECT_METADATA_KEY = "project"
PROJECT_METADATA_VALUE = "AI-Internal-Knowledge-Base-Assistant"


def _get_env(name: str, required: bool = True, default: str | None = None) -> str:
	value = os.getenv(name, default)
	if required and not value:
		raise ValueError(f"Missing required env var: {name}")
	return value or ""


def _format_employee(emp: Dict[str, Any]) -> str:
	parts = [
		f"ID: {emp.get('id')}",
		f"Name: {emp.get('name')}",
		f"Designation: {emp.get('designation')}",
		f"Department: {emp.get('department')}",
		f"Location: {emp.get('location')}",
	]
	return " | ".join([p for p in parts if p])


def _load_employees(json_path: str) -> List[Dict[str, Any]]:
	if not os.path.isfile(json_path):
		raise FileNotFoundError(f"Employee JSON not found: {json_path}")
	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError("employee_data.json should contain a list of employee objects")
	return data


def _embed_texts(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
	embeddings: List[List[float]] = []
	batch_size = 96
	for i in range(0, len(texts), batch_size):
		batch = texts[i : i + batch_size]
		resp = client.embeddings.create(input=batch, model=model)
		embeddings.extend([d.embedding for d in resp.data])
	return embeddings


def ingest_data(json_path: str | None = None) -> None:
	openai_api_key = _get_env("OPENAI_API_KEY")
	embed_model = _get_env("OPENAI_EMBED_MODEL", required=False, default="text-embedding-3-small")

	pinecone_api_key = _get_env("PINECONE_API_KEY")
	index_name = _get_env("PINECONE_INDEX")

	client = OpenAI(api_key=openai_api_key)
	pc = Pinecone(api_key=pinecone_api_key)
	index = pc.Index(index_name)

	if json_path is None:
		base_dir = os.path.dirname(os.path.abspath(__file__))
		json_path = os.path.join(base_dir, "employee_data.json")

	employees = _load_employees(json_path)
	texts = [_format_employee(emp) for emp in employees]
	vectors = _embed_texts(client, texts, embed_model)

	source_tag = os.path.basename(json_path)
	upserts = []
	for emp, vector in zip(employees, vectors):
		vector_id = str(emp.get("id") or uuid.uuid4())
		metadata = {
			"source": source_tag,
			"text": _format_employee(emp),
			"type": "employee_record",
			PROJECT_METADATA_KEY: PROJECT_METADATA_VALUE,
		}
		upserts.append({"id": str(vector_id), "values": vector, "metadata": metadata})

	batch_size = 100
	for i in range(0, len(upserts), batch_size):
		batch = upserts[i : i + batch_size]
		index.upsert(vectors=batch)

	print(f"Ingested {len(upserts)} employee records into index '{index_name}' with project metadata.")


if __name__ == "__main__":
	ingest_data()
