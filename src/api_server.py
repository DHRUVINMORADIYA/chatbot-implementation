import os
import sys
import shutil
from typing import Any, Dict, List

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# Ensure src/ is on path when running via uvicorn from repo root
sys.path.insert(0, os.path.dirname(__file__))

from orchestrator import process_turn
from onboarding_helper import load_tenant_config, tenant_data_path, tenant_vector_store_path
from ingest import run_ingestion

app = FastAPI(title="Multi-tenant Chatbot API")

# In-memory tenant registry.
# Populated at startup by scanning the tenants/ directory.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TENANTS_DIR = os.path.join(BASE_DIR, "tenants")


def _scan_tenants() -> List[str]:
    # Return list of tenant_ids that have a valid config file present.
    if not os.path.isdir(TENANTS_DIR):
        return []
    return [
        d for d in os.listdir(TENANTS_DIR)
        if os.path.isdir(os.path.join(TENANTS_DIR, d))
        and os.path.isfile(os.path.join(TENANTS_DIR, d, "config", "tenant.yml"))
    ]


TENANTS: List[str] = _scan_tenants()


# ─── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    tenant_id: str
    message: str
    user_role: str = "employee"  # placeholder for future role-based access control


class ChatResponse(BaseModel):
    result: Dict[str, Any]


class AddTenantRequest(BaseModel):
    tenant_id: str


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, str]:
    # Basic liveness check.
    return {"status": "ok"}


# ─── Public endpoints ─────────────────────────────────────────────────────────

@app.get("/tenants")
def list_tenants() -> Dict[str, List[str]]:
    # Return tenants available for the chat UI to display.
    return {"tenants": TENANTS}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # Validate tenant exists before routing to orchestrator.
    if req.tenant_id not in TENANTS:
        raise HTTPException(status_code=404, detail=f"Tenant '{req.tenant_id}' not found.")

    # Role hook: placeholder for future enforcement.
    # Example: if req.user_role == "customer": raise HTTPException(403, "Access denied.")

    print(f"[API] POST /chat | tenant={req.tenant_id} | session={req.session_id} | role={req.user_role} | message='{req.message}'")
    result = process_turn(
        session_id=req.session_id,
        tenant_id=req.tenant_id,
        user_message=req.message,
    )
    print(f"[API] Response type: {result.get('type')}")
    return ChatResponse(result=result)


# ─── Admin endpoints ──────────────────────────────────────────────────────────

@app.get("/admin/tenants")
def admin_list_tenants() -> Dict[str, Any]:
    # Admin view: refreshes from disk on each call.
    current = _scan_tenants()
    return {"tenants": current, "count": len(current)}


@app.post("/admin/tenants/register")
def admin_register_tenant(req: AddTenantRequest) -> Dict[str, Any]:
    # Register an already-onboarded tenant (config files already present) into the active list.
    tenant_path = os.path.join(TENANTS_DIR, req.tenant_id, "config", "tenant.yml")
    if not os.path.isfile(tenant_path):
        raise HTTPException(
            status_code=400,
            detail=f"Tenant config not found at {tenant_path}. Create the tenant folder and config first.",
        )
    if req.tenant_id not in TENANTS:
        TENANTS.append(req.tenant_id)
    return {"registered": True, "tenant_id": req.tenant_id, "active_tenants": TENANTS}


@app.post("/admin/tenants/onboard")
async def onboard_tenant(
    tenant_id: str = Form(...),
    display_name: str = Form(...),
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:
    # Build tenant folder structure.
    tenant_root = os.path.join(TENANTS_DIR, tenant_id)
    data_dir = os.path.join(tenant_root, "data")
    config_dir = os.path.join(tenant_root, "config")
    artifacts_dir = os.path.join(tenant_root, "artifacts", "vector_store")

    if os.path.isfile(os.path.join(config_dir, "tenant.yml")):
        raise HTTPException(status_code=400, detail=f"Tenant '{tenant_id}' already exists.")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save uploaded PDFs to tenant data folder.
    saved_files = []
    for upload in files:
        dest = os.path.join(data_dir, upload.filename)
        content = await upload.read()
        with open(dest, "wb") as f:
            f.write(content)
        saved_files.append(upload.filename)

    # Write tenant.yml from provided metadata.
    tenant_config = {
        "tenant_id": tenant_id,
        "domain": tenant_id,
        "display_name": display_name,
        "retrieval": {"top_k": 4, "min_confidence_mode_for_direct_answer": "MEDIUM"},
        "policy": {
            "max_clarification_rounds": 2,
            "fallback_threshold_for_escalation": 2,
            "escalation_message": "I may not have enough reliable context. Would you like me to create a handoff note?",
        },
        "intents_profile": "default_v1",
        "slots_profile": "default_v1",
        "prompts_profile": "default_v1",
    }
    with open(os.path.join(config_dir, "tenant.yml"), "w", encoding="utf-8") as f:
        yaml.dump(tenant_config, f, default_flow_style=False, allow_unicode=True)

    # Write minimal starter intents.yml (user can edit after onboarding).
    default_intents = {
        "general_inquiry": {
            "description": "General questions about the documents in this tenant.",
            "required_slots": [],
            "optional_slots": [],
        },
        "out_of_scope": {
            "description": "Questions not covered by this tenant's documents.",
            "required_slots": [],
            "optional_slots": [],
        },
        "greeting_or_smalltalk": {
            "description": "Greetings, thanks, and casual conversation.",
            "required_slots": [],
            "optional_slots": [],
        },
    }
    with open(os.path.join(config_dir, "intents.yml"), "w", encoding="utf-8") as f:
        yaml.dump(default_intents, f, default_flow_style=False, allow_unicode=True)

    # Write empty starter slots.yml.
    with open(os.path.join(config_dir, "slots.yml"), "w", encoding="utf-8") as f:
        f.write("# Add domain-specific slots here after onboarding.\n")

    # Run ingestion: chunk, embed, persist vector store.
    ingest_result = run_ingestion(tenant_id)

    # Register in active tenant list.
    if tenant_id not in TENANTS:
        TENANTS.append(tenant_id)

    return {
        "onboarded": True,
        "tenant_id": tenant_id,
        "display_name": display_name,
        "files_saved": saved_files,
        "ingestion": ingest_result,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
