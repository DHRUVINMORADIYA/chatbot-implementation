import json
from typing import Dict, Any

from rag_answer import answer_query, call_llm
from onboarding_helper import load_tenant_config
import yaml
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SESSIONS: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    # Create or return in-memory conversation state for one session.
    if session_id not in SESSIONS:
        print(f"[SESSION] New session created: {session_id}")
        SESSIONS[session_id] = {
            "current_intent": None,
            "collected_slots": {},
            "last_user_question": None,
            "last_answer_summary": None,
            "clarification_pending": False,
            "clarification_rounds": 0,
            "fallback_count": 0,
            "escalation_offered": False,
        }
    else:
        print(f"[SESSION] Resumed session: {session_id}")
    return SESSIONS[session_id]


def _load_intents_slots(tenant_id: str) -> tuple[dict, dict]:
    # Load tenant-specific intent and slot schemas from YAML config files.
    cfg_dir = os.path.join(BASE_DIR, "tenants", tenant_id, "config")
    with open(os.path.join(cfg_dir, "intents.yml"), "r", encoding="utf-8") as f:
        intents = yaml.safe_load(f) or {}
    with open(os.path.join(cfg_dir, "slots.yml"), "r", encoding="utf-8") as f:
        slots = yaml.safe_load(f) or {}
    print(f"[CONFIG] Loaded {len(intents)} intents, {len(slots)} slots for tenant: {tenant_id}")
    return intents, slots


def detect_intent(user_message: str, intents_schema: dict) -> str:
    # Classify user input into one configured intent key.
    print(f"[INTENT] Detecting intent for: '{user_message}'")
    prompt = f"""
Classify the user message into exactly one intent key.

Intent keys:
{json.dumps(list(intents_schema.keys()), indent=2)}

User message: "{user_message}"

Return ONLY JSON:
{{"intent":"one_of_the_keys"}}
"""
    out = call_llm(prompt)
    raw_intent = out.get("intent", "out_of_scope")
    intent = raw_intent if raw_intent in intents_schema else "out_of_scope"
    print(f"[INTENT] Detected: '{intent}' (raw LLM output: '{raw_intent}')")
    return intent


def extract_slots(user_message: str, slots_schema: dict) -> dict:
    # Extract slot values from the user utterance using configured slot value options.
    print(f"[SLOTS] Extracting slots from: '{user_message}'")
    slot_values = {k: v.get("possible_values", []) for k, v in slots_schema.items()}
    prompt = f"""
Extract slot values from the user message.

Slots and allowed values:
{json.dumps(slot_values, indent=2)}

User message: "{user_message}"

Return ONLY JSON:
{{"slots": {{"slot_name": "value_or_null"}}}}
Only include slots that are confidently present.
"""
    out = call_llm(prompt)
    extracted = out.get("slots", {}) if isinstance(out.get("slots", {}), dict) else {}
    filled = {k: v for k, v in extracted.items() if v and v != "null"}
    print(f"[SLOTS] Extracted: {filled}")
    return extracted


def process_turn(session_id: str, tenant_id: str, user_message: str) -> Dict[str, Any]:
    # Execute one full Layer-3 turn: intent -> slots -> clarification/fallback/escalation -> answer.
    print(f"\n{'='*60}")
    print(f"[TURN] session={session_id} | tenant={tenant_id}")
    print(f"[TURN] User: '{user_message}'")

    tenant_cfg = load_tenant_config(tenant_id)
    intents_schema, slots_schema = _load_intents_slots(tenant_id)
    state = _get_session(session_id)
    state["last_user_question"] = user_message

    intent = detect_intent(user_message, intents_schema)
    state["current_intent"] = intent

    display_name = tenant_cfg.get("display_name", tenant_id)
    policy = tenant_cfg.get("policy", {})

    if intent == "greeting_or_smalltalk":
        print(f"[TURN] Routing: smalltalk -> direct response")
        msg = policy.get("smalltalk_response") or f"Hello! I'm the {display_name}. How can I help you today?"
        return {"type": "smalltalk", "message": msg}

    if intent == "out_of_scope":
        print(f"[TURN] Routing: out_of_scope -> direct response")
        msg = policy.get("out_of_scope_response") or f"I can only assist with {display_name} questions."
        return {"type": "out_of_scope", "message": msg}

    new_slots = extract_slots(user_message, slots_schema)
    state["collected_slots"].update({k: v for k, v in new_slots.items() if v})
    print(f"[SLOTS] Session state after update: {state['collected_slots']}")

    required = intents_schema[intent].get("required_slots", [])
    missing = [s for s in required if not state["collected_slots"].get(s)]
    print(f"[SLOTS] Required: {required} | Missing: {missing}")

    if missing:
        state["clarification_pending"] = True
        state["clarification_rounds"] += 1
        missing_str = ", ".join(missing)
        print(f"[TURN] Routing: clarification needed (round {state['clarification_rounds']})")
        return {"type": "clarification", "message": f"To answer accurately, please provide: {missing_str}."}

    state["clarification_pending"] = False
    query_augmented = f"Intent: {intent}\nSlots: {state['collected_slots']}\nUser: {user_message}"
    print(f"[RAG] Sending augmented query to answer_query")
    print(f"[RAG] Query: {query_augmented}")
    rag_out = answer_query(query_augmented, tenant_id=tenant_id, k=tenant_cfg["retrieval"]["top_k"])
    print(f"[RAG] Confidence: {rag_out.get('confidence_mode')}")

    if rag_out.get("confidence_mode") == "LOW" or "do not have enough information" in rag_out.get("answer", "").lower():
        state["fallback_count"] += 1
        print(f"[FALLBACK] Low confidence. fallback_count={state['fallback_count']}")
    else:
        state["fallback_count"] = 0
        print(f"[FALLBACK] Confidence OK. fallback_count reset to 0")

    threshold = tenant_cfg["policy"]["fallback_threshold_for_escalation"]
    if state["fallback_count"] >= threshold and not state["escalation_offered"]:
        state["escalation_offered"] = True
        print(f"[ESCALATION] Threshold {threshold} reached. Offering escalation.")
        return {
            "type": "escalation_offer",
            "message": tenant_cfg["policy"]["escalation_message"],
            "rag": rag_out,
        }

    state["last_answer_summary"] = rag_out.get("answer", "")[:160]
    print(f"[TURN] Routing: answer returned")
    print(f"{'='*60}")
    return {"type": "answer", "rag": rag_out}


if __name__ == "__main__":
    # Runpoint for Layer-3 local testing (interactive terminal loop).
    tenant_id = "Pay_Benefits_and_Leave"
    session_id = "local-demo"

    print(f"Layer 3 orchestrator started for tenant: {tenant_id}")
    print("Type 'exit' to stop.\n")

    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            print("Stopping orchestrator.")
            break

        result = process_turn(session_id=session_id, tenant_id=tenant_id, user_message=user_message)
        print("Bot:", json.dumps(result, indent=2))