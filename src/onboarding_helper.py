# src/domain_loader.py

import os
from typing import Any, Dict

import yaml


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TENANTS_DIR = os.path.join(BASE_DIR, "tenants")


REQUIRED_TOP_LEVEL_KEYS = [
    "tenant_id",
    "domain",
    "display_name",
    "retrieval",
    "policy",
    "intents_profile",
    "slots_profile",
    "prompts_profile",
]


def _find_tenant_config_path(tenant_id: str) -> str:
    config_dir = os.path.join(TENANTS_DIR, tenant_id, "config")
    yml_path = os.path.join(config_dir, "tenant.yml")
    yaml_path = os.path.join(config_dir, "tenant.yaml")

    if os.path.exists(yml_path):
        return yml_path
    if os.path.exists(yaml_path):
        return yaml_path

    raise FileNotFoundError(
        f"No tenant config found for '{tenant_id}'. Expected {yml_path} or {yaml_path}"
    )


def load_tenant_config(tenant_id: str) -> Dict[str, Any]:
    config_path = _find_tenant_config_path(tenant_id)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in config]
    if missing:
        raise ValueError(
            f"Tenant config missing required keys: {missing}. File: {config_path}"
        )

    # Basic nested defaults/validation
    retrieval = config.get("retrieval", {})
    policy = config.get("policy", {})

    if "top_k" not in retrieval:
        raise ValueError(f"'retrieval.top_k' missing in {config_path}")
    if "fallback_threshold_for_escalation" not in policy:
        raise ValueError(
            f"'policy.fallback_threshold_for_escalation' missing in {config_path}"
        )

    return config


def tenant_data_path(tenant_id: str) -> str:
    return os.path.join(TENANTS_DIR, tenant_id, "data")


def tenant_vector_store_path(tenant_id: str) -> str:
    return os.path.join(TENANTS_DIR, tenant_id, "artifacts", "vector_store")
