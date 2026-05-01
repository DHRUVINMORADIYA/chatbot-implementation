import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Policy Chatbot", layout="wide")
st.title("Policy Chatbot")

admin_tab, chat_tab = st.tabs(["Admin", "Chat"])


# ── Admin Tab ─────────────────────────────────────────────────────────────────
with admin_tab:
    st.header("Tenant Management")

    # List existing tenants
    try:
        tenants_resp = requests.get(f"{API_BASE}/admin/tenants", timeout=5)
        data = tenants_resp.json()
        st.write(f"**Registered tenants ({data['count']}):**")
        for t in data["tenants"]:
            st.write(f"- `{t}`")
    except Exception as e:
        st.error(f"Could not reach API: {e}")

    st.divider()

    # Onboard a new tenant
    st.subheader("Onboard New Tenant")
    st.caption("Upload policy PDFs and fill in details. The system will build the vector store automatically.")

    col1, col2 = st.columns(2)
    with col1:
        new_tenant_id = st.text_input("Tenant ID", placeholder="e.g. Flight_Support")
    with col2:
        new_display_name = st.text_input("Display Name", placeholder="e.g. Flight Support Assistant")

    uploaded_files = st.file_uploader(
        "Upload policy documents (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Onboard Tenant", type="primary"):
        if not new_tenant_id.strip():
            st.warning("Tenant ID is required.")
        elif not uploaded_files:
            st.warning("At least one PDF document is required.")
        else:
            with st.spinner("Ingesting documents and building vector store. This may take a minute..."):
                files_payload = [
                    ("files", (f.name, f.read(), "application/pdf"))
                    for f in uploaded_files
                ]
                try:
                    resp = requests.post(
                        f"{API_BASE}/admin/tenants/onboard",
                        data={
                            "tenant_id": new_tenant_id.strip(),
                            "display_name": new_display_name.strip() or new_tenant_id.strip(),
                        },
                        files=files_payload,
                        timeout=180,
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(f"Tenant '{result['tenant_id']}' onboarded successfully.")
                        st.write(f"Files saved: {result['files_saved']}")
                        st.write(f"Chunks created: {result['ingestion']['chunks_created']}")
                    else:
                        st.error(f"Error: {resp.json().get('detail', resp.text)}")
                except Exception as e:
                    st.error(f"Request failed: {e}")


# ── Chat Tab ──────────────────────────────────────────────────────────────────
with chat_tab:

    # Fetch available tenants
    try:
        tenants = requests.get(f"{API_BASE}/tenants", timeout=5).json().get("tenants", [])
    except Exception:
        tenants = []

    # ── Initialise session state keys ─────────────────────────────────────────
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "locked_tenant" not in st.session_state:
        st.session_state.locked_tenant = None   # None = not yet confirmed

    left_col, right_col = st.columns([1, 3], gap="large")

    # ── LEFT: tenant selector panel ───────────────────────────────────────────
    with left_col:
        st.subheader("Assistants")

        if not tenants:
            st.warning("No tenants registered yet. Use the Admin tab to onboard one.")
        else:
            if st.session_state.locked_tenant:
                # Show list with the active one highlighted, plus a reset button
                for t in tenants:
                    if t == st.session_state.locked_tenant:
                        st.markdown(
                            f"<div style='padding:8px 12px;border-radius:6px;"
                            f"background:#1f77b4;color:white;font-weight:600;'>"
                            f"✓ {t}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div style='padding:8px 12px;border-radius:6px;"
                            f"background:#f0f0f0;color:#555;'>{t}</div>",
                            unsafe_allow_html=True,
                        )
                    st.write("")   # small spacing

                st.divider()
                if st.button("Switch Assistant", use_container_width=True):
                    st.session_state.locked_tenant = None
                    st.session_state.messages = []
                    st.session_state.session_id = str(uuid.uuid4())
                    st.rerun()
            else:
                # Tenant not yet locked — show radio + confirm button
                chosen = st.radio(
                    "Select an assistant",
                    tenants,
                    index=0,
                    label_visibility="collapsed",
                )
                st.write("")
                if st.button("Start Chat →", type="primary", use_container_width=True):
                    st.session_state.locked_tenant = chosen
                    st.session_state.messages = []
                    st.session_state.session_id = str(uuid.uuid4())
                    st.rerun()

    # ── RIGHT: chat window ────────────────────────────────────────────────────
    with right_col:
        if not st.session_state.locked_tenant:
            st.info("← Select an assistant from the left panel to begin.")
        else:
            locked = st.session_state.locked_tenant
            st.subheader(f"Chat — {locked}")
            st.caption(f"Session `{st.session_state.session_id[:8]}...`")
            st.divider()

            # Render existing chat history
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Chat input
            user_input = st.chat_input("Ask a policy question...")
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            resp = requests.post(
                                f"{API_BASE}/chat",
                                json={
                                    "session_id": st.session_state.session_id,
                                    "tenant_id": locked,
                                    "message": user_input,
                                    "user_role": "employee",
                                },
                                timeout=30,
                            )
                            result = resp.json().get("result", {})
                            rtype = result.get("type", "")

                            if rtype == "answer":
                                bot_text = result["rag"].get("answer", "No answer returned.")
                                citations = result["rag"].get("citations", [])
                                if citations:
                                    cit_str = ", ".join(
                                        f"p{c.get('page', '?')}" for c in citations
                                    )
                                    bot_text += f"\n\n_Sources: {cit_str}_"

                            elif rtype in ("clarification", "out_of_scope", "smalltalk"):
                                bot_text = result.get("message", "")

                            elif rtype == "escalation_offer":
                                bot_text = result.get("message", "")
                                if result.get("rag"):
                                    bot_text += "\n\n" + result["rag"].get("answer", "")

                            else:
                                bot_text = str(result)

                            st.markdown(bot_text)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": bot_text}
                            )

                        except Exception as e:
                            err = f"API error: {e}"
                            st.error(err)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": err}
                            )
