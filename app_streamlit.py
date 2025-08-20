# app_streamlit.py
import os, re, streamlit as st
from openai import OpenAI

# --- SQLite shim para Streamlit Cloud (debe ir ANTES de importar chromadb) ---
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
# ------------------------------------------------------------------------------

import chromadb

# ✅ Config UI lo más arriba posible
st.set_page_config(page_title="A2C Fertrac", page_icon="🤖")

# =====================================================
# Helpers (definidos ARRIBA para que siempre existan)
# =====================================================
_TOK_RX = re.compile(r"[A-Z0-9]+", re.I)

def _preclean_text(s: str) -> str:
    if not s:
        return ""
    t = s.upper()
    t = re.sub(r"\b\d+\s+(?:REFERENCIAS?|REFS?)\b", " ", t)
    t = re.sub(r"\bDE\s+ESTAS?\s+\d+\b", "DE ESTAS ", t)
    return t

def _norm_ref(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())

def parse_price(s: str | None) -> float | None:
    if not s:
        return None
    sx = s.strip().upper()
    if sx in {"#N/D", "N/D", "ND"}:
        return None
    sx = sx.replace("$", "").replace(" ", "")
    sx = sx.replace(".", "").replace(",", ".")
    try:
        return float(sx)
    except:
        return None

def fmt_price(v: float | None, original: str | None) -> str:
    if v is None:
        return original if original else "N/D"
    return f"${v:,.2f}"

def respuesta_desde_meta(q: str, m: dict) -> str:
    ref    = m.get("ref") or "—"
    nom    = m.get("nombre") or "—"
    desc   = m.get("descripcion") or "—"
    marca  = m.get("marca") or "—"
    linea  = m.get("linea") or "—"
    sub    = m.get("sublinea") or "—"
    clas   = m.get("clasificacion") or "—"
    inv    = m.get("inventario") or "—"
    precio = m.get("precio_lista") or "—"

    ql = (q or "").lower()
    if any(k in ql for k in ["inventario", "stock", "existencias", "disponible", "cuánto hay", "cuanto hay"]):
        return f"Para la referencia **{ref}**, el inventario disponible es **{inv}**."
    if any(k in ql for k in ["precio", "vale", "cuánto cuesta", "cuanto cuesta", "lista"]):
        return f"El precio de lista de **{ref}** es **{precio}**."

    return (
        f"**Referencia:** {ref}\n\n"
        f"**Nombre:** {nom}\n\n"
        f"**Descripción:** {desc}\n\n"
        f"**Marca / Línea / Sub-línea:** {marca} / {linea} / {sub}\n\n"
        f"**Clasificación:** {clas}\n\n"
        f"**Inventario:** {inv}\n\n"
        f"**Precio lista:** {precio}"
    )

# ===== Config =====
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"
DB_PATH     = "db"
COLLECTION  = "insumo"

SYSTEM_PROMPT = (
    "Eres el Asesor Comercial de Fertrac. Responde solo con base en los registros encontrados. "
    "Si falta un dato, dilo y sugiere validarlo en el sistema interno. No inventes. Sé claro y breve."
)

# ===== OpenAI =====
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Falta OPENAI_API_KEY. Ve a Settings → Secrets en Streamlit Cloud y agrégala.")
