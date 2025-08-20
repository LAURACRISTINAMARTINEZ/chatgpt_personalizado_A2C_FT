# app_streamlit.py
import os, re, streamlit as st
from openai import OpenAI

# --- SQLite shim para Streamlit Cloud (debe ir ANTES de importar chromadb) ---
try:
    __import__("pysqlite3")           # carga sqlite moderno empacado en pysqlite3-binary
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
# ------------------------------------------------------------------------------

import chromadb

# ✅ Config UI lo más arriba posible
st.set_page_config(page_title="A2C Fertrac", page_icon="🤖")

def _preclean_text(s: str) -> str:
    """
    Elimina conteos como '3 referencias', '2 refs', etc., para que el número
    no se interprete como referencia.
    """
    if not s:
        return ""
    t = s.upper()
    t = re.sub(r"\b\d+\s+(?:REFERENCIAS?|REFS?)\b", " ", t)
    t = re.sub(r"\bDE\s+ESTAS?\s+\d+\b", "DE ESTAS ", t)
    return t


# ===== Config =====
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"
DB_PATH     = "db"
COLLECTION  = "insumo"   # ← igual que en cargar_datos.py

SYSTEM_PROMPT = (
    "Eres el Asesor Comercial de Fertrac. Responde solo con base en los registros encontrados. "
    "Si falta un dato, dilo y sugiere validarlo en el sistema interno. No inventes. Sé claro y breve."
)

# ===== OpenAI =====
# Cargar desde Secrets si no existe en env (para Streamlit Cloud)
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Falta OPENAI_API_KEY. Ve a Settings → Secrets en Streamlit Cloud y agrégala.")
    st.stop()

client = OpenAI(api_key=api_key)

# ===== Chroma (cosine) =====
chroma = chromadb.PersistentClient(path=DB_PATH)
col = chroma.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

# --------- Indexar automáticamente en cada visita ----------
def auto_index_on_visit():
    from cargar_datos import build_index   # usa tu función del loader
    with st.status("🔄 Indexando…", expanded=False) as s:
        added = build_index(
            insumo_dir="insumo",
            db_path=DB_PATH,
            collection=COLLECTION,
            force_drop=False            # no borra: solo agrega lo que falte
        )
        s.update(label=f"✅ Indexación lista (filas nuevas: {added})", state="complete")
    # forzar que se reconstruya el mapa de refs después de indexar
    st.session_state.ref_index_version = st.session_state.get("ref_index_version", 0) + 1

auto_index_on_visit()
# -----------------------------------------------------------


# =====================================================
# Índice de referencias + normalización (cacheado)
# =====================================================
_TOK_RX   = re.compile(r"[A-Z0-9]+", re.I)

def _norm_ref(s: str) -> str:
    """Mayúsculas y sin separadores: deja solo A-Z/0-9."""
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())

@st.cache_data(show_spinner=False)
def _build_ref_index(cache_buster: int = 0):
    """
    Devuelve:
      - ALL_REFS: set de refs canónicas (mayúsculas)
      - REF_MAP : dict normalizado -> set de refs reales (para resolución)
      - REF_ALIASES: dict ref_canónica -> set de alias normalizados (incluye la canónica)
    """
    all_refs = set()
    ref_map  = {}
    ref_aliases = {}  # ref -> set(norm_alias)

    step, offset = 10000, 0
    while True:
        data = col.get(include=["metadatas"], limit=step, offset=offset)  # usa 'col' global
        metas = data.get("metadatas") or []
        if not metas:
            break
        for m in metas:
            if not m:
                continue

            r = (m.get("ref") or "").upper().strip()
            if not r:
                continue

            all_refs.add(r)
            nr = _norm_ref(r)
            ref_map.setdefault(nr, set()).add(r)
            ref_aliases.setdefault(r, set()).add(nr)  # incluye la propia

            # aliases puede venir como str "A|B|C" o list
            aliases_field = m.get("aliases") or ""
            if isinstance(aliases_field, str):
                alias_list = [s for s in re.split(r"[|,;\\s]+", aliases_field) if s]
            elif isinstance(aliases_field, list):
                alias_list = [str(s) for s in aliases_field if s]
            else:
                alias_list = []

            for a in alias_list:
                na = _norm_ref(a)
                if na:
                    ref_map.setdefault(na, set()).add(r)
                    ref_aliases.setdefault(r, set()).add(na)

        if len(metas) < step:
            break
        offset += step

    return all_refs, ref_map, ref_aliases


if "ref_index_version" not in st.session_state:
    st.session_state.ref_index_version = 0

ALL_REFS, REF_MAP, REF_ALIASES = _build_ref_index(st.session_state.ref_index_version)


def _canon_from_norm(n: str) -> str | None:
    """Devuelve la ref real 'canónica' para una forma normalizada."""
    s = REF_MAP.get(n)
    if not s:
        return None
    # preferir la más larga (más específica)
    return sorted(s, key=len, reverse=True)[0]

def detectar_refs(texto: str) -> list[str]:
    """
    Estrategia codiciosa 3→2→1 tokens. Para 1 token:
      - si es solo numérico, exige ≥4 dígitos (evita '3');
      - si es alfanumérico (letras+números), lo admite.
    """
    texto = _preclean_text(texto)
    tokens = _TOK_RX.findall(texto)
    out = []
    i = 0

    def _canon_from_tokens(tok_list):
        n = _norm_ref("".join(tok_list))
        if not any(ch.isdigit() for ch in n):
            return None
        # regla para 1 token puramente numérico
        if len(tok_list) == 1 and n.isdigit() and len(n) < 4:
            return None
        return _canon_from_norm(n)

    while i < len(tokens):
        match, step = None, 1

        if i + 2 < len(tokens):  # prueba 3 tokens
            c = _canon_from_tokens(tokens[i:i+3])
            if c:
                match, step = c, 3

        if match is None and i + 1 < len(tokens):  # prueba 2 tokens
            c = _canon_from_tokens(tokens[i:i+2])
            if c:
                match, step = c, 2

        if match is None:  # por último 1 token
            c = _canon_from_tokens(tokens[i:i+1])
            if c:
                match, step = c, 1

        if match and match not in out:
            out.append(match)
        i += step

    return out

def detectar_ref_segundachance(texto: str) -> str | None:
    # candidatos: tokens A-Z0-9 con al menos un dígito
    toks = [t for t in _TOK_RX.findall(_preclean_text(texto)) if any(ch.isdigit() for ch in t)]
    # prioriza los más largos; probamos pocos
    for tok in sorted(set(toks), key=len, reverse=True)[:6]:
        cand = tok.upper()
        data = col.get(where={"ref": {"$eq": cand}}, include=[])
        if data.get("ids"):
            return cand
    return None


def _prune_overlaps(refs: list[str], texto: str) -> list[str]:
    """
    Si el texto contiene número + sufijo que forma una ref compuesta,
    elimina el número suelto (evita '30437' cuando existe '30437 000L').
    """
    tokens = _TOK_RX.findall(_preclean_text(texto))
    to_remove = set()

    def _canon_from_str(s: str) -> str | None:
        return _canon_from_norm(_norm_ref(s))

    for i, t in enumerate(tokens):
        if not t.isdigit():
            continue
        single = _canon_from_str(t)
        if not single or single not in refs:
            continue
        for L in (1, 2):
            if i + L < len(tokens):
                combined = _canon_from_str(t + "".join(tokens[i+1:i+1+L]))
                if combined and combined in refs:
                    to_remove.add(single)
                    break
    return [r for r in refs if r not in to_remove]

def _filter_refs_present(refs: list[str], texto: str) -> list[str]:
    """
    Mantiene solo refs cuyo ALIAS normalizado (o la ref canónica) aparece
    como subcadena en el texto normalizado. Evita que se cuele una ref
    que no está escrita (como el 'M-...').
    """
    norm_text = _norm_ref(texto)
    keep = []
    for r in refs:
        alias_norms = REF_ALIASES.get(r, { _norm_ref(r) })
        if any(a in norm_text for a in alias_norms):
            keep.append(r)
    return keep

def detectar_ref(texto: str) -> str | None:
    lst = detectar_refs(texto)
    return lst[0] if lst else None

# =====================================================
# UI (Streamlit)
# =====================================================
st.title("👋 Hola, soy tu asistente **A2C Fertrac**")

with st.sidebar:
    st.subheader("⚙️ Ajustes")
    top_k = st.slider("Resultados vectoriales (k)", 1, 8, 5)
    sim_threshold = st.slider("Umbral de similitud (vectorial)", 0.10, 0.60, 0.30, 0.05)
    st.caption(f"DB: `{DB_PATH}` · Colección: `{COLLECTION}` · Registros: {col.count()}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Listo! Pregúntame por una referencia (p. ej., K3492B o 30437 000L) o compárame varias referencias por precio."}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Escribe tu pregunta…")

if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        refs_multi = detectar_refs(q)
        refs_multi = _prune_overlaps(refs_multi, q)
        refs_multi = _filter_refs_present(refs_multi, q)
        refs_multi = list(dict.fromkeys(refs_multi))  # quita duplicados conservando orden
        if len(refs_multi) >= 2:
            registros = []
            for r in refs_multi:
                data = col.get(where={"ref": {"$eq": r}}, include=["metadatas"])
                metas = data.get("metadatas") or []
                if not metas:
                    registros.append((r, None, {}))
                    continue
                # elige la fila con precio numérico más bajo
                best_meta, best_price = None, None
                for m in metas:
                    p = parse_price(m.get("precio_lista"))
                    if p is not None and (best_price is None or p < best_price):
                        best_price, best_meta = p, m
                if best_meta is None:
                    best_meta = metas[0]
                registros.append((r, best_price, best_meta))

            lineas = []
            validos = [t for t in registros if t[1] is not None]
            if validos:
                ganador = min(validos, key=lambda t: t[1])
                for r, p, m in registros:
                    lineas.append(f"- **{r}**: {fmt_price(p, m.get('precio_lista'))}")
                st.markdown(
                    "He encontrado estas referencias y precios:\n\n"
                    + "\n".join(lineas)
                    + f"\n\n**La referencia con menor precio es {ganador[0]}** "
                      f"({fmt_price(ganador[1], ganador[2].get('precio_lista'))})."
                )
            else:
                for r, p, m in registros:
                    lineas.append(f"- **{r}**: {m.get('precio_lista') or 'N/D'}")
                st.markdown("No hay precios numéricos comparables. Precios encontrados:\n\n" + "\n".join(lineas))

            st.session_state.messages.append({"role": "assistant", "content": "(ver arriba)"})
            st.stop()

        # --- Búsqueda exacta por referencia (estricto, sin fallback) ---
        ref = detectar_ref(q)
        if not ref:
            ref = detectar_ref_segundachance(q)
        if ref:
            with st.spinner("Buscando por referencia exacta…"):
                data = col.get(where={"ref": {"$eq": ref}}, include=["documents", "metadatas"])
                metas = data.get("metadatas") or []

            if metas:
                m0 = metas[0] or {}
                st.markdown(respuesta_desde_meta(q, m0))
            else:
                st.warning(f"No encontré la referencia **{ref}** en la base. ¿Puedes verificar el código completo o enviar otra variante?")
            st.session_state.messages.append({"role": "assistant", "content": "(ver arriba)"})
            st.stop()

        # --- Retrieval vectorial SOLO SI NO HAY REF ---
        with st.spinner("Buscando contexto similar…"):
            emb = client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding
            res = col.query(
                query_embeddings=[emb],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            docs  = (res.get("documents", [[]])[0])  or []
            metas = (res.get("metadatas", [[]])[0])  or []
            dists = (res.get("distances", [[]])[0])  or []

            # Convertir distancia HNSW (cosine) a similitud (1 - dist)
            pairs = []
            for i in range(len(docs)):
                dist = float(dists[i]) if dists else 1.0
                sim  = 1.0 - dist
                if sim >= sim_threshold:
                    pairs.append((sim, docs[i], metas[i]))

            # Re-ranking básico (ya están por similitud; aquí podrías añadir reglas)
            pairs.sort(key=lambda x: x[0], reverse=True)

        if pairs:
            best_meta = pairs[0][2] or {}
            st.markdown(respuesta_desde_meta(q, best_meta))
        else:
            st.markdown("No encontré contexto suficientemente parecido en la base. ¿Puedes dar una referencia o más detalles?")

    st.session_state.messages.append({"role": "assistant", "content": "(ver arriba)"})
