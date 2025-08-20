# cargar_datos.py
# -*- coding: utf-8 -*-
import os, re, hashlib
import pandas as pd

# --- SQLite shim para Streamlit Cloud ---
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
# ---------------------------------------

import chromadb
from tqdm import tqdm
from openai import OpenAI

# =========================
# Configuración
# =========================
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE  = 64
DB_PATH     = "db"
COLLECTION  = "insumo"     # <-- usa el MISMO nombre en tu app_streamlit.py
INSUMO_DIR  = "insumo"     # carpeta donde pones tus .xlsx/.xls/.csv/.txt

# API key (no la hardcodees)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Falta OPENAI_API_KEY en el entorno.")
client = OpenAI(api_key=api_key, timeout=60, max_retries=5)

# Chroma (HNSW espacio coseno = ideal para embeddings OpenAI)
ch = chromadb.PersistentClient(path=DB_PATH)
col = ch.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}  # 👈 importante
)

# =========================
# Utilidades
# =========================
REF_RX = re.compile(r"\b([A-Z0-9\-]+)\b")

def read_table(path: str) -> pd.DataFrame:
    """Lee .xlsx/.xls/.csv/.txt (delimitado) a DataFrame de strings."""
    p = path.lower()
    if p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(path, engine="openpyxl", dtype=str)
    if p.endswith(".csv"):
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    # .txt / delimitado: autodetección
    try:
        return pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)
    except Exception:
        # fallback común: punto y coma
        return pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)

def norm(s) -> str:
    if pd.isna(s) or s is None: return ""
    return str(s).strip()

def extract_ref(s: str) -> str | None:
    """
    Devuelve la referencia completa a partir de la celda:
    - Mayúsculas
    - Toma TODOS los tokens [A-Z0-9]+ y los une con un espacio.
    Ejemplos:
      '30437 000L'          -> '30437 000L'
      '30437/10.168MANS'    -> '30437 10 168MANS'
      'B30437/30240-5'      -> 'B30437 30240 5'
    """
    s = norm(s).upper()
    if not s:
        return None
    toks = re.findall(r"[A-Z0-9]+", s)
    if not toks:
        return None
    return " ".join(toks)


def to_doc(row: dict) -> str:
    """Documento corto por fila (útil para recuperación semántica)."""
    campos = [
        ("REFERENCIA", row.get("REFERENCIA","")),
        ("REFERENCIAS ALTERNAS", row.get("REFERENCIAS ALTERNAS","")),
        ("NOMBRE", row.get("NOMBRE","")),
        ("DESCRIPCION", row.get("DESCRIPCION","")),
        ("MARCA", row.get("MARCA","")),
        ("LINEA", row.get("LINEA","")),
        ("SUB-LINEA", row.get("SUB-LINEA","")),
        ("CLASIFICACION", row.get("CLASIFICACION","")),
        ("INVENTARIO", row.get("INVENTARIO","")),
        ("PRECIO LISTA", row.get("PRECIO LISTA","")),
    ]
    return "\n".join(f"{k}: {norm(v)}" for k, v in campos if norm(v))

def id_for(ref: str | None, row_idx: int, filename: str) -> str:
    """ID determinista y único por archivo + fila + ref."""
    sig = f"{os.path.basename(filename)}|row={row_idx}|ref={ref or ''}"
    return hashlib.md5(sig.encode()).hexdigest()

def mask_missing(collection, ids: list[str]) -> list[int]:
    """Devuelve índices de 'ids' que NO existen aún en la colección."""
    keep: list[int] = []
    for s in range(0, len(ids), 1000):
        batch = ids[s:s+1000]
        got = collection.get(ids=batch, include=[])
        existing = set(got.get("ids", []))
        for i, _id in enumerate(batch, start=s):
            if _id not in existing:
                keep.append(i)
    return keep

def clean_meta(meta: dict) -> dict:
    """Convierte None/NaN a cadena y asegura tipos válidos (Bool/Int/Float/Str).
       Si hay listas (ej. aliases), las serializa a string."""
    out = {}
    for k, v in meta.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out[k] = ""
        elif isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, list):
            # serializa listas a un string seguro (pipe-sep)
            out[k] = "|".join(str(x) for x in v if x is not None)
        else:
            out[k] = str(v)
    return out


# --- Aliases desde "REFERENCIAS ALTERNAS" ---
def _norm_ref(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())

def split_aliases(alt_raw: str) -> list[str]:
    """
    Parte 'REFERENCIAS ALTERNAS' en posibles alias y los normaliza (A-Z/0-9).
    Acepta separadores: coma, ;, /, |, espacios múltiples.
    """
    if not alt_raw: return []
    parts = re.split(r"[;,/|]+|\s{2,}", alt_raw.upper())
    out = []
    for p in parts:
        p = _norm_ref(p)
        if not p: 
            continue
        # Debe tener algún dígito y longitud mínima
        if any(ch.isdigit() for ch in p) and len(p) >= 3:
            out.append(p)
    return sorted(set(out))

def build_index(insumo_dir=INSUMO_DIR, db_path=DB_PATH, collection=COLLECTION, force_drop: bool=False) -> int:
    """
    Construye/actualiza el índice Chroma leyendo los archivos de `insumo/`.
    Devuelve cuántas filas nuevas añadió. Si `force_drop=True`, borra y rehace.
    Usa los mismos helpers y el cliente OpenAI/chroma definidos arriba.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno.")

    # Cliente/colección (por si se llama desde la app)
    ch_local = chromadb.PersistentClient(path=db_path)
    col_local = ch_local.get_or_create_collection(
        name=collection,
        metadata={"hnsw:space": "cosine"}
    )
    if force_drop:
        try:
            col_local.delete(where={})
        except Exception:
            pass

    if not os.path.isdir(insumo_dir):
        print(f"⚠️ No existe la carpeta '{insumo_dir}/'.")
        return 0

    files = [
        os.path.join(insumo_dir, f)
        for f in os.listdir(insumo_dir)
        if os.path.isfile(os.path.join(insumo_dir, f))
        and f.lower().endswith((".xlsx", ".xls", ".csv", ".txt"))
    ]
    if not files:
        print(f"⚠️ No hay archivos tabulares en '{insumo_dir}/'.")
        return 0

    total_added = 0
    for path in files:
        df = read_table(path)
        if df.empty:
            print(f"⚠️ Vacío: {path}")
            continue

        df.columns = [c.strip().upper() for c in df.columns]
        docs, ids, metas = [], [], []

        for i, row in df.iterrows():
            ref_raw = row.get("REFERENCIA","")
            ref     = (extract_ref(ref_raw) or "").upper()
            alt_raw = norm(row.get("REFERENCIAS ALTERNAS","")).upper()
            aliases = split_aliases(alt_raw)

            nombre = norm(row.get("NOMBRE",""))
            desc   = norm(row.get("DESCRIPCION",""))
            marca  = norm(row.get("MARCA",""))
            linea  = norm(row.get("LINEA",""))
            subl   = norm(row.get("SUB-LINEA",""))
            clas   = norm(row.get("CLASIFICACION",""))
            inv    = norm(row.get("INVENTARIO","")).replace(".", "").replace(",", ".")
            precio = norm(row.get("PRECIO LISTA","")).replace(".", "").replace(",", ".")

            meta_raw = {
                "source": os.path.basename(path),
                "row": int(i),
                "ref": ref,
                "nombre": nombre,
                "descripcion": desc,
                "marca": marca,
                "linea": linea,
                "sublinea": subl,
                "clasificacion": clas,
                "inventario": inv,
                "precio_lista": precio,
                "alt_raw": alt_raw,
                "aliases": aliases,  # clean_meta la serializa a string
            }
            docs.append(to_doc(row))
            ids.append(id_for(ref, int(i), path))
            metas.append(clean_meta(meta_raw))

        keep_idx = mask_missing(col_local, ids)
        if not keep_idx:
            print(f"✅ {os.path.basename(path)} ya estaba indexado.")
            continue

        print(f"{os.path.basename(path)}: indexando {len(keep_idx)} filas nuevas…")
        for s in range(0, len(keep_idx), BATCH_SIZE):
            idxs = keep_idx[s:s+BATCH_SIZE]
            resp = client.embeddings.create(model=EMBED_MODEL, input=[docs[i] for i in idxs])
            embs = [d.embedding for d in resp.data]
            col_local.add(
                ids=[ids[i] for i in idxs],
                documents=[docs[i] for i in idxs],
                embeddings=embs,
                metadatas=[metas[i] for i in idxs],
            )
            total_added += len(idxs)

    print(f"✅ Ingesta completa. Filas nuevas añadidas: {total_added}")
    return total_added



# =========================
# Ingesta
# =========================
if __name__ == "__main__":
    os.makedirs(INSUMO_DIR, exist_ok=True)
    files = [
        os.path.join(INSUMO_DIR, f)
        for f in os.listdir(INSUMO_DIR)
        if os.path.isfile(os.path.join(INSUMO_DIR, f))
        and f.lower().endswith((".xlsx", ".xls", ".csv", ".txt"))
    ]
    if not files:
        print("⚠️ No hay archivos tabulares en 'insumo/'.")
        raise SystemExit(0)

    total_added = 0
    for path in files:
        df = read_table(path)
        if df.empty:
            print(f"⚠️ Vacío: {path}")
            continue

        # Normaliza nombres de columnas a MAYÚSCULAS
        df.columns = [c.strip().upper() for c in df.columns]

        docs, ids, metas = [], [], []

        for i, row in df.iterrows():
            # Campos base (todos normalizados a string)
            ref_raw = row.get("REFERENCIA","")
            ref     = extract_ref(ref_raw) or ""     # evitar None
            alt_raw = norm(row.get("REFERENCIAS ALTERNAS","")).upper()
            aliases = split_aliases(alt_raw)         # 👈 lista de alias normalizados

            nombre = norm(row.get("NOMBRE",""))
            desc   = norm(row.get("DESCRIPCION",""))
            marca  = norm(row.get("MARCA",""))
            linea  = norm(row.get("LINEA",""))
            subl   = norm(row.get("SUB-LINEA",""))
            clas   = norm(row.get("CLASIFICACION",""))
            inv    = norm(row.get("INVENTARIO","")).replace(".", "").replace(",", ".")
            precio = norm(row.get("PRECIO LISTA","")).replace(".", "").replace(",", ".")

            meta_raw = {
                "source": os.path.basename(path),
                "row": int(i),
                "ref": ref.upper(),            # canónica
                "nombre": nombre,
                "descripcion": desc,
                "marca": marca,
                "linea": linea,
                "sublinea": subl,
                "clasificacion": clas,
                "inventario": inv,
                "precio_lista": precio,
                "alt_raw": alt_raw,
                "aliases": aliases,            # 👈 importante para resolver variantes
            }
            meta = clean_meta(meta_raw)

            docs.append(to_doc(row))
            ids.append(id_for(ref, int(i), path))
            metas.append(meta)

        # Reanudar: solo lo que falte
        keep_idx = mask_missing(col, ids)
        if not keep_idx:
            print(f"✅ {os.path.basename(path)} ya estaba indexado.")
            continue

        # Embeddings en lotes
        print(f"{os.path.basename(path)}: indexando {len(keep_idx)} filas nuevas…")
        for s in tqdm(range(0, len(keep_idx), BATCH_SIZE), desc=os.path.basename(path)):
            idxs = keep_idx[s:s+BATCH_SIZE]
            batch_docs = [docs[i]  for i in idxs]
            batch_ids  = [ids[i]   for i in idxs]
            batch_meta = [metas[i] for i in idxs]

            resp = client.embeddings.create(model=EMBED_MODEL, input=batch_docs)
            embs = [d.embedding for d in resp.data]

            col.add(ids=batch_ids, documents=batch_docs, embeddings=embs, metadatas=batch_meta)
            total_added += len(batch_ids)

    print(f"✅ Ingesta completa. Filas nuevas añadidas: {total_added}")



