# =============================================================
# Nombre del archivo: build_del.py
#
# Descripción:
#   DEL (días en leche) de cada foto de Dataset_Fr_BCS a partir de
#   GeneralidadesOrdeno (exportes diarios del robot, "Generalidades
#   Ordeño_AAAAMMDD *.csv", separados por ';', latin1).
#   - Vaca: nombre de la carpeta <vacaId>_<BCS> (data.py).
#   - Fecha de la foto: nombre del archivo (ByIDxx__AAAA-MM-DD-hh-mm-ss_...).
#   - "Días en ordeño" corresponde a la fecha del archivo, así que
#     parto = fecha del archivo más cercano a la foto - DEL y
#     DEL de la foto = fecha de la foto - parto.
#   - Sin DEL (vacío) si el archivo más cercano está a más de 7 días o la
#     foto es anterior al parto.
#   Misma lógica que Support Code/build_dataset_fr_del.py de la rama Adri.
#
# Uso (desde Modeling/Main model/BCS-Regions):
#   docker compose run --rm --entrypoint python bcs del/build_del.py
# Salida: <work>/del.csv
# =============================================================
import argparse, csv, re, sys
import datetime as dt
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import load_records

MAX_GAP = 7  # días máximos entre la foto y el archivo de Generalidades usado

def read_generalidades(ordeno_dir):
    """vacaId -> [(fecha del archivo, DEL, lactación, grupo)] ordenado por fecha."""
    gen = defaultdict(list); files = sorted(Path(ordeno_dir).glob("*.csv"))
    if not files: raise SystemExit(f"No hay CSVs en {ordeno_dir}")
    for f in files:
        d = dt.datetime.strptime(re.search(r"_(\d{8})", f.name).group(1), "%Y%m%d").date()
        for r in csv.DictReader(open(f, encoding="latin1"), delimiter=";"):
            if r["Días en ordeño"].strip().isdigit():
                gen[r["Número del animal"]].append((d, int(r["Días en ordeño"]), r["Nº Lactación"], r["Número del grupo"]))
    print(f"[del] {len(files)} archivos de Generalidades, {len(gen)} vacas")
    return {k: sorted(v) for k, v in gen.items()}

def main(a):
    gen = read_generalidades(a.ordeno_dir); rows = []
    for r in load_records(a.data_dir):
        stem = Path(r["path"]).stem.split("__")[-1]
        taken = dt.datetime.strptime(stem[:19], "%Y-%m-%d-%H-%M-%S")
        row = {"foto": r["name"], "vaca": r["vaca"], "fechaFoto": taken.isoformat(sep=" "), "DEL": "", "fechaParto": "",
               "lactacion": "", "grupo": "", "archivoGeneralidades": "", "diasFotoArchivo": "", "motivo": ""}
        recs = gen.get(r["vaca"], [])
        if not recs:
            row["motivo"] = "vaca sin registros en GeneralidadesOrdeno"
        else:
            d, dim, lact, grp = min(recs, key=lambda x: abs((x[0] - taken.date()).days))
            gap = (taken.date() - d).days; parto = d - dt.timedelta(days=dim)
            row.update(fechaParto=parto.isoformat(), lactacion=lact, grupo=grp, archivoGeneralidades=d.isoformat(), diasFotoArchivo=gap)
            if abs(gap) > MAX_GAP: row["motivo"] = f"archivo de Generalidades más cercano a {gap} días"
            elif taken.date() < parto: row["motivo"] = "foto antes del parto (lactación anterior)"
            else: row["DEL"] = (taken.date() - parto).days
        rows.append(row)
    out = Path(a.work_dir) / "del.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sin = [r for r in rows if r["DEL"] == ""]
    print(f"[del] {len(rows) - len(sin)}/{len(rows)} fotos con DEL -> {out}")
    for r in sin: print(f"  sin DEL: {r['foto']} ({r['motivo']})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS")
    ap.add_argument("--ordeno_dir", default="/ordeno", help="carpeta GeneralidadesOrdeno")
    ap.add_argument("--work_dir", default="/work")
    main(ap.parse_args())
