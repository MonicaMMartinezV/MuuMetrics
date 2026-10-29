# =============================================================
# Nombre del archivo: build_dataset_fr_del.py
# Descripción: Arma el CSV de ID + DEL corregidos a partir de Dataset_Fr
#   (fotos de ByID con el ID del robot confirmado a mano), en reemplazo
#   de dataset_del.csv, que ligó cada foto con la hora de INICIO de ordeño
#   más cercana y suele quedarse con la vaca que entra detrás.
#
#   - vacaId: carpeta de Dataset_Fr (<ID>/ByIDxx__<foto>.jpg).
#   - DEL: "Días en ordeño" de GeneralidadesOrdeno. A diferencia de
#     "Ordeños Ayer" / "Producción de leche ayer" (archivo D -> día D-2),
#     el DEL corresponde a la fecha del archivo (8771 sale con DEL 1 en el
#     archivo del 15/07 y su primer ordeño en el robot es el 14/07).
#     Parto = fecha del archivo más cercano a la foto - DEL; DEL de la
#     foto = fecha de la foto - parto.
#   - BCS: carpeta de Dataset/Images (<BCS>/<foto>.jpg), con el BCS de
#     dataset_del.csv como referencia.
#
#   Las columnas vacaId, ImgName, BCS, DEL y valido son las que lee
#   filmModel.py (load_records), así que el CSV sirve tal cual con --csv.
#
# Uso:
#   python3 build_dataset_fr_del.py --fr_dir Dataset/Dataset_Fr --ordeno_dir GeneralidadesOrdeno \
#       --images_dir Dataset/Images --old_csv dataset_del.csv --out Dataset/dataset_fr_del.csv
#
#   Con --bcs_from_dirname, <fr_dir> es Dataset_Fr_BCS (<ID>_<BCS>/ByIDxx__<foto>.jpg):
#   vacaId y BCS corregido salen del nombre de la carpeta ("<ID>_MAYO4.25_JUNIO4"
#   = BCS por mes de la foto; carpetas ilegibles se marcan no válidas).
#   python3 build_dataset_fr_del.py --fr_dir Dataset/Dataset_Fr_BCS --bcs_from_dirname \
#       --ordeno_dir GeneralidadesOrdeno --images_dir Dataset/Images --old_csv dataset_del.csv \
#       --out Dataset/dataset_frbcs_del.csv
# =============================================================
import argparse, csv, re
import datetime as dt
from collections import defaultdict
from pathlib import Path

MAX_GAP = 7  # días máximos entre la foto y el archivo de Generalidades usado


def read_generalidades(ordeno_dir):
    """ID -> [(fecha archivo, DEL, lactación, grupo)] ordenado por fecha."""
    gen = defaultdict(list)
    for f in Path(ordeno_dir).glob("*.csv"):
        d = dt.datetime.strptime(re.search(r"_(\d{8})", f.name).group(1), "%Y%m%d").date()
        for r in csv.DictReader(open(f, encoding="latin1"), delimiter=";"):
            if r["Días en ordeño"].strip().isdigit():
                gen[r["Número del animal"]].append((d, int(r["Días en ordeño"]), r["Nº Lactación"], r["Número del grupo"]))
    return {k: sorted(v) for k, v in gen.items()}


def bcs_from_images(images_dir):
    out = {}
    for p in Path(images_dir).rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            try:
                out.setdefault(p.stem, float(p.parent.name))
            except ValueError:
                pass
    return out


MONTHS = {"ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4, "MAYO": 5, "JUNIO": 6, "JULIO": 7,
          "AGOSTO": 8, "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12}


def cow_bcs_from_dirname(dirname, month):
    """'1204_4.5' -> ('1204', 4.5); '8758_MAYO4.25_JUNIO4' -> BCS del mes; ilegible -> (id, None)."""
    cow, _, spec = dirname.partition("_")
    by_month = {}
    for tok in spec.split("_"):
        name = tok.rstrip("0123456789.")
        if name.upper() in MONTHS:
            by_month[MONTHS[name.upper()]] = float(tok[len(name):])
    if by_month:
        return cow, by_month.get(month)
    try:
        return cow, float(spec)
    except ValueError:
        return cow, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fr_dir", required=True)
    ap.add_argument("--ordeno_dir", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--old_csv", required=True, help="dataset_del.csv (solo para comparar)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bcs_from_dirname", action="store_true",
                    help="<fr_dir> con carpetas <ID>_<BCS> (Dataset_Fr_BCS): vacaId y BCS del nombre de la carpeta")
    args = ap.parse_args()

    gen = read_generalidades(args.ordeno_dir)
    bcs_new = bcs_from_images(args.images_dir)
    old = {r["ImgName"]: r for r in csv.DictReader(open(args.old_csv))}

    rows = []
    for p in sorted(Path(args.fr_dir).glob("*/*")):
        cow, (src, name) = p.parent.name, p.name.split("__", 1)
        stem = Path(name).stem
        taken = dt.datetime.strptime(stem[:19], "%Y-%m-%d-%H-%M-%S")
        o = old.get(stem, {})
        bcs = bcs_new.get(stem, o.get("BCS", ""))
        if args.bcs_from_dirname:
            cow, bcs = cow_bcs_from_dirname(cow, taken.month)
            bcs = "" if bcs is None else bcs
        row = {"vacaId": cow, "ImgName": stem, "BCS": bcs,
               "DEL": "", "fechaParto": "", "lactacion": "", "grupo": "", "archivoGeneralidades": "",
               "diasFotoArchivo": "", "fechaFoto": taken.isoformat(sep=" "), "carpetaByID": src.replace("ByID", "", 1).strip("_"),
               "BCS_dataset_del": o.get("BCS", ""), "vacaId_dataset_del": o.get("vacaId", ""),
               "DEL_dataset_del": o.get("DEL", ""), "motivo": ""}
        recs = gen.get(cow, [])
        if not recs:
            row["motivo"] = "ID sin registros en GeneralidadesOrdeno"
        else:
            d, dim, lact, grp = min(recs, key=lambda r: abs((r[0] - taken.date()).days))
            gap = (taken.date() - d).days
            parto = d - dt.timedelta(days=dim)
            row.update(fechaParto=parto.isoformat(), lactacion=lact, grupo=grp, archivoGeneralidades=d.isoformat(),
                       diasFotoArchivo=gap)
            if abs(gap) > MAX_GAP:
                row["motivo"] = f"archivo de Generalidades más cercano a {gap} días"
            elif taken.date() < parto:
                row["motivo"] = "foto antes del parto (lactación anterior)"
            else:
                row["DEL"] = (taken.date() - parto).days
        if row["BCS"] == "":
            row["motivo"] = "; ".join(filter(None, [row["motivo"], "sin BCS"]))
        row["valido"] = not row["motivo"]
        rows.append(row)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["valido"]]
    same_id = sum(r["vacaId"] == r["vacaId_dataset_del"] for r in rows)
    del_diff = [abs(int(r["DEL"]) - int(float(r["DEL_dataset_del"]))) for r in ok if r["DEL_dataset_del"] not in ("", "nan")]
    bcs_diff = sum(r["BCS_dataset_del"] != "" and float(r["BCS"]) != float(r["BCS_dataset_del"]) for r in rows if r["BCS"] != "")
    print(f"Fotos: {len(rows)} | válidas: {len(ok)} | vacas: {len({r['vacaId'] for r in ok})}")
    print(f"ID igual al de dataset_del: {same_id}/{len(rows)}")
    if del_diff:
        print(f"DEL vs dataset_del (válidas): igual {sum(x == 0 for x in del_diff)}, |dif| > 7 días {sum(x > 7 for x in del_diff)}, "
              f"máx {max(del_diff)}")
    print(f"BCS de Images distinto al de dataset_del: {bcs_diff}")
    for r in rows:
        if not r["valido"]:
            print("  no válida:", r["vacaId"], r["ImgName"], r["motivo"])
    print(f"Listo: {args.out}")


if __name__ == "__main__":
    main()
