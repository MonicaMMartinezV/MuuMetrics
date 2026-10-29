# =============================================================
# Nombre del archivo: match_byid_robot_visits.py
# Descripción: Liga las fotos de ByID con el número de animal del robot
#   usando el registro de ordeños con hora (combined_four_digit_files.csv).
#
#   1. combined_four_digit_files.csv son los reportes por vaca concatenados
#      (un bloque por vaca, cada uno con su encabezado), sin el número de
#      animal. Se recupera comparando la producción diaria de cada bloque
#      con "Producción de leche ayer" de GeneralidadesOrdeno: el archivo
#      con fecha D reporta el día D-2 (verificado: coinciden al kg).
#   2. Las cámaras (cam0 y cam4) están a la salida del VMS 3: las fotos
#      caen 0-2 min después de que TERMINA un ordeño en VMS 3 (inicio +
#      duración) y nunca durante el ordeño. Cada ráfaga de fotos (< 90 s
#      entre fotos) se asigna al ordeño de VMS 3 que terminó justo antes.
#   3. Solo hay registro de ~34 vacas, así que muchas ráfagas no tienen
#      ordeño conocido. Se reporta la tasa de coincidencias por azar
#      (misma ventana desplazada) para saber qué tanto confiar.
#
#   Salida en <out_dir>:
#     - bloques_robot.csv: bloque -> ID de animal y calidad del cruce
#     - rafagas.csv: cada ráfaga de ByID con el ordeño asignado (si hay)
#     - vacas_byid.csv: por vaca de ByID, IDs asignados y su frecuencia
#     - hojas/robot_<ID>.jpg: fotos de ByID asignadas a cada ID + CAETEC
#     - agrupado/{seguros,dudosos,sin_match}/: todas las fotos de ByID
#   Con --proposals (propuestas.csv de reid_head.py) se suman las carpetas
#   que la re-identificación por apariencia sube a seguros/dudosos (usa
#   "Nivel final" si se llenó a mano; si no, "Nivel propuesto").
#
# Uso (en muumetrics-convnext, por PIL):
#   python match_byid_robot_visits.py --visits /data/combined_four_digit_files.csv \
#       --ordeno_dir /data/GeneralidadesOrdeno --byid_dir /data/ByID \
#       --caetec_dir /data/opencowid/caetec_crops_448 --out_dir /data/robot_visits
# =============================================================

import argparse, bisect, csv, re, shutil
import datetime as dt
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

GEN_LAG = dt.timedelta(days=2)   # archivo Generalidades del día D -> producción del día D-2
WINDOW = (0, 150)                # segundos entre fin de ordeño y foto
BURST_GAP = 90                   # segundos entre fotos de una misma ráfaga


def parse_time(s):
    return dt.datetime.strptime(s.replace("a. m.", "AM").replace("p. m.", "PM"), "%d/%m/%Y %I:%M %p")


def read_blocks(path):
    blocks, cur = [], None
    for r in list(csv.reader(open(path, encoding="utf-8")))[1:]:
        if r[0] == "Hora de inicio":
            cur = []
            blocks.append(cur)
        else:
            cur.append(r)
    return blocks


def read_generalidades(ordeno_dir):
    gen = defaultdict(dict)
    for f in Path(ordeno_dir).glob("*.csv"):
        day = dt.datetime.strptime(re.search(r"_(\d{8})", f.name).group(1), "%Y%m%d").date() - GEN_LAG
        for r in csv.DictReader(open(f, encoding="latin1"), delimiter=";"):
            try:
                gen[r["Número del animal"]][day] = float(r["Producción de leche ayer"].replace(",", "."))
            except (ValueError, KeyError):
                pass
    return gen


def identify_blocks(blocks, gen):
    """Para cada bloque: (ID, fracción de días que coinciden < 0.3 kg, días comparados, 2º ID, su fracción)."""
    out = []
    for rows in blocks:
        prod = defaultdict(float)
        for r in rows:
            if r[1] == "Ordeño":
                prod[parse_time(r[0]).date()] += float(r[3] or 0)
        scores = []
        for animal, g in gen.items():
            days = [d for d in prod if d in g]
            if len(days) >= 3:
                scores.append((sum(abs(prod[d] - g[d]) < 0.3 for d in days) / len(days), len(days), animal))
        scores.sort(reverse=True)
        best = scores[0] if scores else (0, 0, "")
        second = scores[1] if len(scores) > 1 else (0, 0, "")
        out.append({"id": best[2], "frac": best[0], "dias": best[1], "id2": second[2], "frac2": second[0]})
    return out


def read_visits(blocks, ids, min_frac):
    visits = []
    for rows, info in zip(blocks, ids):
        if info["frac"] < min_frac:
            continue
        for r in rows:
            vms = next((x for x in r if x.startswith("VMS")), None)
            if r[1] != "Ordeño" or not vms or not r[2]:
                continue
            m, s = r[2].split(":")
            start = parse_time(r[0])
            visits.append({"id": info["id"], "vms": vms, "inicio": start,
                           "fin": start + dt.timedelta(minutes=int(m), seconds=int(s))})
    return visits


def read_bursts(byid_dir):
    by_cow = defaultdict(list)
    for p in Path(byid_dir).rglob("*.jpg"):  # incluye subcarpetas (p. ej. "No encontre grupo/duda blanca")
        m = re.match(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(cam\d)", p.name)
        if m and " - Copy" not in p.name:
            by_cow[p.parent.relative_to(byid_dir).as_posix()].append((dt.datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M-%S"), m.group(2), p))
    bursts = []
    for cow, fs in by_cow.items():
        fs.sort()
        cur = [fs[0]]
        for f in fs[1:]:
            if (f[0] - cur[-1][0]).total_seconds() < BURST_GAP:
                cur.append(f)
            else:
                bursts.append((cow, cur))
                cur = [f]
        bursts.append((cow, cur))
    return bursts


def assign(burst, ends, end_ids, shift=0):
    """Ordeño de VMS 3 que terminó justo antes de alguna foto de la ráfaga (el más cercano)."""
    best = None
    for t, _, _ in burst:
        t = t - dt.timedelta(seconds=shift)
        i = bisect.bisect_left(ends, t - dt.timedelta(seconds=WINDOW[1]))
        while i < len(ends) and ends[i] <= t - dt.timedelta(seconds=WINDOW[0]):
            d = (t - ends[i]).total_seconds()
            if best is None or d < best[0]:
                best = (d, end_ids[i])
            i += 1
    return best


def sheet(rid, hits, caetec_dir, path, W=300, H=260, cols=4):
    cae = sorted((Path(caetec_dir) / rid).glob("*.jpg"))[:4] if caetec_dir else []
    items = [(b[0][2], f"ByID {cow}  {b[0][0]:%m-%d %H:%M} {b[0][1]} +{d / 60:.1f}m", "red") for cow, b, d in hits]
    items += [(p, f"CAETEC {rid}", "blue") for p in cae]
    sh = Image.new("RGB", (cols * W, ((len(items) + cols - 1) // cols) * H + 30), "white")
    dr = ImageDraw.Draw(sh)
    dr.text((5, 5), f"ID robot {rid}: rafagas de ByID tras su ordeño en VMS 3 (rojo) y fotos CAETEC (azul)", fill="black")
    for k, (p, label, color) in enumerate(items):
        im = Image.open(p).convert("RGB")
        im.thumbnail((W - 8, H - 24))
        x, y = (k % cols) * W, (k // cols) * H + 30
        sh.paste(im, (x + 4, y + 4))
        dr.rectangle([x + 2, y + 2, x + W - 3, y + H - 3], outline=color, width=2)
        dr.text((x + 6, y + H - 18), label, fill="black")
    sh.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visits", required=True)
    ap.add_argument("--ordeno_dir", required=True)
    ap.add_argument("--byid_dir", required=True)
    ap.add_argument("--caetec_dir", default=None)
    ap.add_argument("--proposals", default=None, help="propuestas.csv de reid_head.py (apariencia)")
    ap.add_argument("--min_frac", type=float, default=0.75, help="Fracción mínima de días que coinciden para aceptar el ID de un bloque")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    shutil.rmtree(out / "hojas", ignore_errors=True)
    (out / "hojas").mkdir(parents=True)

    blocks = read_blocks(args.visits)
    ids = identify_blocks(blocks, read_generalidades(args.ordeno_dir))
    with open(out / "bloques_robot.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Bloque", "Filas", "ID animal", "Días que coinciden (%)", "Días comparados", "2º ID", "2º (%)", "Usado"])
        for k, (rows, i) in enumerate(zip(blocks, ids)):
            w.writerow([k, len(rows), i["id"], round(100 * i["frac"]), i["dias"], i["id2"], round(100 * i["frac2"]),
                        "sí" if i["frac"] >= args.min_frac else "no"])

    visits = read_visits(blocks, ids, args.min_frac)
    v3 = sorted((v["fin"], v["id"]) for v in visits if v["vms"] == "VMS 3")
    ends, end_ids = [t for t, _ in v3], [i for _, i in v3]
    bursts = read_bursts(args.byid_dir)

    # Coincidencias esperadas por azar: misma ventana desplazada lejos del pico
    shifts = [-1200, -900, -600, 600, 900, 1200]
    real = sum(assign(b, ends, end_ids) is not None for _, b in bursts)
    chance = sum(sum(assign(b, ends, end_ids, s) is not None for _, b in bursts) for s in shifts) / len(shifts)
    print(f"Bloques con ID: {sum(i['frac'] >= args.min_frac for i in ids)}/{len(blocks)} | "
          f"ráfagas ByID: {len(bursts)} | con ordeño asignado: {real} (por azar se esperan ~{chance:.0f})")

    hits = defaultdict(list)
    assigned = {}
    per_cow = defaultdict(Counter)
    n_bursts = Counter()
    with open(out / "rafagas.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID ByID", "Primera foto", "Fecha/hora", "Cámara", "Fotos", "ID robot asignado", "Foto − fin ordeño (min)"])
        for cow, b in sorted(bursts, key=lambda x: x[1][0][0]):
            a = assign(b, ends, end_ids)
            n_bursts[cow] += 1
            if a:
                assigned[id(b)] = a
                hits[a[1]].append((cow, b, a[0]))
                per_cow[cow][a[1]] += 1
            w.writerow([cow, b[0][2].name, f"{b[0][0]:%Y-%m-%d %H:%M:%S}", b[0][1], len(b),
                        a[1] if a else "", round(a[0] / 60, 2) if a else ""])

    with open(out / "vacas_byid.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID ByID", "Ráfagas", "Ráfagas con ordeño asignado", "ID robot más frecuente", "Ráfagas con ese ID",
                    "Otros IDs (ráfagas)", "ID robot confirmado", "Notas"])
        key = lambda c: (0, int(c)) if c.isdigit() else (1, c)
        for cow in sorted(n_bursts, key=key):
            top = per_cow[cow].most_common()
            w.writerow([cow, n_bursts[cow], sum(per_cow[cow].values()), top[0][0] if top else "",
                        top[0][1] if top else "", "; ".join(f"{k}: {n}" for k, n in top[1:]), "", ""])

    for rid, hs in hits.items():
        sheet(rid, sorted(hs, key=lambda h: h[1][0][0]), args.caetec_dir, out / "hojas" / f"robot_{rid}.jpg")
    group_by_robot(bursts, {i: assigned[id(b)] for i, (_, b) in enumerate(bursts) if id(b) in assigned},
                   per_cow, args.byid_dir, out / "agrupado", load_proposals(args.proposals))
    print(f"Listo: {out}")


def load_proposals(path):
    """propuestas.csv de reid_head.py -> {carpeta ByID: (nivel, ID robot, motivo)}."""
    out = {}
    if not path:
        return out
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        level = r["Nivel final (a mano: seguros/dudosos/no)"].strip() or r["Nivel propuesto"]
        if level in ("seguros", "dudosos") and r["ID propuesto"]:
            out[r["ID ByID"]] = (level, r["ID propuesto"], r["Motivo"] + (f"; {r['Notas']}" if r["Notas"] else ""))
    return out


def group_by_robot(bursts, assigned, per_cow, byid_dir, dst, proposals=None):
    """Copia TODAS las fotos de ByID a <dst>/{seguros,dudosos,sin_match}/:
      - seguros/<ID robot>: la carpeta de ByID tiene un único ID que cae en >= 2
        ráfagas (días u horas distintas; por azar es muy improbable). Se copia
        la carpeta completa, salvo ráfagas asignadas a otro ID (van a dudosos).
        En carpetas con >= 2 IDs seguros (mezclan vacas) solo van las ráfagas
        con hora; el resto va a dudosos/ByIDxx_<ID>_o_<ID>.
      - dudosos/<ID robot>: carpeta con una sola ráfaga ligada a un único ID
        (se copia completa), o ráfagas sueltas que no cuadran con su carpeta.
      - sin_match/ByIDxx: carpetas sin ningún ordeño conocido.
      Las carpetas en `proposals` (apariencia) van completas a su nivel/ID propuesto.
    "No encontre grupo" no es una vaca: solo se agrupan sus ráfagas con hora."""
    shutil.rmtree(dst, ignore_errors=True)
    burst_of = {p: i for i, (_, b) in enumerate(bursts) for _, _, p in b}
    rows = []
    for folder in sorted({p.parent for p in Path(byid_dir).rglob("*.jpg")}):
        cow = folder.relative_to(byid_dir).as_posix()
        tag = f"ByID{int(cow):02d}" if cow.isdigit() else "ByID_" + re.sub(r"\W+", "_", cow).strip("_")
        sure = sorted(r for r, n in per_cow[cow].items() if n >= 2)
        prop = (proposals or {}).get(cow) if not sure else None
        if prop:
            default = prop[:2]
        elif not cow.isdigit():
            default = ("sin_match", tag)
        elif len(sure) == 1:
            default = ("seguros", sure[0])
        elif len(sure) > 1:
            default = ("dudosos", f"{tag}_" + "_o_".join(sure))
        elif len(per_cow[cow]) == 1:
            default = ("dudosos", next(iter(per_cow[cow])))
        elif per_cow[cow]:
            default = ("dudosos", f"{tag}_" + "_o_".join(sorted(per_cow[cow])))
        else:
            default = ("sin_match", tag)
        for p in sorted(folder.glob("*.jpg")):
            if " - Copy" in p.name:
                continue
            a = assigned.get(burst_of.get(p))
            if a is None:
                level, rid, why = *default, f"carpeta; {prop[2]}" if prop else "carpeta"
            elif prop and a[1] == prop[1]:
                level, rid, why = prop[0], a[1], f"ráfaga con hora; {prop[2]}"
            elif per_cow[cow][a[1]] >= 2:
                level, rid, why = "seguros", a[1], "ráfaga con hora (repetida)"
            elif default == ("dudosos", a[1]):
                level, rid, why = "dudosos", a[1], "ráfaga con hora (única)"
            else:
                level, rid, why = "dudosos", a[1], "ráfaga con hora (no cuadra con su carpeta)"
            out = dst / level / rid
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out / f"{tag}__{p.name}")
            rows.append([level, rid, cow, p.name, why, round(a[0] / 60, 2) if a else ""])
    with open(dst / "resumen.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Nivel", "Carpeta destino", "ID ByID", "Imagen", "Motivo", "Foto − fin ordeño (min)"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
