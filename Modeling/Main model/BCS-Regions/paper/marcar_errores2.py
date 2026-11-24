"""PaperDINOcow.pdf (versión 19-nov 21:16, 45 pp.): marca en rojo cada error con etiqueta en el margen derecho."""
import sys
import pymupdf

src, dst = sys.argv[1], sys.argv[2]
d = pymupdf.open(src)
ROJO, ROSA, NAR, NARF = (0.85, 0, 0), (1, 0.82, 0.82), (0.9, 0.45, 0), (1, 0.9, 0.75)

# (página, texto a buscar, etiqueta, ocurrencia, tipo)  tipo R = error, F = referencia a figura/tabla
ERR = [
 (1, "Bárbara P. Alcántara-Vega", "Missing affiliation superscript", 0, "E"),
 (1, "José Soto Castro, ", "Stray '1' before 'and'; also Vásquez here vs Vázquez in contributions, María vs Maria", 0, "E"),
 (1, "in segments", "'... decoder segments six regions'", 0, "E"),
 (1, "a MAE 0.415", "'an MAE of 0.415'", 0, "E"),
 (2, "AMR = antimicrobial resistance", "Template example: list the real abbreviations", 0, "E"),
 (4, "continuous BCS estimation from 2D side-view", "Missing verb: 'Yao et al. (2026a) addressed continuous ...'", 0, "E"),
 (10, "before monitoring as a non-contact approach before", "Garbled: 'Pseudo-labels have been used before for non-contact cattle monitoring (Qiao et al., 2022)'", 0, "E"),
 (11, "al. (1994b)", "Ferguson et al. (1994): 1994a and 1994b are the same paper", 0, "E"),
 (11, "average difference", "'mean absolute difference' (drop 'average')", 0, "E"),
 (11, "Ferguson et al. (1994b)", "-> Ferguson et al. (1994)", 0, "E"),
 (12, "Image augmentation", "DINOcow uses no augmentation and the baselines use other settings: remove or state which experiment this is", 0, "E"),
 (16, "registering tokens", "'register tokens'", 0, "E"),
 (16, "pool the feature mask", "'pool the feature map'", 0, "E"),
 (17, "Three details complete with the descriptor", "'Three details complete the descriptor'", 0, "E"),
 (17, "and on all 76 cows", "Add '(Supplementary Figure S3)'", 0, "F"),
 (18, "clipped it", "'clipped'", 0, "E"),
 (18, "Dinov2-B/14", "DINOv2-B/14", 0, "E"),
 (18, "Training used mixed precision and fixed random seeds.", "MISSING in Methods: validation schemes (LOCO, 5-fold x 10), bootstrap CIs, nested selection, learning curve, baselines (all written in LaTeX)", 0, "E"),
 (19, "Predicting performance on unseen cows", "'Predictive performance on unseen cows'", 0, "E"),
 (19, "And the repeated five-fold", "Do not start with 'And'", 0, "E"),
 (20, "as seen in Figure 9", "-> Figure 6", 0, "F"),
 (20, "herd's", "'herd mean'", 0, "E"),
 (20, "slope of predicted reference", "'slope of predicted against reference score'", 0, "E"),
 (21, "crop methods, MAE per cow by 0.111", "Missing verb: '... raised the MAE per cow by 0.111'", 0, "E"),
 (21, "(95% CI 0.010", "Delete: old CI, duplicates '(CI 0.008-0.057)'", 0, "E"),
 (21, "of Table 7), and adding", "-> Table 8 (the 15 folds are those of Table 8)", 0, "F"),
 (22, "s with the head fixed", "s -> so", 0, "E"),
 (22, "Region descriptor against simpler descriptors from the same frozen tokens.", "Wrong caption (copy of Table 7). Use: Comparison with the baselines on unseen cows (five cow-level folds x 3 seeds)", 0, "E"),
 (22, "training cows", "Missing footnote mark c on this row", 0, "E"),
 (23, "for seed 0, where the test curves", "Add '(Supplementary Figure S1)'", 0, "F"),
 (23, "as shown in Figure 10", "-> Figure 7 (ROC)", 0, "F"),
 (25, "the form $", "Stray '$' before the formula", 0, "E"),
 (25, "as shown in Figure 11", "-> Figure 8 (learning curve)", 0, "F"),
 (26, "as seen in Figure 12", "-> Supplementary Figure S2 (LoRA vs DINOcow maps, not yet in the document)", 0, "F"),
 (26, "Figure 12 averages", "-> Figure 9", 0, "F"),
 (27, "as seen in Figure 13", "-> Figure 10 (tracking)", 0, "F"),
 (28, "and referenced BCS", "'reference BCS'", 0, "E"),
 (29, "The model achieves applied importance", "DUPLICATE paragraph (lines 489-496 = 566-573): delete this one", 0, "E"),
 (29, "as seen in figure 14", "-> Figure 11", 0, "F"),
 (29, "five manually segmented", "Regions are segmented automatically: 'five automatically segmented'", 0, "E"),
 (29, "the ribs region reduced rather than", "Region is called 'loin' in the paper; '(-2.6%)' unclear: 'MAE per cow 2.6% higher'", 0, "E"),
 (30, "Removing the ribs", "'Removing the loin'", 0, "E"),
 (30, "MAE values between 0.614", "'MAE per cow between 0.614 and 0.713'", 0, "E"),
 (31, "real world", "'real-world'", 0, "E"),
 (31, "slightly more accurate", "Not significant (CI -0.025 to +0.066): 'as accurate ... difference not significant'", 0, "E"),
 (31, "0.94 and 0.90 for DINOcow", "Add 'and 0.94 and 0.91 for LoRA'", 0, "E"),
 (32, "as illustrated in Figure 14", "-> Figure 11", 0, "F"),
 (32, "3D imaging systems often achieve", "Claim needs a citation", 0, "E"),
 (33, "Holstein Cow Lactation Curve.", "Caption too short: say what is plotted and the sources (Lee 2020, Xavier 2022, Souza 2025)", 0, "E"),
 (33, "The following diagram shows", "-> 'Figure 12 shows'", 0, "F"),
 (34, "greatest influence on scoring", "Contradicts Results: hook bones and right thurl largest; tailhead and rump moved less", 0, "E"),
 (36, "main six regions", "Five regions (loin dropped)", 0, "E"),
 (36, "NOTESThis", "Missing line break after NOTES", 0, "E"),
 (36, "conflicts of interest", "Missing final period", 0, "E"),
 (36, "(Lo que hizo cada quien)", "Template text: remove with 'Ejemplos' and the six unnamed lines", 0, "E"),
 (37, "Investigation and Editing", "'Investigation' listed twice", 0, "E"),
 (37, "Methodology, Investigation, Writing", "'Investigation' listed twice", 0, "E"),
 (37, "C. Souza, V., B. Gleason", "Malformed: 'Souza, V. C., Gleason, C. B., Price, T. P., ...'", 0, "E"),
 (38, "Thomsen, N. (1994b)", "Duplicate of 1994a: keep one entry", 0, "E"),
 (39, "Martinet, Jacques.", "Malformed entry: Martinet, J., Houdebine, L.-M., & Head, H. H. (Eds.) (1999). Biology of Lactation. INRA", 0, "E"),
 (39, "Nutrient Requirements of Dairy Cattle. (2021)", "Author is NASEM: 'NASEM (National Academies of Sciences, Engineering, and Medicine). 2021.'", 0, "E"),
 (40, "(2026b). Automated", "Duplicate of 2026a: keep one entry", 0, "E"),
 (40, "in the 37 with known", "'in the 37 cows with known DIM'", 0, "E"),
 (41, "as seen in Figure 6", "-> Figure 13 (French chart)", 0, "F"),
 (42, "used to draw Figure 7", "-> Figure 14 (traffic light)", 0, "F"),
 (42, "reference curve (Figure 7)", "-> (Figure 12a): the 37 cows are plotted there", 0, "F"),
 (42, "Each prediction was then automatically assigned", "DUPLICATE sentence: delete (or add 'automatically' to the previous one)", 0, "E"),
 (42, "Adapted curve based on the live weight study", "Old caption: Traffic-light zones of BCS along the lactation ... (new caption)", 0, "E"),
 (43, "Figure 14 was built in the same way", "-> 'Figure 11 was built ...' and move this paragraph next to Figure 11", 0, "F"),
 (43, "C. Souza et", "'Souza et al. (2025)'", 0, "E"),
 (43, "Requirements of Dairy Cattle (2021), because", "Cite as NASEM (2021)", 0, "E"),
 (43, "Figure 14. Associations among", "Second 'Figure 14': -> Figure 15", 0, "F"),
 (43, "Figure 14 was built in the same way", "DUPLICATE paragraph (lines 869-877): delete", 1, "E"),
 (44, "Mellado et al., 2016", "Missing from the reference list", 0, "E"),
 (44, "Baselines, training behavior and DINOcow error", "Add title 'Supplemental Material' above; S2 and S3 figures missing", 0, "E"),
 (44, "Table 10 extends Table 8", "-> Table S1", 0, "F"),
 (44, "Table 10. Detailed", "-> Table S1", 0, "F"),
 (45, "Figure 8. Training curves", "Second 'Figure 8': -> Supplementary Figure S1", 0, "F"),
]

pend, faltan = [], []
for pg, txt, corto, k, tipo in ERR:
    p = d[pg - 1]; rs = p.search_for(txt)
    if len(rs) <= k: faltan.append((pg, txt)); continue
    r = rs[k]; c, f = (ROJO, ROSA) if tipo == "E" else (NAR, NARF)
    a = p.add_rect_annot(r + (-1, -1, 1, 1)); a.set_colors(stroke=c); a.set_border(width=1.2); a.set_info(content=corto); a.update()
    pend.append((pg - 1, r.y0, ("REF " if tipo == "F" else "") + corto, f))

usados = {}
for n, y, texto, fondo in sorted(pend, key=lambda t: (t[0], t[1])):
    p = d[n]; ancho = 88
    y0 = max(y - 2, usados.get(n, 0) + 2)
    lin = max(1, int(len(texto) * 3.3 / (ancho - 6)) + 1)
    r = pymupdf.Rect(521, y0, 521 + ancho, y0 + 7.6 * lin + 3)
    a = p.add_freetext_annot(r, texto, fontsize=5.5, fontname="helv", text_color=(0, 0, 0), fill_color=fondo, border_width=0.5)
    a.update(); usados[n] = r.y1

L = d.new_page(0, width=612, height=792)
L.insert_text((50, 50), "PaperDINOcow (19 Nov, 21:16, 45 pp.) - errors marked", fontsize=12, fontname="helv")
L.draw_rect(pymupdf.Rect(50, 64, 70, 74), color=None, fill=NARF); L.insert_text((78, 72), "Orange = wrong or missing figure/table reference", fontsize=8.5, fontname="helv")
L.draw_rect(pymupdf.Rect(50, 80, 70, 90), color=None, fill=ROSA); L.insert_text((78, 88), "Pink = text, number or reference-list error", fontsize=8.5, fontname="helv")
nE = sum(1 for e in ERR if e[4] == "E"); nF = len(ERR) - nE
L.insert_text((50, 110), f"{len(ERR)} marks: {nF} references, {nE} other errors.", fontsize=8.5, fontname="helv")
y = 130
for t in ["Correct figure order: 6 agreement, 7 ROC, 8 learning curve, 9 averaged region map, 10 tracking,",
          "11 DMI/milk/live-weight curves, 12 BCS along lactation (37 cows), 13 French chart, 14 traffic light, 15 Martinet.",
          "Supplemental: Table S1, Figure S1 training curves, S2 LoRA vs DINOcow maps (missing), S3 alpha and learning curve (missing)."]:
    L.insert_text((50, y), t, fontsize=8, fontname="helv"); y += 13
d.save(dst, garbage=3, deflate=True)
print("marcas:", len(pend), "| sin ubicar:", faltan)
