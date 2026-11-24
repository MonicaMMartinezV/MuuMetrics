"""PaperDINOcow.pdf (versión 20-nov 16:04, 44 pp.): marca errores, referencias a figuras, puntos de la guía
de JDS (Instructions for Authors 2026) y dónde va la figura nueva de resolución x bloque.
Uso: python3 marcar_errores3.py PaperDINOcow.pdf PaperDINOcow_errores_v3.pdf"""
import sys
import pymupdf

src, dst = sys.argv[1], sys.argv[2]
d = pymupdf.open(src)
TIPOS = {  # borde, fondo, prefijo
    "E": ((0.85, 0, 0), (1, 0.82, 0.82), "ERROR "),
    "F": ((0.9, 0.45, 0), (1, 0.9, 0.75), "REF "),
    "J": ((0.1, 0.55, 0.2), (0.82, 0.95, 0.82), "JDS "),
    "G": ((0, 0.35, 0.8), (0.8, 0.9, 1), "NEW FIG "),
}
# (página, texto, etiqueta, ocurrencia, tipo)
ERR = [
 # --- portada y resumen
 (1, "DINOcow: Body Condition Scoring", "Add a running head (45 characters max., e.g. 'Body condition scoring from 2D rear images') and an interpretive summary (100 words max., no abbreviations) at the top of page 1; a draft is in the LaTeX", 0, "J"),
 (1, "*Corresponding author", "Give full name, mailing address and phone for each corresponding author (max. 2)", 0, "J"),
 (1, "achieved a MAE of", "'an MAE of'", 0, "E"),
 (2, "Nonstandard abbreviations", "List every author-defined abbreviation, alphabetical: AUC, CNN, FPN, IoU, LOCO, LoRA, MAE, mIoU, PIR, QWK, RMSE, ROC, ViT", 0, "J"),
 (2, "INTRODUCTION", "JDS asks for an Introduction under 1.5 pages; this one is about 3", 0, "J"),
 (6, "Diagram of the space", "JDS: tables and figures go in separate sections at the END of the manuscript, not in the text", 0, "J"),
 # --- métodos
 (9, "before monitoring as a non-contact approach before", "Garbled: 'Pseudo-labels have been used before for non-contact cattle monitoring (Qiao et al., 2022)'", 0, "E"),
 (10, "Ferguson et al. (1994), as shown", "OK; but the caption and line 169 still say 1994b", 0, "E"),
 (11, "(1994b)", "-> (1994): 1994a and 1994b are the same paper", 0, "E"),
 (11, "Ferguson et al. (1994b)", "-> Ferguson et al. (1994)", 0, "E"),
 (11, "±0.35", "JDS: '±' means SE. Write 'a mean absolute difference of 0.35 points'", 0, "J"),
 (11, "Image augmentation", "DINOcow uses no augmentation and the baselines use other settings: remove, or say which experiment this is", 0, "E"),
 (14, "labelled", "US spelling (JDS): labeled, leveled, grayscale, center", 0, "J"),
 (15, "registering tokens", "'register tokens'", 0, "E"),
 (15, "pool the feature mask", "'pool the feature map'", 0, "E"),
 (15, "mask value:", "JDS: equations must be editable (MathType), not images", 0, "J"),
 (16, "Three details complete with", "'Three details complete the descriptor'", 0, "E"),
 (18, "Training used mixed precision", "MISSING in Methods: evaluation (LOCO, 5-fold x 10), bootstrap CIs, nested selection, learning curve, baselines and leakage test. JDS needs experimental unit (cow) and statistics described. Text is in the LaTeX", 0, "J"),
 # --- resultados
 (18, "The evaluation on cows the model had never seen", "Broken sentence: 'Evaluated on cows it had never seen, DINOcow estimated ...'", 0, "E"),
 (19, "On the configuration inside each training fold", "Broken: 'Choosing the configuration inside each training fold, without access to the test cows, gave the same error (0.406)'", 0, "E"),
 (19, "improved significantly.", "INSERT here the paragraph on the resolution x block grid and the new Figure 6 (grid_canonico.pdf), right after Table 6. Text in the LaTeX. Figures 6-16 then become 7-17", 0, "G"),
 (19, "greyscale", "grayscale (US spelling)", 0, "J"),
 (19, "herd's", "'herd mean'", 0, "E"),
 (19, "slope of predicted reference", "'slope of predicted against reference score'", 0, "E"),
 (19, "bias toward the mean:", "Avoid the colon: '..., indicating a bias toward the mean, so thin cows ...'", 0, "E"),
 (21, "reached 0.642, yielded", "Grammar: 'reached 0.642 and was no better than the same network fine-tuned'", 0, "E"),
 (21, "Comparison with the baselines on unseen cows.", "JDS: table footnotes numbered (1, 2, 3, 4), one per line; letters are for mean separation. Explain bold in a footnote", 0, "J"),
 (22, "levelled off", "'leveled off' (US spelling)", 0, "J"),
 (22, "for seed 0 in the APPENDIX", "'in the Appendix' (not capitals)", 0, "E"),
 (24, "improved in all", "Wrong meaning: 'where DINOcow was better in all eight folds'", 0, "E"),
 (25, "one-fold", "'in one fold'", 0, "E"),
 (25, "as seen in Figure 12", "Figure 12 is the lactation figure. Delete the reference (the LoRA vs DINOcow maps are not in the paper)", 0, "F"),
 (27, "referenced BCS", "'reference BCS'", 0, "E"),
 # --- discusión
 (28, "manually segmented", "Regions are segmented automatically: 'five automatically segmented regions'", 0, "E"),
 (28, "including the ribs", "The region is called 'loin' in the rest of the paper", 0, "E"),
 (28, "Removing the ribs", "Repeats the previous sentence (0.416 -> 0.405): delete one", 0, "E"),
 (29, "MAE values between", "'MAE per cow between 0.614 and 0.713'", 0, "E"),
 (29, "labelled", "'labeled'", 0, "J"),
 (29, "real world", "'real-world'", 0, "E"),
 (30, "slightly more accurate", "Not significant (CI -0.025 to +0.066): 'the two were equally accurate ... difference not significant'", 0, "E"),
 (30, "0.94 and 0.90 for DINOcow", "Add 'and 0.94 and 0.91 for LoRA'", 0, "E"),
 (31, "weight of nadir", "'Around the weight nadir'", 0, "E"),
 (32, "3D imaging systems often", "Claim needs a citation (e.g. Mullins et al., 2019)", 0, "E"),
 (33, "greatest influence on scoring", "Contradicts Results (Figure 9): hook bones and right thurl largest at both ends; tailhead and rump moved less", 0, "E"),
 (33, "The following diagrams show", "-> 'Figure 12 shows ... for 4 cows, 2 losing condition in early lactation and 2 gaining it in late lactation'", 0, "F"),
 (34, "regions in particular change", "'particular regions change earlier'", 0, "E"),
 (34, "demonstrated to be", "'DINOcow proved to be'", 0, "E"),
 (34, "achieved a MAE of", "'an MAE of'", 0, "E"),
 # --- notas
 (35, "The authors have not stated any conflicts of interest", "JDS: add the AI disclosure statement if generative AI was used in writing", 0, "J"),
 (35, "Maria José", "'María José' as on the title page", 0, "E"),
 (35, "Investigation and Editing", "'Investigation' listed twice", 0, "E"),
 (35, "Mónica Martínez Vázquez", "Title page says 'Martínez-Vásquez'; 'Investigation' listed twice", 0, "E"),
 # --- referencias
 (36, "Ali, U., & Muhammad, W.", "JDS reference style, not APA: 'Ali, U., and W. Muhammad. 2025. Title. J. Abbrev. vol:pages. DOI'. Text citations use 'and', not '&'", 0, "J"),
 (36, "C. Souza, V.", "Malformed: 'Souza, V. C., C. B. Gleason, T. P. Price, ...'", 0, "E"),
 (36, "(1994b)", "Duplicate of 1994a: keep one entry", 0, "E"),
 (37, "Martinet, Jacques", "Malformed: Martinet, J., L.-M. Houdebine, and H. H. Head, ed. 1999. Biology of Lactation. INRA Editions", 0, "E"),
 (37, "Nutrient Requirements of Dairy Cattle. (2021)", "Author is NASEM (National Academies of Sciences, Engineering, and Medicine). Cite as NASEM (2021)", 0, "E"),
 (38, "Siméoni, O.", "arXiv preprint: JDS lists only published or accepted work", 0, "J"),
 (38, "(2026b). Automated", "Duplicate of 2026a: keep one entry", 0, "E"),
 # --- apéndice
 (39, "in the 37 with known", "'in the 37 cows with known DIM'", 0, "E"),
 (39, "centre line", "'center line'", 0, "J"),
 (39, "as seen in Figure 6", "-> Figure 13 (French chart); 14 after inserting the new figure", 0, "F"),
 (40, "used to draw Figure 7", "-> Figure 14 (traffic light); 15 after inserting the new figure", 0, "F"),
 (40, "reference curve (Figure 7)", "-> (Figure 12a; 13a after renumbering): the 37 cows are plotted there", 0, "F"),
 (40, "Each prediction was then automatically", "DUPLICATE sentence: delete", 0, "E"),
 (41, "Adapted curve based on the live weight", "Old caption: 'Traffic-light zones of BCS along the lactation ...' (text in the LaTeX)", 0, "E"),
 (41, "Figure 14 was built in the same way", "-> 'The lactation curves of Figure 11 were built ...'", 0, "F"),
 (41, "C. Souza et", "'Souza et al. (2025)'", 0, "E"),
 (41, "Requirements of Dairy Cattle (2021), because", "Cite as NASEM (2021)", 0, "E"),
 (42, "Figure 14 was built in the same way", "DUPLICATE paragraph: delete", 0, "E"),
 (43, "Mellado et al., 2016", "Missing from the reference list", 0, "E"),
 (43, "Table 10 extends", "OK as appendix. Keep it short (JDS: appendix not essential, no raw data)", 0, "J"),
]

pend, faltan = [], []
for pg, txt, corto, k, tipo in ERR:
    p = d[pg - 1]; rs = p.search_for(txt)
    if len(rs) <= k: faltan.append((pg, txt)); continue
    r = rs[k]; borde, fondo, pref = TIPOS[tipo]
    a = p.add_rect_annot(r + (-1, -1, 1, 1)); a.set_colors(stroke=borde); a.set_border(width=1.2 if tipo != "G" else 2); a.set_info(content=corto); a.update()
    pend.append((pg - 1, r.y0, pref + corto, fondo))

usados = {}
for n, y, texto, fondo in sorted(pend, key=lambda t: (t[0], t[1])):
    p = d[n]; ancho = 88
    y0 = max(y - 2, usados.get(n, 0) + 2)
    lin = max(1, int(len(texto) * 3.3 / (ancho - 6)) + 1)
    r = pymupdf.Rect(521, y0, 521 + ancho, y0 + 7.6 * lin + 3)
    a = p.add_freetext_annot(r, texto, fontsize=5.5, fontname="helv", text_color=(0, 0, 0), fill_color=fondo, border_width=0.5)
    a.update(); usados[n] = r.y1

L = d.new_page(0, width=612, height=792)
L.insert_text((50, 50), "PaperDINOcow (20 Nov, 16:04, 44 pp.) - corrections and JDS checks", fontsize=12, fontname="helv")
y = 72
for tipo, txt in [("G", "Blue = where the new resolution x block figure goes"), ("F", "Orange = wrong or missing figure/table reference"),
                  ("E", "Pink = text, number or reference-list error"), ("J", "Green = JDS Instructions for Authors 2026")]:
    L.draw_rect(pymupdf.Rect(50, y - 8, 70, y + 2), color=None, fill=TIPOS[tipo][1]); L.insert_text((78, y), txt, fontsize=8.5, fontname="helv"); y += 15
c = {t: sum(1 for e in ERR if e[4] == t) for t in TIPOS}
L.insert_text((50, y + 4), f"{len(ERR)} marks: {c['E']} errors, {c['F']} references, {c['J']} JDS, {c['G']} new figure.", fontsize=8.5, fontname="helv"); y += 24
for t in [
    "Figure order after inserting the new figure as Figure 6 (after Table 6):",
    "6 resolution x block grid (NEW), 7 agreement, 8 ROC, 9 learning curve, 10 averaged region map, 11 tracking,",
    "12 lactation curves, 13 BCS along lactation (37 cows), 14 French chart, 15 traffic light, 16 Martinet, 17 training curves.",
    "",
    "JDS points that apply to the whole manuscript (not marked one by one):",
    "- Tables and figures in separate sections at the end of the manuscript; each caption on the page of its figure.",
    "- Tables built with the Word table feature; numbered footnotes; title without final period.",
    "- Figures at final size (8.9, 14 or 19 cm wide), fonts 8-12 pt, lines and axes >= 1 pt, no gray lines,",
    "  no unnecessary grid lines; upload TIFF at 300 dpi or more (TIFF versions at 1200 dpi are ready).",
    "- Equations editable (MathType); references in JDS style with all authors and abbreviated journal names.",
    "- Reporting guideline checklist must be uploaded with the submission.",
    "- Length: about 30 manuscript pages for a full paper; Results and figure references should not be repeated",
    "  in the Discussion.",
]:
    L.insert_text((50, y), t, fontsize=8, fontname="helv"); y += 12
d.save(dst, garbage=3, deflate=True)
print("marcas:", len(pend), "| sin ubicar:", faltan)
