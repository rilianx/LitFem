#!/usr/bin/env python3
"""
Audita cada resumen del manifiesto contra su libro, sin llamadas al LLM:
  citas    : las frases entre comillas o en bloque (>) ¿estan en el libro? Se normaliza
             puntuacion y mayusculas, y una cita con "..." se parte en fragmentos.
  posicion : donde cae cada cita encontrada (0 % inicio - 100 % final) segun el momento
             del resumen en que aparece (A, B, C). Una cita de A al 90 % del libro es una alarma.
  nombres  : nombres propios del resumen que no aparecen nunca en el libro.

  python scripts/auditar_resumenes.py --manifiesto corpus/manifiesto.csv --raiz corpus/textos
"""
import argparse, os, re, sys, unicodedata
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptmv_corpus import leer_manifiesto, PATRON_MOMENTOS


def norm(t):
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def citas(texto):
    """[(momento, cita)] con las citas de >= 5 palabras, en orden de aparicion."""
    marcas = [(m.start(), m.group(1)) for m in re.finditer(PATRON_MOMENTOS, texto)]
    def momento(pos):
        etapa = "-"
        for p, etiqueta in marcas:
            if p <= pos:
                etapa = re.search(r"[ABC]", etiqueta.split(".")[0] if "." in etiqueta[:3] else etiqueta).group(0)
        return etapa
    out = []
    for m in re.finditer(r"[\"“]([^\"”\n]{20,}?)[\"”]", texto):
        if len(m.group(1).split()) >= 5:
            out.append((momento(m.start()), m.group(1)))
    return out


def buscar(cita, libro_n):
    """Fraccion de fragmentos encontrados y posicion relativa del primero."""
    frags = [norm(f) for f in re.split(r"\.\.\.|…|—|--", cita)]
    frags = [f for f in frags if len(f.split()) >= 4]
    if not frags:
        return None, None
    pos = [libro_n.find(f) for f in frags]
    hallados = [p for p in pos if p >= 0]
    return len(hallados) / len(frags), (hallados[0] / len(libro_n) if hallados else None)


COMUNES = set("""The A An And But In On At Of To For With From By As It He She They His Her Their This That These
Those When While After Before During Despite However Although Though Here There Then Thus Meanwhile Finally Later
Ultimately Eventually Once One Two Three First Second Chapter Book Part Act Introduction Body Finale Conclusion
Rising Action Narrator Narrative Main Characters Character Metadata Voice Perspective Type Knowledge Scope
Mediation Comprehensive Plot Summary Heterodiegetic Homodiegetic Omniscient External Internal Limited Unlimited
Authorial Moment Stage Mr Mrs Miss Dr Sir Lady Lord Saint St God I We You Its Our My Your Him Them Upon Within
Without Through Under Over Between Among Into Onto Not No Yes All Each Every Some Many Most Both Either Neither
Such What Which Who Whom Whose Why How Where Now Even Still Yet Also Only Just So If Because Since Unlike Like
Other Another Following Upon Instead Overwhelmed Desperate Determined Realizing Seeing Feeling Knowing Having
Being Upon Returning Arriving Leaving Left Right""".split())


def nombres_ausentes(resumen, libro):
    cand = set(re.findall(r"\b([A-Z][a-z]{2,}(?:[-'][A-Z]?[a-z]+)?)\b", resumen)) - COMUNES
    libro_l = libro.lower()
    return sorted(n for n in cand if n.lower() not in libro_l)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifiesto", required=True)
    p.add_argument("--raiz", default=".")
    p.add_argument("--detalle", help="CSV de salida con cada cita")
    a = p.parse_args()
    man = leer_manifiesto(a.manifiesto, a.raiz).drop_duplicates("obra")
    filas, detalle = [], []
    for _, r in man.iterrows():
        libro = open(r["ruta_libro"], encoding="utf-8", errors="replace").read()
        resumen = open(r["ruta_resumen"], encoding="utf-8", errors="replace").read()
        libro_n = norm(libro)
        cs = citas(resumen)
        res = []
        for mom, c in cs:
            frac, pos = buscar(c, libro_n)
            if frac is None:
                continue
            res.append((mom, frac, pos))
            detalle.append({"obra": r["obra"], "momento": mom, "encontrada": frac, "posicion": pos, "cita": c})
        n = len(res)
        exactas = sum(f == 1 for _, f, _ in res)
        fuera = [(m, p) for m, f, p in res if p is not None and
                 ((m == "A" and p > 0.5) or (m == "C" and p < 0.5))]
        filas.append({"obra": r["obra"], "citas": n,
                      "% exactas": round(100 * exactas / n) if n else None,
                      "% ninguna parte": round(100 * sum(f == 0 for _, f, _ in res) / n) if n else None,
                      "fuera de su momento": len(fuera),
                      "nombres que no estan en el libro": ", ".join(nombres_ausentes(resumen, libro)[:8])})
    t = pd.DataFrame(filas)
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print(t.to_string(index=False))
    if a.detalle:
        pd.DataFrame(detalle).to_csv(a.detalle, index=False)


if __name__ == "__main__":
    main()
