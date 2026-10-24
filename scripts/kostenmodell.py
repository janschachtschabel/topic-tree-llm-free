"""Kostenmodell für den LLM-basierten topic-tree-generator (Token-Zählung aus den echten Prompts).

Braucht einen Checkout des topic-tree-generator (für src/prompts.py und die DTOs):

    uv run --with tiktoken --with pydantic --with pydantic-settings         python scripts/kostenmodell.py <pfad/zum/topic-tree-generator> data/kosten.json

Ergebnis: data/kosten.json (Grundlage für docs/kosten.md und scripts/grafiken_kosten.py).
"""
import sys, json
sys.path.insert(0, sys.argv[1])
import tiktoken
from src.prompts import BASE_INSTRUCTIONS, MAIN_PROMPT_TEMPLATE, SUB_PROMPT_TEMPLATE, LP_PROMPT_TEMPLATE, DESCRIPTION_PROMPT_TEMPLATE
from src.DTOs.collection import Collection
from src.DTOs.properties import Properties
Collection.model_rebuild()
enc = tiktoken.get_encoding("o200k_base"); t = lambda s: len(enc.encode(s))
P = {"gpt-5.6-luna": dict(i=0.20,o=1.20,c=0.02,w30=0.25), "gpt-6-luna": dict(i=0.10,o=0.50,c=0.01,w30=0.125)}
M = list(P)
theme = "Physik in Anlehnung an die Lehrpläne der Sekundarstufe 2"
ctx = "Kontext-Informationen:\nFachbereich-URIs: http://w3id.org/openeduhub/vocabs/discipline/460\nFachbereich: ['Physik']\nBildungsstufe-URIs: http://w3id.org/openeduhub/vocabs/educationalContext/sekundarstufe_2\nZielgruppe / Bildungsstufe: ['Sekundarstufe II']"
base = ("Elektromagnetische Induktion verbindet Magnetfelder und elektrische Spannungen auf faszinierende Weise. "
 "Lernende untersuchen, wie bewegte Magnete in Spulen Ströme erzeugen. Sie analysieren das Faradaysche Gesetz und die Lenzsche Regel an konkreten Experimenten. "
 "Generatoren, Transformatoren und Induktionsherde zeigen alltägliche Anwendungen. Messreihen mit Oszilloskop und Datenlogger fördern das quantitative Verständnis. "
 "Schülerinnen und Schüler berechnen induzierte Spannungen und deuten Energieumwandlungen im Stromnetz. ")
kw = ["Induktion","Faradaysches Gesetz","Lenzsche Regel","Generator","Transformator"]
def mkdesc(words): w=(base*3).split(); return " ".join(w[:words])
def cost(m, tin=0, tout=0, tcached=0, twrite=0):
    p=P[m]; return (tin*p["i"] + tout*p["o"] + tcached*p["c"] + twrite*p["w30"])/1e6
def C(**kw): return {m: round(cost(m,**kw),4) for m in M}
STRUCT = {"5×3×2 (Default)":(5,3,2), "12×5×5":(12,5,5), "12×10×5":(12,10,5)}
def tree(NM,NS,NL,desc_words=60,reason=0,fix=False):
    desc=mkdesc(desc_words)
    def coll(title):
        pr = Properties(cclom_general_keyword=kw, ccm_collectionshorttitle=["Kurz"], cm_description=[desc], cm_title=[title])
        return Collection(title=title, shorttitle="Kurz", properties=pr, subcollections=[])
    NODE = t(json.dumps({"title":"Elektromagnetische Induktion","shorttitle":"Induktion","description":desc,"keywords":kw}, ensure_ascii=False, indent=2))+2
    SYS=t(BASE_INSTRUCTIONS)
    mains=[coll(f"Hauptthema Elektrodynamik {i}") for i in range(NM)]; subs=[coll(f"Unterthema Induktion {i}") for i in range(NS)]
    pm = "\n".join(f"- {m.title if fix else m}" for m in mains); ps = "\n".join(f"- {s.title if fix else s}" for s in subs)
    main_in=SYS+t(MAIN_PROMPT_TEMPLATE.format(themenbaumthema=theme,num_main=NM,existing_titles="",special_instructions="Keine besonderen Anweisungen.",context_instructions=ctx,max_description_length=60))
    sub_in=SYS+t(SUB_PROMPT_TEMPLATE.format(themenbaumthema=theme,main_theme=mains[0].title,num_sub=NS,context_instructions=ctx,previous_main_topics=pm,max_description_length=60))
    lp_in=SYS+t(LP_PROMPT_TEMPLATE.format(themenbaumthema=theme,main_theme=mains[0].title,sub_theme=subs[0].title,num_lp=NL,context_instructions=ctx,previous_main_topics=pm,previous_sub_topics=ps,max_description_length=60))
    calls=1+NM+NM*NS; nodes=NM+NM*NS+NM*NS*NL
    tin=main_in+NM*sub_in+NM*NS*lp_in
    vis=(NM*NODE+150)+NM*(NS*NODE+10)+NM*NS*(NL*NODE+10)
    return dict(calls=calls,nodes=nodes,tin=tin,tout=vis+calls*reason,vis=vis,NODE=NODE,SYS=SYS,branch=(1+NS+NS*NL)*NODE)
D = {"uc1":[], "rev":[], "uc2":[], "assign":[], "uc3":[], "komp":[]}
for s,(a,b,c) in STRUCT.items():
    row={"struct":s}
    for case,(dw,r) in {"basis":(60,0),"hoch":(100,1000)}.items():
        for code in ("ist","fix"):
            d=tree(a,b,c,dw,r,code=="fix")
            row.update(calls=d["calls"],nodes=d["nodes"])
            row[f"{case}_{code}"]=dict(tin=d["tin"],tout=d["tout"],**C(tin=d["tin"],tout=d["tout"]))
    D["uc1"].append(row)
for s in ("12×5×5","12×10×5"):
    a,b,c=STRUCT[s]; row={"struct":s}
    for case,(dw,r) in {"basis":(60,0),"hoch":(100,4000)}.items():
        d=tree(a,b,c,dw,0); tin=d["SYS"]+d["vis"]+150; br=d["branch"]
        row[case]=dict(tree_tok=d["vis"], branch_tok=br,
            A=dict(tin=tin,tout=d["vis"]+r,**C(tin=tin,tout=d["vis"]+r)),
            B=dict(tin=tin,tout=br+r,**C(tin=tin,tout=br+r)),
            Bc={m: round((cost(m,150,br+r,twrite=tin-150)+4*cost(m,150,br+r,tcached=tin-150))/5,4) for m in M})
    D["rev"].append(row)
INSTR = t(DESCRIPTION_PROMPT_TEMPLATE)+200
META_OUT = 250
for label,ft in {"Volltext kurz (≈2k Tokens)":2000,"Volltext mittel (≈5k)":5000,"Volltext lang (≈15k, gekappt)":15000}.items():
    tin=100*(INSTR+ft)
    D["uc2"].append(dict(label=label,ft=ft,tin=tin,basis=dict(tout=100*META_OUT,**C(tin=tin,tout=100*META_OUT)),hoch=dict(tout=100*(META_OUT+800),**C(tin=tin,tout=100*(META_OUT+800)))))
for s in ("12×5×5","12×10×5"):
    nodes=tree(*STRUCT[s])["nodes"]; tl=nodes*14+300
    D["assign"].append(dict(struct=s,titlelist=tl,ohne=C(tin=100*(tl+250),tout=100*60),mit=C(tin=100*250,tout=100*60,twrite=tl,tcached=99*tl)))
# UC3: corpus of 100 contents in context of every generator call (prompt fixed + corpus as static prefix)
CORPUS = {"nur Metadaten (≈250/Inhalt)":250, "Metadaten + Volltext kurz (≈2,25k/Inhalt)":2250, "Metadaten + Volltext mittel (≈5,25k/Inhalt)":5250}
META_PRE = {250:5000, 2250:2000, 5250:5000}  # Metadaten-Erzeugung vorab (Volltextlänge)
for s in ("12×5×5","12×10×5"):
    d=tree(*STRUCT[s],fix=True)
    for label,per in CORPUS.items():
        corp=100*per+300
        meta=[x for x in D["uc2"] if x["ft"]==META_PRE[per]][0]["basis"]
        no = C(tin=d["tin"]+d["calls"]*corp, tout=d["tout"])
        yes= {m: round(cost(m,tin=d["tin"],tout=d["tout"],twrite=corp,tcached=(d["calls"]-1)*corp),4) for m in M}
        D["uc3"].append(dict(struct=s,label=label,corpus_tok=corp,calls=d["calls"],tin_nocache=d["tin"]+d["calls"]*corp,ohne=no,mit=yes,meta=meta))
for s,(a,b,c) in STRUCT.items():
    n=tree(a,b,c)["nodes"]
    D["komp"].append(dict(struct=s,nodes=n,low=C(tin=n*56000,tout=n*4000),high=C(tin=n*74000,tout=n*6000),
        tok_low=n*60000,tok_high=n*80000,h_seq=round(n*35/3600,1),min_par10=round(n*35/10/60)))
D["prices"]=P
json.dump(D, open(sys.argv[2],"w",encoding="utf-8"), ensure_ascii=False, indent=1)
for k in ("uc3","komp"):
    for r in D[k]: print(k, {x:r[x] for x in r if x not in ("meta",)})
