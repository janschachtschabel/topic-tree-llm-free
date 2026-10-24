# Kosten eines LLM-basierten Themenbaums

Diese Analyse schätzt die LLM-Kosten des bestehenden Themenbaum-Generators
(`topic-tree-generator`, FastAPI, API 1.2.6) für drei Use Cases. Sie war der
Anlass für diese Demo: Lässt sich die Themenbaum-Erzeugung auch **ohne generatives LLM** umsetzen?
Die Ergebnisse sind in der App unter **Kosten** kompakt aufbereitet, mit einem Rechner für eigene Annahmen.

Alle Beträge in US-Dollar. Gerechnet für **gpt-6-luna** und **gpt-5.6-luna**.

## Kurzfassung

- **Der Themenbaum selbst ist günstig:** 0,07–0,16 $ für 12×5×5 (372 Knoten), 0,15–0,33 $ für 12×10×5 (732 Knoten).
- **Ein Kompendium-Text je Sammlung dominiert** mit über 90 % der Kosten: 2,83–8,18 $ für 372 Knoten. Bei mehreren
  hundert Knoten kommt trotz kleiner Einzelbeträge einiges zusammen.
- **Inhalte skalieren linear mit der Menge:** beschreibende Metadaten (Titel, Beschreibung, Keywords) kosten etwa
  0,07–0,14 Cent je Inhalt; Fach und Bildungsstufe laufen ohne LLM.
- **Überarbeitungswünsche sind billig**, wenn nur der betroffene Zweig neu erzeugt wird (1–4 Cent).
- **Teuer wird es nur, wenn Volltexte vieler Inhalte in jeden Aufruf wandern** und kein Prompt-Caching genutzt wird
  (bis 14 $ je Baum).

![Kosten je Use Case, Themenbaum 12×5×5](kosten/00_uebersicht_usecases_12x5x5.png)

| Use Case (12×5×5, inkl. Kompendium) | gpt-6-luna | gpt-5.6-luna |
|---|---:|---:|
| Themenbaum aus einem Thema | 3,42 $ | 7,22 $ |
| … + 1 Überarbeitung | 3,43 $ | 7,25 $ |
| … + 100 eigene Inhalte | 3,50 $ | 7,39 $ |
| Themenbaum aus 100 Inhalten, nur Metadaten | 3,48 $ | 7,35 $ |
| Themenbaum aus 100 Inhalten, mit Volltext (mit Cache) | 3,90 $ | 8,20 $ |
| Themenbaum aus 100 Inhalten, mit Volltext (ohne Cache) | 7,30 $ | 14,98 $ |

Dieselbe Übersicht für 12×10×5: [`kosten/00b_uebersicht_usecases_12x10x5.png`](kosten/00b_uebersicht_usecases_12x10x5.png).

## Preise und Methode

| Modell (je 1 Mio. Tokens) | Input | Output | Cache lesen | Cache schreiben (30 min) |
|---|---:|---:|---:|---:|
| gpt-6-luna | 0,10 $ | 0,50 $ | 0,01 $ | 0,125 $ |
| gpt-5.6-luna | 0,20 $ | 1,20 $ | 0,02 $ | 0,25 $ |

- Die Prompts stammen 1:1 aus `src/prompts.py` des Generators, gezählt mit dem o200k-Tokenizer (Abweichung zum echten
  Tokenizer etwa ±10–15 %).
- Ablauf des Generators: 1 Aufruf für die Hauptthemen, 1 je Hauptthema für die Unterthemen, 1 je Unterthema für die
  Lehrplanthemen. 12×5×5 = 73 Aufrufe, 12×10×5 = 133 Aufrufe.
- System-Prompt 578 Tokens. Ein Knoten im Output (Titel, Kurztitel, Beschreibung, 5 Keywords) hat 182 Tokens bei
  60 Wörtern Beschreibung, 264 Tokens bei 100 Wörtern.
- **Basis:** 60 Wörter, kein Reasoning. **Hoch:** 100 Wörter und 1.000 Reasoning-Tokens je Aufruf.
- Die Modelle schöpfen die Obergrenzen voll aus (laut Prompt sind es Höchstwerte; real entstehen eher weniger Knoten).
- Nicht enthalten: Crawler, Hosting, Wiederholungen nach API-Fehlern.

## Use Case 1: Themenbaum aus einem Thema

![Kosten pro Themenbaum](kosten/01_themenbaum_aus_thema.png)

| Baum | Aufrufe | Knoten | gpt-6-luna | gpt-5.6-luna | hoch (6 / 5.6) |
|---|---:|---:|---:|---:|---:|
| 5×3×2 (Default), aktueller Code | 21 | 50 | 0,011 $ | 0,023 $ | 0,024 / 0,055 $ |
| 12×5×5, aktueller Code | 73 | 372 | 0,071 $ | 0,157 $ | 0,131 / 0,297 $ |
| 12×5×5, mit Prompt-Fix | 73 | 372 | 0,044 $ | 0,103 $ | 0,095 / 0,224 $ |
| 12×10×5, aktueller Code | 133 | 732 | 0,151 $ | 0,330 $ | 0,267 / 0,600 $ |
| 12×10×5, mit Prompt-Fix | 133 | 732 | 0,086 $ | 0,200 $ | 0,181 / 0,428 $ |

„Prompt-Fix“ = in den Kontextlisten nur Titel statt ganzer Objekte (siehe Befunde unten).

### Zusätzlicher Überarbeitungswunsch

Bestehender Baum + Änderungswunsch gehen erneut ins Modell.

![Kosten eines Überarbeitungswunschs](kosten/02_ueberarbeitungswunsch.png)

| Variante (12×10×5) | gpt-6-luna | gpt-5.6-luna |
|---|---:|---:|
| A: ganzer Baum neu ausgeben | 0,084 $ | 0,197 $ |
| B: nur betroffener Zweig (1 Hauptthema) | 0,020 $ | 0,042 $ |
| B + Cache (Ø über 5 Wünsche in 30 min) | 0,011 $ | 0,023 $ |

Variante A passt bei 12×10×5 nicht in eine Antwort (140–200 Tsd. Output-Tokens bei typischen 128 Tsd. Grenze) und
müsste gestückelt werden. Variante B ist die sinnvolle Umsetzung.

## Use Case 2: Nutzer bringt 100 Inhalte mit

Volltexte kommen vom Crawler, Fach und Bildungsstufe ohne LLM. Per LLM entstehen nur Titel, Beschreibung und
Keywords (1 Aufruf je Inhalt). Optional werden die Inhalte einem bestehenden Baum zugeordnet; dort lohnt sich
Prompt-Caching, weil die Titelliste des Baums in allen Aufrufen gleich bleibt.

![Kosten für 100 Inhalte](kosten/03_100_eigene_inhalte.png)

| Schritt (100 Inhalte) | gpt-6-luna | gpt-5.6-luna |
|---|---:|---:|
| Metadaten, Volltext kurz (≈2 Tsd. Tokens) | 0,038 $ | 0,082 $ |
| Metadaten, Volltext mittel (≈5 Tsd.) | 0,068 $ | 0,142 $ |
| Metadaten, Volltext lang (≈15 Tsd., gekappt) | 0,168 $ | 0,342 $ |
| Zuordnung zu 12×5×5, ohne / mit Cache | 0,061 / 0,012 $ | 0,122 / 0,025 $ |

## Use Case 3: Struktur aus 100 Inhalten ableiten

Es gibt keine Strukturvorgabe; alle 100 Inhalte gehen mit Metadaten (wahlweise mit Volltext) in **jeden** Aufruf
des Generators. Caching wirkt stark (Korpus einmal schreiben, danach zu 10 % lesen), setzt aber umgebaute Prompts
voraus: Korpus vorne, variable Teile hinten.

![Baum aus 100 Inhalten, 12×5×5](kosten/04_baum_aus_100_inhalten_12x5x5.png)

| Korpus im Kontext (12×5×5) | ohne Cache (6 / 5.6) | mit Cache (6 / 5.6) |
|---|---:|---:|
| nur Metadaten (≈250 Tokens je Inhalt) | 0,23 / 0,47 $ | 0,07 / 0,15 $ |
| + Volltext kurz (≈2,25 Tsd.) | 1,69 / 3,39 $ | 0,23 / 0,48 $ |
| + Volltext mittel (≈5,25 Tsd.) | 3,88 / 7,77 $ | 0,49 / 0,99 $ |

12×10×5: [`kosten/04b_baum_aus_100_inhalten_12x10x5.png`](kosten/04b_baum_aus_100_inhalten_12x10x5.png) (bis 14,17 $ ohne Cache).
**Kontextfenster:** 100 Volltexte à 5 Tsd. Tokens ergeben rund 525 Tsd. Tokens je Aufruf. Das sprengt viele
Kontextfenster; dann Volltexte auf etwa 2 Tsd. Tokens kappen oder zweistufig arbeiten.

## Kompendium-Text je Sammlung

Nach Messung 60–80 Tsd. Tokens und 30–40 Sekunden je Knoten, gerechnet mit 4–6 Tsd. Output-Tokens.

![Kosten der Kompendium-Texte](kosten/05_kompendium.png)

| Baum | Knoten | gpt-6-luna | gpt-5.6-luna | Laufzeit seriell | bei 10 parallel |
|---|---:|---:|---:|---:|---:|
| 5×3×2 | 50 | 0,38–0,52 $ | 0,80–1,10 $ | 0,5 h | 3 min |
| 12×5×5 | 372 | 2,83–3,87 $ | 5,95–8,18 $ | 3,6 h | 22 min |
| 12×10×5 | 732 | 5,56–7,61 $ | 11,71–16,10 $ | 7,1 h | 43 min |

Nur Ebene 1 und 2 mit Kompendium (132 statt 732 Knoten) senkt Kosten und Zeit um 82 %.

## Befunde im bestehenden Generator, die Kosten treiben

1. **Ganze Objekte statt Titel im Prompt** (`main.py:230`, `main.py:270`): `f"- {main_topic_title}"` über
   `main_topics` schreibt das ganze `Collection`-Objekt samt Beschreibung und URIs in den Prompt (≈238 Tokens je
   Eintrag). Bei 12×10×5 sind das 80 % des Inputs; mit `.title` wird ein Baum 35–43 % günstiger.
2. **Die Wortgrenze kommt nie an** (`src/prompts.py:47`): `{max_description_length}` in `BASE_INSTRUCTIONS` wird nie
   formatiert; die Templates haben den Platzhalter nicht. Beschreibungen werden eher länger.
3. **Kein `reasoning_effort` gesetzt:** Bei Reasoning-Modellen fallen standardmäßig Reasoning-Tokens an (50–100 %
   mehr Output).
4. **Caching greift beim Baum nicht:** Der gemeinsame Präfix (578 Tokens) liegt unter der üblichen Mindestlänge.
5. **Kaputtes JSON wird bezahlt und verworfen** (`src/structured_text_helper.py:85`): kein neuer Versuch, der Zweig fehlt.

## Fazit

- Die Baumerzeugung ist nicht das Kostenproblem, auch nicht ein einzelner Kompendium-Text. Die Kosten wachsen mit
  der **Zahl der Knoten** (Kompendium) und der **Zahl der Inhalte** (beschreibende Metadaten).
- Ein Themenbaum ohne LLM spart deshalb wenig Geld. Gründe dafür sind eher Reproduzierbarkeit, Nachvollziehbarkeit,
  Lehrplannähe und Unabhängigkeit von LLM-Anbietern. Diese Demo zeigt, wie weit man damit kommt
  (siehe [verfahren.md](verfahren.md)).
- Auch AcademicCloud über die B-API ist inzwischen kostenpflichtig wie jede andere API.

## Nachrechnen

- **Rechner in der App** (Bereich „Kosten“): Inhaltsmenge, Anpassungsschleifen, Baumgröße, Preise und Kompendium frei
  einstellbar. Das Token-Modell ist aus den gemessenen Prompts abgeleitet.
- **Skripte:** [`scripts/kostenmodell.py`](../scripts/kostenmodell.py) zählt die Tokens aus einem Checkout des
  Generators und schreibt [`data/kosten.json`](../data/kosten.json);
  [`scripts/grafiken_kosten.py`](../scripts/grafiken_kosten.py) erzeugt daraus die Grafiken in `docs/kosten/`.
