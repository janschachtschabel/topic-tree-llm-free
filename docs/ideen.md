# Ideen und Testkandidaten

Sammlung aller Ideen aus der Entwicklung dieser Demo: was umgesetzt ist, was getestet wurde und nichts gebracht
hat, und was noch offen ist. Messwerte stammen, wenn nicht anders angegeben, aus den 100 Beispielinhalten
(`data/beispiel_inhalte.json`, 10 Soll-Kategorien, KI-generiert) mit 10 Hauptgruppen.

**Kennzahlen:** Reinheit = Anteil der Inhalte in der überwiegenden Soll-Kategorie ihrer Gruppe; ARI und NMI =
Übereinstimmung mit den Soll-Kategorien (0 = zufällig, 1 = identisch); Namensnähe = Ähnlichkeit zwischen
Gruppenname und Soll-Kategorie (grobes Hilfsmaß).

Bei 100 Inhalten liegen Unterschiede von etwa ±0,03 ARI im Bereich des Messrauschens. Alles Offene sollte mit echten
WLO-Inhalten gemessen werden.

## Inhalte → Baum

### Umgesetzt

| Idee | Ergebnis | Wo |
|---|---|---|
| Embeddings + hierarchisches Clustering (Ward) | mpnet: Reinheit 64 %, ARI 0,39 | `contents.py`, „Gruppen bilden“ |
| model2vec `m2v-gte-256-edu` | etwa 300-mal schneller als mpnet, ohne PyTorch | Auswahl „Ähnlichkeit über“ |
| Text vorbereiten: nur Substantive (für model2vec) | ARI 0,20 → 0,35 | „Text vorbereiten“ |
| Grundformen + Komposita (simplemma + Grundwort-Regel) | ARI 0,35 → 0,41, Reinheit 59 → 64 %; automatisch beim Gruppenbilden mit model2vec | „Text vorbereiten“ |
| Benennung über Begriff aus den Texten (c-TF-IDF + Abdeckung + Nähe zum Zentrum) | Namensnähe 0,61 | „Gruppen benennen mit“ |
| Hybrid: GND-Schlagwort mit exakt gleichem Namen | Namensnähe 0,57–0,58 | „Gruppen benennen mit“ |
| Hybrid: Wikipedia-Artikel als Name und Beschreibung | Namensnähe 0,55–0,58, liefert echte Beschreibungstexte | „Gruppen benennen mit“ |
| Hybrid: in Wikidata-Taxonomie einordnen (obere zwei Ebenen, Ebene 1 leicht bevorzugt) | Namensnähe 0,61–0,65, beste Namen | „Gruppen benennen mit“ |
| In fertiges Gerüst einsortieren (Wikipedia/Wikidata/GND) | Wikidata 48 %, ARI 0,18; Wikipedia 32 % | „In Gerüst einsortieren“ |
| Tab „Inhalte“ zum Durchblättern, mit Gruppe aus dem letzten Lauf | – | Oberfläche |

### Getestet, ohne Gewinn (nicht übernommen)

| Idee | Ergebnis | Grund |
|---|---|---|
| TF-IDF statt Embeddings | ARI 0,03 | zu wenig gemeinsame Wörter bei kurzen Texten |
| Andere Clustering-Verfahren (Average-Linkage, k-Means, Spectral) | ARI 0,20–0,40, schwankend | Ward am stabilsten |
| Häufigste Hauptkomponenten entfernen („all-but-the-top“) | ARI sinkt (mpnet 0,39 → 0,31) | entfernt auch Themeninformation |
| Füllwörter automatisch (Substantive in >12–30 % der Inhalte entfernen) | ARI 0,35–0,36 | feste Füllwortliste reicht; strengere Schwellen entfernen Fachbegriffe |
| Gelenktes Clustering mit 12 Wikidata-Knoten „Physik“ als Startpunkten | ARI 0,13–0,18 | Gerüst passt nicht zu den Inhalten |
| Gelenktes Clustering, Startpunkte nach Datenlage aus 103 Knoten gewählt | ARI 0,18–0,27 | schlechter als freies Clustering |
| Oberbegriff frei aus Wikidata (gemeinsamer Vorfahr über „Unterklasse von“/„Teil von“) | Namensnähe 0,46 | Suche bildet Begriffe oft auf falsche Objekte ab; Namen wie „physikalische Eigenschaft“ |
| Oberbegriff nur innerhalb des Wikidata-Gerüsts | Namensnähe 0,57 | findet selten etwas, fällt auf Textbegriff zurück |
| Grundformen + Komposita beim Einsortieren ins Gerüst | ARI 0,17 → 0,13 | dort bleibt „nur Substantive“ |
| HDBSCAN direkt auf den Embeddings | ARI ≈ 0,00, 2–4 Gruppen | funktioniert in hoher Dimension nicht |
| UMAP + HDBSCAN (BERTopic-Standard), 15 Einstellungen × 5 Startwerte | model2vec bestenfalls ARI 0,34 (Ward 0,41); mpnet bestenfalls 0,40 (Ward 0,39), Spanne 0,34–0,43, 3–20 % als „Rauschen“ | nicht besser, zufallsabhängig, zusätzliche Abhängigkeit (umap-learn, numba) |

Testskripte: [`scripts/test_verbesserungen.py`](../scripts/test_verbesserungen.py),
[`scripts/test_umap_hdbscan.py`](../scripts/test_umap_hdbscan.py).

### Offen (Kandidaten für weitere Tests)

| Priorität | Idee | Erwarteter Nutzen | Aufwand | Hinweise |
|---|---|---|---|---|
| 1 | **Echte WLO-Inhalte, bestehende Sammlungen als Soll-Kategorien** | belastbare Messwerte für alle Varianten | klein bis mittel | Voraussetzung für jede weitere Entscheidung |
| 2 | **Kleines lokales Sprachmodell nur für Namen und Kurzbeschreibungen** (1–4 Mrd. Parameter, z. B. Qwen 2.5 3B, CPU) | deutlich bessere Namen; Struktur bleibt ohne LLM | mittel | bekommt nur Kandidatenbegriffe und drei Titel je Gruppe; Temperatur 0 für Reproduzierbarkeit |
| 2 | **Vorhandene Keywords und Fach als Zusatzsignal**; bei gemischten Mengen erst nach Fach trennen | weniger Mischgruppen | klein | mit den Beispieldaten nicht prüfbar |
| 3 | **Andere Embeddings:** bge-m3, multilingual-e5, gte-multilingual-base, `m2v-bge-m3-edu` (512 Dim.) | feinere Bedeutung | klein | reiner Austausch, gleiche Messung |
| 3 | **UMAP (15 Nachbarn, 5 Dim.) + Ward für mpnet** | ARI 0,39 → 0,43 (Spanne 0,42–0,45) auf Beispieldaten | klein | bei model2vec schlechter; mit festem Startwert reproduzierbar; mit echten Daten prüfen |
| 3 | **Cross-Encoder zum Sortieren der Namenskandidaten** (≈100 Mio. Parameter) | treffsicherere Namen ohne generatives Modell | klein bis mittel | |
| 3 | **Mehrfachzuordnung**, wenn zwei Knoten fast gleich gut passen | querliegende Themen werden nicht zerrissen | klein | |
| 4 | **Beschreibungen aus Sätzen zusammensetzen** (repräsentativste Sätze statt drei Titel) | lesbarere Gruppenbeschreibungen | klein | |
| 4 | **Gruppenzahl stabiler wählen** (Bootstrap statt Silhouette) | weniger Zufall bei „automatisch“ | klein | |
| 4 | **Gelenktes Clustering mit passendem Gerüst** (Lehrplan, Redaktions-Taxonomie) | Struktur folgt einem fachlich gewollten Gerüst | mittel | am Wikidata-Gerüst gescheitert, weil es nicht passte |
| 4 | **Komposita mit Wörterbuch zerlegen** statt Grundwort-Regel | genauere Zerlegung | klein | `charsplit` ließ sich nicht sauber nutzen |
| 4 | **Neue Inhalte in bestehenden Baum einordnen** statt neu zu clustern | stabile Bäume über die Zeit | klein | nächstes Gruppenzentrum, Schwelle für „Sonstiges“ |
| 4 | **Redaktion einbinden:** Gruppen zusammenlegen, teilen, umbenennen | Qualität durch Menschen | mittel | Oberfläche nötig |
| 5 | **Embedding auf WLO-Sammlungen nachtrainieren** (Inhalte derselben Sammlung nah beieinander) | vermutlich größter Hebel | groß | Trainingsdaten und Rechenzeit nötig |

## Thema → Baum

| Status | Idee | Hinweise |
|---|---|---|
| umgesetzt | Wikipedia-Gliederung + Hauptartikel; Wikidata „Teil von/Unterklasse von/hat Teil“; GND-Unterbegriffe und Sachgruppen | Wikipedia insgesamt am brauchbarsten, als Standard gesetzt |
| offen | **Lehrpläne als Quelle** (PDF → Markdown, z. B. docling; Überschriften als Gliederung) | didaktisch passend, aber Parser je Bundesland; Überschriften in PDFs oft nicht ausgezeichnet |
| offen | **Allgemeine Wikipedia-Abschnitte filtern** („Geschichte“, „Rezeption“, „Formulierungen“ auf Ebene 3) | bewusst noch nicht gefiltert, um die Rohqualität zu sehen |
| offen | **Kombination:** Struktur aus Wikipedia/Wikidata, GND-IDs als Schlagwörter | Normdaten-Anbindung für edu-sharing |
| offen | **Bildungsstufe berücksichtigen** (z. B. Abschnitte nach Fachsprache filtern) | keine Quelle ist didaktisch gegliedert |

## Kosten und LLM-Generator

| Status | Idee | Effekt laut Kostenanalyse |
|---|---|---|
| offen | Im bestehenden Generator nur Titel statt ganzer Objekte in die Kontextlisten (`main.py:230`, `main.py:270`) | Baum 35–43 % günstiger |
| offen | Wortgrenze `{max_description_length}` tatsächlich einsetzen (`prompts.py:47`) | kürzere, gleichmäßigere Beschreibungen |
| offen | `reasoning_effort` setzen | bis zu 50–100 % weniger Output bei Reasoning-Modellen |
| offen | Bei kaputtem JSON neu versuchen statt Zweig zu verwerfen | Vollständigkeit |
| offen | Prompts für Caching umbauen (großer, gleicher Kontext vorne) | „Baum aus Inhalten“: 0,96 → 0,15 $ |
| offen | Kompendium nur für Ebene 1 und 2 | Kosten und Laufzeit −82 % |
| offen | Überarbeitung nur als betroffener Zweig | ein Fünftel bis ein Viertel der Kosten von „ganzer Baum neu“ |

Details: [kosten.md](kosten.md), [verfahren.md](verfahren.md).
