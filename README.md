# Diseases Diagnostician (V2.0): KI-gestützter medizinischer Chatbot mit Bilderkennung

> _Projektstudium 2 an der Schnittstelle von KI, NLP, Computer Vision und Gesundheitswesen._

![Python](https://img.shields.io/badge/Python-3.11%2B-blau)
![Framework](https://img.shields.io/badge/Backend-Flask-orange)
![Lizenz](https://img.shields.io/badge/Lizenz-MIT-grün)

---

## Kurzbeschreibung

**Diseases Diagnostician (V2.0)** ist ein Prototyp eines intelligenten Chatbots, der:

- Textbasierte Symptome analysiert → Diagnosevorschläge macht.
- Medizinische Bilder klassifiziert (Röntgen).
- Über eine Web-Oberfläche bedienbar ist.

<img width="1126" height="763" alt="image" src="https://github.com/user-attachments/assets/361b0bd0-73ac-46db-b07b-79cb4a5e9191" />

---

## Umsetzung

### 1. Sprachmodell (Symptom → Diagnose)

- **Daten**: Eigenkuration & Bereinigung medizinischer Symptom-Diagnose-Paare.
- **Modell**: Feinabstimmung von **GPT-2**.
- **Training**: Mit Hugging Face & PyTorch → BERTSCORE

```
Precision: 86%
Recall: 87%
F1: 87%

```

### 2. Bildklassifikation

- **Modelle**: **ResNet50** & **Vision Transformer**.
- **Daten**: HAM10000, CheXpert – mit Augmentation vorbereitet.
- **Genauigkeit**: Bis zu 93% .


### 3. Webanwendung

- **Backend**: Flask (Python) – REST-API für Text- und Bildanalyse.
- **Frontend**: Reaktive Oberfläche (HTML/CSS/JS) zum Chatten und Hochladen von Bildern.

---

## Starten

```bash
git clone https://github.com/Abodx9/Diseases-Diagnostician_v2.git
cd Diseases-Diagnostician_v2
pip install -r requirements.txt
python server.py
```

> ⚠️ Modelldateien nicht im Repo – ggf. separat trainieren oder [herunterladen](https://www.mediafire.com/file/m8gnz4uk1zxjp5d/models.tar.gz/file)
---


## Was wir gelernt haben

- Teamarbeit ist entscheidend – besonders mit unterschiedlichen Vorkenntnissen.
- Saubere Daten sind wichtiger als komplexe Modelle.
- KI im Gesundheitswesen braucht Verantwortungsbewusstsein & Fallbacks.
- Gute UX ist genauso wichtig wie Modellgenauigkeit.

---

_„Technik ersetzt keine Ärzte – aber Ärzte, die Technik nutzen, werden die sein, die bleiben.“_
