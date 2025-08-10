# TradeRepublic Portfolio Analyzer

Dieses Projekt bietet eine einfache Möglichkeit, das eigene TradeRepublic-Portfolio
zu analysieren. Der Download der aktuellen Portfoliozusammensetzung erfolgt über
die Kommandozeile. Weitere Auswertungen wie Sunburst-Visualisierung und
Efficient-Frontier-Analyse werden über ein Webinterface bereitgestellt. Zwei
Sunburst-Diagramme zeigen die Verteilung nach Branchen und nach Ländern, wobei
der innere Ring nach ETFs und Einzelaktien unterteilt ist. Das Portfolio wird so
optimiert, dass das Sharpe Ratio maximal ist. Optional können Constraints für
einzelne Werte gesetzt werden (Format `ISIN<=0.2,ISIN>=0.1`).

## Installation

```bash
pip install -r requirements.txt
```

## Portfolio herunterladen

```bash
python cli.py <telefonnummer> <pin>
```
Dies erstellt eine `portfolio.csv` im Projektverzeichnis.

## Weboberfläche starten

```bash
python app.py
```
Anschließend kann die Oberfläche unter `http://localhost:5000` im Browser
geöffnet werden. Im Formular lassen sich optionale Constraints eingeben, bevor
die Optimierung gestartet wird.
