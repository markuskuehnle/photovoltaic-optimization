## Optimierung einer Photovoltaik-Anlage

In diesem Projekt wurde ich als Berater hinzugezogen, um Optionen zur mathematischen Optimierung für eine Elektrolyseanlage eines Energieunternehmens zu erarbeiten. Durch die Kombination von Simulationstechniken und fortgeschrittenem Machine Learning zielt mein Ansatz darauf ab, die Energieflüsse innerhalb des Systems zu maximieren und den Wasserstoffoutput zu optimieren. In diesem Notebook wird detailliert beschrieben, wie ich durch gezielte Optimierung und präzise Vorhersagemodelle die Energieeffizienz und Leistung der PV-Anlage verbessern konnte.

### 1. Beschreibung des Problems

#### 1.1 Systemkomponenten

- PV-Anlage (Photovoltaik)
- Hausverbrauch (inkl. Wärmepumpe)
- Batterie
- Elektrolyseur

#### 1.2 Ziel der Optimierung

Die Last der Wärmepumpe, des Hausverbrauchs und die Leistung der PV-Anlage sollen zur Vereinfachung als eine Größe betrachtet werden. Zum Ausgleich soll die Batterie entweder be- oder entladen werden. Die Elektrolyse soll möglichst viel Leistung aufnehmen, um den Wasserstoffoutput zu maximieren, ohne dass Strom aus dem Netz bezogen wird.

**Betrachtungszeitraum:**
Es sollen nur die Sommermonate betrachtet werden, in denen die PV-Anlage auch so viel produziert, dass das realistisch ist.

**Erwartetes Ergebnis:**
- größtmöglicher Wasserstoffoutput
- die optimalen Energieflüsse, um diesen Output zu erreichen

### 2. Betrachtete Probleme

1. **Optimierung der Be- und Entladezyklen**
    - Wie können die Lade- und Entladezyklen der Batterie so optimiert werden, dass der Wasserstoffoutput maximiert wird?
    - Welche Algorithmen eignen sich am besten zur Optimierung der Batterienutzung?
        - Lineare Programmierung (LP)
        - Nicht-Lineare Optimierung (NLP)
        - Genetische Algorithmen (GA): Viele Variablen + Nichtlinearität
        - Dynamische Programmierung (DP): Entscheidungen über mehrere Zeiträume
        - Stochastische Programmierung (evtl. hilfreich wenn Wetterdaten berücksichtigt werden) 

2. **Energieflussgleichgewicht**
    - Wie kann sichergestellt werden, dass die erzeugte Energie (PV-Leistung) und die verbrauchte Energie (Hausverbrauch, Elektrolyse) zu jedem Zeitpunkt im Gleichgewicht sind?
        - Bilanzgleichung in Optimierungsproblem: Energieflussgleichgewicht

3. **Vorhersage der PV-Leistung**
    - Wie können Wetterdaten und historische PV-Leistungsdaten verwendet werden, um genaue Vorhersagen für die PV-Leistung zu treffen?
        - Wetterdaten weisen eine jährliche Saisonalität auf
        - PV-Leistung hat eine circadiane Rhythmik
    - Welche Machine-Learning-Modelle sind am effektivsten für die Vorhersage der PV-Leistung?

4. **Integration und Simulation der Optimierungslösung**
    - Wie kann die Kombination aus Vorhersagemodellen und Optimierungsalgorithmen in einer Simulationsumgebung implementiert werden?
        - Datenvorbereitung: Historische Daten zu PV-Leistung, Hausverbrauch und Wetterbedingungen sammeln und vorbereiten.
        - Vorhersagemodell: Machine-Learning-Modell (z.B. Random Forest) zur Vorhersage der PV-Leistung und des Hausverbrauchs trainieren.
        - Optimierungsalgorithmus: Optimierungsproblem definieren, das die Vorhersagen als Eingangsgrößen verwendet (z.B. lineare Programmierung mit pulp).
        - Integration in Simulationsumgebung: Kombination von Vorhersagemodell und Optimierungsalgorithmus in einer Simulationsumgebung wie SimPy implementieren.
    - Wie können die optimalen Energieflüsse simuliert und validiert werden?
        - Langzeit-Simulation: Simulation über längere Zeiträume (Tage/Monate) zur Beobachtung der langfristigen Auswirkungen der Optimierung.
        - Sensitivitätsanalyse: Überprüfung der Robustheit der Optimierung gegenüber Änderungen in den Eingangsgrößen (z.B. Wetterbedingungen).
        - Vergleich mit historischen Daten: Validierung der Vorhersagemodelle durch Vergleich mit tatsächlichen historischen Daten.
        - Szenarienanalyse: Testen verschiedener Szenarien (z.B. unterschiedliche Anfangszustände der Batterie, variierende PV-Leistung).
        - Visuelle Darstellung: Erstellung von Diagrammen und Grafiken zur anschaulichen Darstellung der Ergebnisse der Simulation.

5. **Berücksichtigung der Systemgrenzen und Kapazitäten**
    - Wie können die physikalischen Grenzen und Kapazitäten der Systemkomponenten (Batterie, PV-Anlage, Elektrolyseur) in die Optimierung einbezogen werden?
        - Definition der Kapazitätsgrenzen und Leistung der Komponenten
        - Anpassung der Optimierung an die aktuellen Betriebsbedingungen und Systemgrenzen in Echtzeit 
    - Wie kann die Lade- und Entladeleistung der Batterie innerhalb der festgelegten Grenzen gehalten werden?
        - Leistungsgrenzen
        - Energieflussgleichgewicht
        - Anpassung der Zielfunktion: Evtl. Einbeziehung von Straf-Termen in die Zielfunktion, um extreme Lade- und Entladezyklen zu vermeiden und die Batterie innerhalb der sicheren Grenzen zu betreiben

6. **Weitere Überlegungen**
    - Gibt es Effizienzverluste oder zeitliche Einschränkungen?

### 3. Mathematische Modellierung

#### 3.1 Variablendefinition

- $P_{\text{PV}}(t)$: Leistung der PV-Anlage zu Zeit $(t)$
- $P_{\text{Haus}}(t)$: Hausverbrauch zu Zeit $(t)$
- $P_{\text{Bat, In}}(t)$: Ladeleistung der Batterie zu Zeit $(t)$
- $P_{\text{Bat, Out}}(t)$: Entladeleistung der Batterie zu Zeit $(t)$
- $P_{\text{Elek}}(t)$: Leistung des Elektrolyseurs zu Zeit $(t)$
- $P_{\text{Wasserstoff}}(t)$: Wasserstoffproduktion zu Zeit $(t)$
- $E_{\text{Bat, Max}}(t)$: Maximale Kapazität der Batterie zu Zeit $(t)$
- $\text{SoC}_{\text{Batterie}}(t)$: Batterieladung zu Zeit $(t)$
- $\eta$ : Wirkungsgrad des Elektrolyseurs (?) 


#### 3.2 Optimierungsziel / Zielfunktion

Maximierung des Wasserstoffoutputs über den betrachteten Zeitraum:

$$
\text{Maximiere} \sum_{t} P_{\text{Wasserstoff}}(t)
$$

$$
\text{Maximiere} \sum_{t=1}^{T} P_{\text{Wasserstoff}}(t) = \eta \sum_{t=1}^{T} P_{\text{Elek}}(t)
$$

#### 3.3 Randbedingungen

**PV System Leistung:**

$$
P_{\text{PV}} = 80 \, \text{kWp}
$$

Kombinierte Last der Wärmepumpe, des Hausverbrauchs und der PV-Leistung als Netto-Last:

$$
P_{\text{Netto}}(t) = P_{\text{Haus}}(t) - P_{\text{PV}}(t)
$$

**Batterie:**

*Maximale Batterie-Leistung:*

$$
E_{\text{Bat, max}} = 80 \, \text{kWh}
$$

*Batterie Input und Output:*

$$
P_{\text{Bat, in}} = P_{\text{Bat, out}} \leq 80 \, \text{kW}
$$

*Sachlogische Grenzwerte (Batterie-Kapazität):*

$$
0 \leq \text{SoC}_{\text{Batterie}} \leq E_{\text{Bat, max}}
$$

### Batteriestatusupdate
$$ 
\text{SoC}_{\text{Batterie}}(t+1) = \text{SoC}_{\text{Batterie}}(t) + \left(P_{\text{Bat, in}}(t) - P_{\text{Bat, out}}(t)\right) \Delta t 
$$

**Elektrolyse Leistung:**

$$
P_{\text{Elek}} = \begin{cases} 
0 \, \text{kW} \\
1.28 \, \text{kW} \leq P_{\text{Elek}} \leq 9.38 \, \text{kW}
\end{cases}
$$

**Energieflussgleichgewicht:**

$$
P_{\text{Netto}}(t) + P_{\text{Batterie, Entlade}}(t) = P_{\text{Batterie, Lade}}(t) + P_{\text{Elek}}(t)
$$

### 4. Lösungsansatz

#### 4.1 Lineare Optimierung

**Voraussetzungen:**
- Daten: Zeitreihen für $P_{\text{PV}}(t)$ und $P_{\text{Haus}}(t)$
- Datenvorbereitung: Leistungsgrenzen definieren

**Weitere Überlegungen:** 
- jährliche Saisonalität der Wetterdaten
- Abhängigkeit von circadianer Rhythmik (e.g. In/Out PV nachts)

**Fragestellung:** 
- Optimierung für Tage oder auch für Monate möglich?

**Simulation der Optimierungslösung**

Mögliche Algorithmen zur Simulation der Optimierungslösung:

- **Monte-Carlo-Simulation:**
    - Simuliert eine große Anzahl möglicher Szenarien basierend auf stochastischen Eingangsgrößen (z.B. PV-Leistung, Hausverbrauch).
    - Passend, um Unsicherheiten und Variabilitäten in den Daten zu berücksichtigen.

- **Agentenbasierte Simulation:**
    - Modelliert individuelle Systemkomponenten (z.B. Batterie, Elektrolyseur) als "Agenten" mit spezifischem Verhalten.
    - Hilfreich zur Untersuchung des Zusammenspiels und der Interaktionen zwischen verschiedenen Systemkomponenten.

- **Deterministische Simulation:**
    - Verwendet feste Eingabewerte (z.B. vorhergesagte PV-Leistung) zur Simulation.
    - Nützlich zur Validierung der Optimierungsergebnisse unter idealisierten Bedingungen.

#### 4.2 Vorhersage der PV-Leistung mittels Machine Learning

**Voraussetzungen:**
- Daten (siehe oben)
- Wetterdaten (open-meteo API)
- Feature Engineering (Tageszeit, Temperatur, Sonneneinstrahlung, Bewölkungsgrad, etc. könnten nützlich sein)

**Ansatz:**
einfache Regressionsmodelle wie Random Forest/Gradient Boosting

**Wichtiger Hinweis:** Dieser Ansatz ist sehr komplex und vor allem fehleranfällig, da eine geringe Vorhersageabweichung über Stunden/Tage akkumuliert. Zudem erfordert es ein hohes Maß an Datenvorbereitung und Feature Engineering.

### 5. Implementierung der Optimierung und Simulation

#### 5.1 Beispielcode

Hier ist ein Beispielcode, der zeigt, wie die Optimierung der Be- und Entladezyklen der Batterie zur Maximierung des Wasserstoffoutputs in Python implementiert werden könnte:

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pulp

# Daten generieren (Beispiel)
np.random.seed(42)
zeitraum = 24
temperatur = np.random.uniform(15, 30, zeitraum)
sonneneinstrahlung = np.random.uniform(0, 1, zeitraum)
bewölkungsgrad = np.random.uniform(0, 1, zeitraum)
tageszeit = np.arange(zeitraum)

# RF-Modell trainieren (Beispiel)
X = np.column_stack((temperatur, sonneneinstrahlung, bewölkungsgrad, tageszeit))
y = np.random.uniform(0, 80, zeitraum)  # PV-Leistung

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, y)
pv_forecast = rf.predict(X)

# Optimierungsproblem definieren
model = pulp.LpProblem("Maximize_Hydrogen_Output", pulp.LpMaximize)

P_Batterie_Lade = pulp.LpVariable.dicts("P_Batterie_Lade", range(zeitraum), lowBound=0, upBound=80)
P_Batterie_Entlade = pulp.LpVariable.dicts("P_Batterie_Entlade", range(zeitraum), lowBound=0, upBound=80)
P_Elektrolyseur = pulp.LpVariable.dicts("P_Elektrolyseur", range(zeitraum), lowBound=0, upBound=9.38)
SoC_Batterie = pulp.LpVariable.dicts("SoC_Batterie", range(zeitraum + 1), lowBound=0, upBound=80)

# Anfangsbedingungen
model += SoC_Batter

ie[0] == 0

# Zielfunktion
model += pulp.lpSum(P_Elektrolyseur[t] for t in range(zeitraum))

# Randbedingungen
for t in range(zeitraum):
    model += pv_forecast[t] + P_Batterie_Entlade[t] == 50 + P_Batterie_Lade[t] + P_Elektrolyseur[t]  # Beispiel Hausverbrauch = 50 kW
    if t < zeitraum - 1:
        model += SoC_Batterie[t + 1] == SoC_Batterie[t] + P_Batterie_Lade[t] - P_Batterie_Entlade[t]

# Modell lösen
model.solve()

# Ergebnisse
for t in range(zeitraum):
    print(f"Stunde {t}: Elektrolyseur Leistung: {P_Elektrolyseur[t].varValue}, Batterie SoC: {SoC_Batterie[t].varValue}")
```

### 6. Validierung des Codes

- Simulation: Durchführung mit unterschiedlichen Parametern
- Sensitivitätsanalyse: Wie wirken sich Veränderungen in den Vorhersagedaten auf die Optimierungsergebnisse aus?

### 7. Weitere Literatur

- Optimal Charge/Discharge Scheduling of Battery Storage Interconnected With Residential PV System. A. Kapoor and A. Sharma, 2020.
