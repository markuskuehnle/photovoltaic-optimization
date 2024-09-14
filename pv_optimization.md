## Optimization of a Photovoltaic System

In this project, I was brought in as a consultant to develop mathematical optimization options for an electrolysis plant of an energy company. By combining simulation techniques and advanced machine learning, my approach aims to maximize energy flows within the system and optimize hydrogen output. This notebook describes in detail how targeted optimization and precise forecasting models improved the energy efficiency and performance of the PV system.

### 1. Problem Description

#### 1.1 System Components

- PV system (photovoltaics)
- Household consumption (including heat pump)
- Battery
- Electrolyzer

#### 1.2 Optimization Goal

The load from the heat pump, household consumption, and PV system performance should be simplified into one value. To balance the system, the battery should either charge or discharge. The electrolyzer should absorb as much power as possible to maximize hydrogen output without drawing power from the grid.

**Analysis period:** 
Only the summer months, during which the PV system generates enough power to make this realistic, should be considered.

**Expected result:**
- Maximum hydrogen output
- Optimal energy flows to achieve this output

### 2. Considered Problems

1. **Optimization of Charging and Discharging Cycles**
   - How can the battery's charging and discharging cycles be optimized to maximize hydrogen output?
   - Which algorithms are most suitable for optimizing battery usage?
     - Linear programming (LP)
     - Non-linear optimization (NLP)
     - Genetic algorithms (GA): Many variables + nonlinearity
     - Dynamic programming (DP): Decisions over multiple periods
     - Stochastic programming (possibly useful if weather data is considered)

2. **Energy Flow Balance**
   - How can we ensure that the generated energy (PV output) and the consumed energy (household consumption, electrolysis) are balanced at all times?
     - Balance equation in optimization problem: energy flow balance

3. **Forecasting PV Output**
   - How can weather data and historical PV output data be used to accurately predict future PV output?
     - Weather data shows yearly seasonality
     - PV output has a circadian rhythm
   - What machine learning models are most effective for predicting PV output?

4. **Integration and Simulation of the Optimization Solution**
   - How can the combination of forecasting models and optimization algorithms be implemented in a simulation environment?
     - Data preparation: Gather and prepare historical data on PV output, household consumption, and weather conditions.
     - Forecasting model: Train a machine learning model (e.g., Random Forest) to predict PV output and household consumption.
     - Optimization algorithm: Define an optimization problem that uses these forecasts as input (e.g., linear programming with pulp).
     - Integration into simulation environment: Implement the combination of forecasting model and optimization algorithm in a simulation environment like SimPy.
   - How can the optimal energy flows be simulated and validated?
     - Long-term simulation: Simulate over extended periods (days/months) to observe the long-term effects of the optimization.
     - Sensitivity analysis: Test the robustness of the optimization against changes in input variables (e.g., weather conditions).
     - Comparison with historical data: Validate the forecasting models by comparing them to actual historical data.
     - Scenario analysis: Test various scenarios (e.g., different initial battery states, varying PV output).
     - Visual representation: Create charts and graphs to visually represent the simulation results.

5. **Considering System Boundaries and Capacities**
   - How can the physical limits and capacities of the system components (battery, PV system, electrolyzer) be included in the optimization?
     - Define capacity limits and performance of components
     - Adjust the optimization to current operating conditions and system limits in real-time
   - How can the charging and discharging of the battery be kept within the established limits?
     - Performance limits
     - Energy flow balance
     - Adjust the objective function: Possibly include penalty terms to avoid extreme charging/discharging cycles and operate the battery within safe limits

6. **Other Considerations**
   - Are there efficiency losses or time constraints?

### 3. Mathematical Modeling

#### 3.1 Variable Definitions

- $P_{\text{PV}}(t)$: Power from PV system at time $(t)$
- $P_{\text{House}}(t)$: Household consumption at time $(t)$
- $P_{\text{Bat, In}}(t)$: Battery charging power at time $(t)$
- $P_{\text{Bat, Out}}(t)$: Battery discharging power at time $(t)$
- $P_{\text{Electro}}(t)$: Electrolyzer power at time $(t)$
- $P_{\text{Hydrogen}}(t)$: Hydrogen production at time $(t)$
- $E_{\text{Bat, Max}}(t)$: Maximum battery capacity at time $(t)$
- $\text{SoC}_{\text{Battery}}(t)$: Battery state of charge at time $(t)$
- $\eta$: Efficiency of the electrolyzer (?)

#### 3.2 Optimization Goal / Objective Function

Maximize hydrogen output over the analysis period:

$$
\text{Maximize} \sum_{t} P_{\text{Hydrogen}}(t)
$$

$$
\text{Maximize} \sum_{t=1}^{T} P_{\text{Hydrogen}}(t) = \eta \sum_{t=1}^{T} P_{\text{Electro}}(t)
$$

#### 3.3 Constraints

**PV System Output:**

$$
P_{\text{PV}} = 80 \, \text{kWp}
$$

Combined load of the heat pump, household consumption, and PV output as net load:

$$
P_{\text{Net}}(t) = P_{\text{House}}(t) - P_{\text{PV}}(t)
$$

**Battery:**

*Maximum battery capacity:*

$$
E_{\text{Bat, max}} = 80 \, \text{kWh}
$$

*Battery input and output:*

$$
P_{\text{Bat, in}} = P_{\text{Bat, out}} \leq 80 \, \text{kW}
$$

*Logical bounds (battery capacity):*

$$
0 \leq \text{SoC}_{\text{Battery}} \leq E_{\text{Bat, max}}
$$

**Battery state update:**

$$
\text{SoC}_{\text{Battery}}(t+1) = \text{SoC}_{\text{Battery}}(t) + \left(P_{\text{Bat, in}}(t) - P_{\text{Bat, out}}(t)\right) \Delta t
$$

**Electrolyzer power:**

$$
P_{\text{Electro}} = \begin{cases} 
0 \, \text{kW} \\
1.28 \, \text{kW} \leq P_{\text{Electro}} \leq 9.38 \, \text{kW}
\end{cases}
$$

**Energy flow balance:**

$$
P_{\text{Net}}(t) + P_{\text{Battery, Discharge}}(t) = P_{\text{Battery, Charge}}(t) + P_{\text{Electro}}(t)
$$

### 4. Solution Approach

#### 4.1 Linear Optimization

**Requirements:**
- Data: Time series for $P_{\text{PV}}(t)$ and $P_{\text{House}}(t)$
- Data preparation: Define performance limits

**Additional considerations:**
- Annual seasonality of weather data
- Circadian rhythm dependence (e.g., In/Out PV at night)

**Key Question:**
- Is optimization possible for both days and months?

**Simulation of Optimization Solution**

Possible algorithms for simulating the optimization solution:

- **Monte Carlo Simulation:**
  - Simulates a large number of possible scenarios based on stochastic input variables (e.g., PV output, household consumption).
  - Suitable for considering uncertainties and variabilities in the data.

- **Agent-based Simulation:**
  - Models individual system components (e.g., battery, electrolyzer) as "agents" with specific behaviors.
  - Useful for examining interactions between various system components.

- **Deterministic Simulation:**
  - Uses fixed input values (e.g., predicted PV output) for simulation.
  - Helpful for validating optimization results under ideal conditions.

### 4.2 Predicting PV Output Using Machine Learning

**Requirements:**
- Data (see above)
- Weather data (open-meteo API)
- Feature Engineering (Time of day, temperature, solar irradiance, cloud cover, etc., may be useful)

**Approach:**
Simpler regression models like Random Forest or Gradient Boosting

**Important Note:** This approach is quite complex and particularly error-prone, as even minor prediction errors accumulate over hours/days. It also requires a high degree of data preparation and feature engineering.

### 5. Implementing Optimization and Simulation

Here is a sample code that shows how to implement the optimization of battery charge and discharge cycles to maximize hydrogen output in Python:

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pulp

# Generate data (example)
np.random.seed(42)
time_period = 24
temperature = np.random.uniform(15, 30, time_period)
solar_irradiance = np.random.uniform(0, 1, time_period)
cloud_cover = np.random.uniform(0, 1, time_period)
time_of_day = np.arange(time_period)

# Train RF model (example)
X = np.column_stack((temperature, solar_irradiance, cloud_cover, time_of_day))
y = np.random.uniform(0, 80, time_period)  # PV Output

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, y)
pv_forecast = rf.predict(X)

# Define optimization problem
model = pulp.LpProblem("Maximize_Hydrogen_Output", pulp.LpMaximize)

P_Battery_Charge = pulp.LpVariable.dicts("P_Battery_Charge", range(time_period), lowBound=0, upBound=80)
P_Battery_Discharge = pulp.LpVariable.dicts("P_Battery_Discharge", range(time_period), lowBound=0, upBound=80)
P_Electrolyzer = pulp.LpVariable.dicts("P_Electrolyzer", range(time_period), lowBound=0, upBound=9.38)
SoC_Battery = pulp.LpVariable.dicts("SoC_Battery", range(time_period + 1), lowBound=0, upBound=80)

# Initial conditions
model += SoC_Battery[0] == 0

# Objective function
model += pulp.lpSum(P_Electrolyzer[t] * pv_forecast[t] for t in range(time_period))

# Constraints
for t in range(time_period):
    model += SoC_Battery[t + 1] == SoC_Battery[t] + P_Battery_Charge[t] - P_Battery_Discharge[t]
    model += P_Battery_Charge[t] - P_Battery_Discharge[t] >= 0
    model += P_Electrolyzer[t] <= P_Battery_Discharge[t]

# Solve the problem
model.solve()

# Output results
print("Battery Charging and Discharging Schedule:")
for t in range(time_period):
    print(f"Hour {t}: Charge {P_Battery_Charge[t].varValue}, Discharge {P_Battery_Discharge[t].varValue}, Electrolyzer {P_Electrolyzer[t].varValue}")
```

### 6. Validation

- Simulation: Conduct simulations with different parameters.
- Sensitivity Analysis: How do changes in forecast data affect the optimization results?