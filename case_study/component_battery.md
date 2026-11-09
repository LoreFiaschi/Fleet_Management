# Traction battery of a battery-electric city bus: degradation family and parameters

Status: done (9 Nov 2026). Sections 1-6 filled; 11 sources entered (B1-B11; B1 groups six short
maker documents) and 8 carried entries (C1-C8). The brief of 9 Nov 2026 includes two points from
the user: (1) the battery has a repair action, the change of some cells, modules or packs; ρ, cost,
downtime and the status of the practice are in section 2.4 and section 4; (2) the bus type is free;
the reference bus is the Mercedes eCitaro 12 m (section 1). Next, if more work is wanted: the
items of section 5 (FTA Report 0253 first) and operator BMS data (section 4).
Downloads: scratchpad downloads\bat\01-22 (08: a truck paper, not used; 09, 10, 21: empty or HTML
error pages); texts in scratchpad text_bat\; my computations in scratchpad scripts_bat\
(`blast_numbers.py`, `line_increments.py`).

Scope: the traction battery of one 12 m battery-electric bus (BEB) type, tracked as one pack, with a
repair action (partial renewal) and a replacement action (new pack). Reference bus: the Mercedes
eCitaro 12 m of line 759 (264 kWh, opportunity charging); the 18 m eCitaro G of line 768 (550 kWh
usable, depot charging) is treated only by scaling (section 2.2). Rates are given per unit of a
physical driver (km, kWh, equivalent full cycle, day), so that any period length can be built from
them.

Abbreviations: SOH = state of health (remaining capacity over initial capacity); EOL = end of life;
EFC = equivalent full cycle (charge throughput divided by the nominal capacity); DOD = depth of
discharge; SOC = state of charge; NMC = lithium nickel manganese cobalt oxide cathode; LFP = lithium
iron phosphate cathode; LTO = lithium titanate anode; BMS = battery management system; CV =
coefficient of variation; ARD∞ = the tool's repair law that multiplies the damage by 1 - ρ.

## 1. Parameter table

**Reference bus: Mercedes-Benz eCitaro 12 m, NMC battery, 264 kWh** (the VBG buses of line 759).
Reasons: it is the case-study vehicle; the maker documents its chemistry, pack design, cycle-life
claim, guarantee and repair offers (B1); a German fleet that is mostly eCitaro-type 12 m buses gives
field energy use (C8). Its empty mass is not public in any source read (only the gross vehicle weight,
20 t, B1e). No other 12 m type read here has public mass, chemistry and consumption together either.

"Evidence" = evidence basis of the source: F = read in full here; P = read in part here; C = carried
from another agent's entry (C1-C8); M = my computation or my choice from the cited inputs.

| Parameter | Value | Unit | Source and location | Evidence | Note |
|---|---|---|---|---|---|
| Damage variable D | capacity fade = 1 - SOH | fraction of nominal capacity | SOH as remaining capacity over initial capacity: B9 Eq. 5, B10 Eq. 1 | F | measured SOH is noisy, see the last row |
| Threshold τ (central) | 0.20 (EOL at 80 % SOH) | fraction | 80 % "by market definition", makers expect it after 6-8 years: C5 p. 26; Shenzhen replaces below 80 %: C7 p. 60; VBZ designs a 20 % degradation reserve: C3 p. 34; EV cells usable above 80 %: B3 §2.2 | C, F | the most cited European value |
| Threshold τ (alternative) | 0.30 (EOL at 70 %) | fraction | "70-80%" typical EOL: B7 §I; 70 % or 80 % by field: B4 §3.3; probable life 8-12 years at 70 % remaining capacity: C8 pp. 9-10; BYD warranty 70 % of 80 % of nominal: C6 p. 3 | F, C | operators in China replaced at 60-85 % SOH (B2 Table 3): EOL is an operator decision |
| Risk level ε | no source | - | no source read gives a risk level for scheduling against EOL; Honda checks lower limits with 99.7 % bands (B5 §4.3) | F | a modelling choice: for example 0.05 or 0.01 |
| Family | Gamma (recommended); Gaussian or Wiener as alternatives | - | section 2 | M | |
| Calendar fade c_cal (central, at about 25 °C, mid-life) | 2.7e-5 (about 1 % per year) | per day | between 0.34 %/year (LFP 250 Ah, B8a) and 1.36 %/year (NMC, B8b), both marginal rates at age 3 years, 25 °C, 50 % SOC | F, M | range 0.9e-5 to 4e-5 per day; eCitaro packs are kept near 25 °C (B1a) |
| Cycle fade k_EFC (central) | 4.0e-5 | per EFC | LFP 250 Ah: 3.4-4.2e-5 per EFC at 1,000-3,000 EFC, 25 °C, DOD 0.8, 0.3C (B8a); maker claim 4,000 cycles to EOL gives 5e-5 per cycle, calendar included (B1a, C8) | F, C, M | range 3e-5 to 1.1e-4 (lower: B8a at 3,000 EFC; upper: trolleybus fleet, calendar included, C1) |
| Field fade per km (check) | 2.4e-7 to 7.8e-7 | per km | 2.4 % (or 7.1 %) per 100,000 km, 105 European buses: B10 §3.3; 3.6 % in 46,380 km, one LFP bus: B9 §4.2 | F | the central model gives 2.5e-7 to 3.1e-7 per km on the long lines (section 2) |
| Energy per km e_j | e_j = 0.66 + 0.065 g_j + P_HVAC / v_j | kWh/km | VBZ standard bus: 0.74 kWh/km at 1.2 % mean grade, 0.91 at 3.8 % (C3 p. 35, chart); HVAC power over speed: B6 §2.6.1 | C, P, M | g_j in %, v_j in km/h, P_HVAC in kW |
| P_HVAC, annual mean | 4.8 | kW | 8 kW average auxiliaries (B6 §4.1) minus 40 % for the eCitaro heat pump (B1b) | F, M | |
| P_HVAC, cold winter day | 14.4 | kW | 24 kW heating at -10 °C, 12 m bus (B6 §2.6.1) minus 40 % (B1b) | F, M | winter +20 to +35 % in a German fleet (C8 p. 10) |
| Field energy use (check) | 1.0-1.4, mean about 1.2 | kWh/km | mostly 12 m city buses, heating included: C8 p. 8; China 0.8-1.2: B2 p. 8 | C, F | the formula gives 0.91-1.30 kWh/km on the 15 lines |
| Nominal energy E_nom | 264 | kWh | case study (VBG story 14 Mar 2023); NMC2 packs of about 33 kWh (search snippet, not read) | C | eCitaro now 444-666 kWh (B1e p. 10) |
| Spread between buses, CV of yearly fade | 0.29 | - | 4.06 ± 1.19 points per year over 110 vehicles: C1 (S4 Table 9) | C, M | the only fleet spread found |
| Gamma rate β | 60 | per unit of fade | β = 1/(CV² τ) = 1/(0.29² · 0.20), section 2 | M | same for every line and period length |
| Increment variance v (Gaussian) | μ_inc / 60 | fraction² | section 2 | M | |
| Wiener σ² per mission period | μ̄_period / 60 | fraction² | μ̄ = fleet-mean increment per period | M | |
| Inverse Gaussian η | 1/(60 μ_inc) | - | section 2 | M | depends on the period length |
| Initial fade μ_0, buses 2-4 years old | 0.10 (range 0.05-0.20) | fraction | fleet mean SOH 89.98 % at the start of a 12-month record: C1 (S4 Table 10); LFP buses 70-90 % after 3-4 years: C2; 3.6 % in the first year of one LFP bus: B9 | C, F, M | the central model of section 2 gives about 0.08 for a line-759 bus in service since February 2023 (about 1,370 days at 6.2e-5 per day) |
| Initial variance v_0 | (0.29 μ_0)², 8.4e-4 at μ_0 = 0.10; Gamma: gamma_beta_0 = 1/(0.29² μ_0) = 119 | fraction² | C1 spread | M | add the SOH measurement error if μ_0 comes from BMS data |
| Replacement value μ_new, v_new | 0, 0 | fraction | a new pack; capacity can start slightly above nominal (B7 Eq. 2; C2) | F, C | |
| Replacement cost C_rep | about 60,000 (range 40,000-110,000) | EUR per 12 m bus | 56,000 EUR discounted (about 63,000 undiscounted, my computation) for a 12 m bus in 2023: C4 (R4 p. 377); 30-50 % of the bus price: B2 p. 5; pack prices 2025: NMC $128/kWh, Europe +56 % over China: B11 | C, F, M | range: 264 kWh × 150-430 EUR/kWh (lower: BNEF 2025 Europe level plus integration, my assumption; upper: 2017 LFP system price, B6 Table 7); labour included in no figure |
| Replacement downtime | no source | days | - | - | gap, section 4 |
| Repair (partial renewal), ρ | 0.17 (one of six packs); module level 0.03-0.10, more for an early-failed module | - | no source gives ρ; my derivation in section 2 | M | maker offers exist (B1c, B1d); one fleet's practice (C7) |
| Repair cost C_R | about (C_rep/6)/0.025 ≈ 400,000 per unit of fade removed, at D = 0.15 | EUR per unit of fade | M from C_rep and ρ = 1/6 | M | the cost of one pack does not depend on D; a linear C_R holds only near the chosen D |
| Opportunity charging modifier | 1.0 (range 0.5-2) | multiplier on k_EFC | narrow DOD slows NMC fade (B7 Eq. 5: ×1.41 at 40 % DOD against ×4.6 at 80 %); fast charging speeds it in a model fitted only up to 0.65C (B8a); operators prefer depot charging for battery life (C4 p. 121); maker: NMC4 robust under 300 kW charging (B1c) | F, C | no field split by charging regime (B10) |
| SOH measurement noise | about ±1 to ±1.5 % | fraction | temperature effect on measured capacity: B3 §2.1; capacity offset 1 %: B9 §3; regeneration after rest: C2 | F, C | affects μ_0 and v_0 if taken from BMS data, not the true fade |

## 2. Recommended family and the formula from line features to the increment per period

### 2.1 Which family

**Recommendation: Gamma.** Reasons:

1. **The true fade is monotone.** The main mechanism, loss of cyclable lithium by growth of the
   solid-electrolyte interphase (SEI), is irreversible, and so is the loss of electrode sites (B7
   §III). The measured SOH is not monotone, but for reasons that are not damage: temperature moves
   the measured capacity by about 0.46 % per °C (B3 §2.1), and capacity "regenerates" after rest
   (C2). The Gamma process has monotone increments; the Wiener process does not (B4 §3.2).
2. **The variance can follow the load of each line.** In the tool, a Gamma increment has mean μ and
   variance μ/β, with β per cell and per mission. One β then fits every line and every period
   length. The tool's Wiener model adds one σ² per mission period, the same on every line, so a
   light line and a heavy line get the same spread.
3. **The independence assumption is acceptable** if the spread is calibrated at the threshold
   (section 2.3).

**Alternatives.**

- **Gaussian** (mean and variance per line): set v = μ/β; it reproduces the Gamma moments but
  allows negative increments.
- **Wiener**: named among the stochastic models for battery prognostics (B4 §2) and chosen by AIC,
  with a random rate per unit, in one maker's cell study (B5). Use σ² = μ̄/β with μ̄ the
  fleet-mean increment per period.
- **Inverse Gaussian**: η = v/μ² = 1/(β μ_inc); η then changes with the period length.
- **Rainflow**: the natural choice if SOC histories per line are simulated. NREL's life models
  already count cycles with the rainflow algorithm and give a fade per cycle from DOD, C-rate and
  temperature (B8c). The tool's Palmgren-Miner reading would need such a damage per cycle; no bus
  source gives one.

**Established practice, one study, or my choice.**

- Established: semi-empirical life models are what "the industry mainly uses" (NREL, B7 §I).
  Calendar fade follows a power of time (t^0.5 to t^0.75, B7 Eq. 4, B8a, B8b).
- Named but not shown as established: Wiener, Gamma and inverse Gaussian processes for batteries
  (B4 §2, with general references only).
- One study: Wiener diffusion plus a random rate per cell fits cell data best by AIC (B5).
- My choice: Gamma, for reasons 1-3 above. No source fits a Gamma process to bus field data.

### 2.2 From line features to the mean increment

For a bus that serves line j for one period of Δ days:

  μ_j(Δ) = Δ · [ c_cal + m_ch · k_EFC · d_j · e_j / E_nom ]

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| d_j | km per vehicle and day on line j | column `km_per_vehicle_wd` (or `_sa`, `_su`) of `zurich_line_features.csv` | case study |
| e_j | energy per km drawn from the battery | e_j = e_0 + k_g g_j + P_HVAC / v_j | below |
| e_0 | energy per km at zero mean grade, 12 m bus, no electric heating | 0.66 kWh/km | two-point fit on VBZ lines 66 and 37 (C3 p. 35) |
| k_g | grade coefficient | 0.065 kWh/km per percentage point | same fit |
| g_j | mean absolute grade of the round trip | (ascent_d0 + descent_d0 + ascent_d1 + descent_d1) / (length_d0 + length_d1), in % | case-study CSV |
| v_j | mean speed | column `mean_speed_kmh` | case-study CSV |
| P_HVAC | heating, ventilation and air conditioning power | 4.8 kW annual mean; 14.4 kW on a cold day | B6, B1b (section 1) |
| E_nom | nominal pack energy | 264 kWh | case study |
| k_EFC | fade per equivalent full cycle | 4.0e-5 | B8a, B1a (section 1) |
| c_cal | calendar fade per day | 2.7e-5 | B8a, B8b (section 1) |
| m_ch | charging-regime multiplier | 1.0 (0.5-2) | section 1 |

Stops per km enter only through the mean speed v_j (more stops, lower speed, more HVAC energy per
km). The VBZ fit has no separate stop term; no source read gives one for a 12 m bus.

Limits of the energy fit. It rests on two lines of one lighter bus (Caetano, 150 kWh) with a diesel
heater (C3). VBZ's "average gradient" is a height difference over a stretch of line (75 m over about
2 km for line 37); g_j is the mean absolute grade of the whole round trip. The two are close for a
line that climbs steadily, not in general. Treat k_g as an order of magnitude.

**Result per line** (my computation, scratchpad `scripts_bat\line_increments.py`; weekday km; the
12 m reference bus on every line, also on line 768, which in reality runs 18 m buses):

| Line | g_j [%] | v_j [km/h] | e_j mean [kWh/km] | e_j cold day [kWh/km] | d_j [km/day] | EFC per day | μ per day | μ per 1,000 km | Years to τ = 0.20 |
|---|---|---|---|---|---|---|---|---|---|
| 739 | 0.90 | 25.3 | 0.91 | 1.29 | 113.9 | 0.39 | 4.3e-5 | 3.8e-4 | 12.8 |
| 742 | 1.97 | 27.3 | 0.96 | 1.32 | 177.3 | 0.65 | 5.3e-5 | 3.0e-4 | 10.4 |
| 749 | 1.92 | 27.8 | 0.96 | 1.30 | 63.2 | 0.23 | 3.6e-5 | 5.7e-4 | 15.1 |
| 759 | 1.07 | 21.8 | 0.95 | 1.39 | 241.9 | 0.87 | 6.2e-5 | 2.6e-4 | 8.9 |
| 766 | 1.00 | 25.1 | 0.92 | 1.30 | 59.0 | 0.20 | 3.5e-5 | 6.0e-4 | 15.6 |
| 768 | 1.65 | 21.1 | 0.99 | 1.45 | 229.2 | 0.86 | 6.2e-5 | 2.7e-4 | 8.9 |
| 769 | 3.09 | 30.0 | 1.02 | 1.34 | 39.7 | 0.15 | 3.3e-5 | 8.4e-4 | 16.5 |
| 771 | 2.35 | 20.6 | 1.05 | 1.51 | 90.8 | 0.36 | 4.1e-5 | 4.6e-4 | 13.2 |
| 772 | 1.63 | 17.6 | 1.04 | 1.58 | 90.3 | 0.36 | 4.1e-5 | 4.6e-4 | 13.3 |
| 773 | 2.80 | 21.1 | 1.07 | 1.52 | 28.2 | 0.11 | 3.2e-5 | 1.1e-3 | 17.4 |
| 774 | 3.85 | 16.0 | 1.21 | 1.81 | 33.0 | 0.15 | 3.3e-5 | 1.0e-3 | 16.6 |
| 781 | 0.95 | 19.6 | 0.97 | 1.46 | 165.0 | 0.60 | 5.1e-5 | 3.1e-4 | 10.7 |
| 787 | 1.05 | 20.0 | 0.97 | 1.45 | 255.6 | 0.94 | 6.5e-5 | 2.5e-4 | 8.5 |
| 795 | 2.34 | 9.9 | 1.30 | 2.27 | 26.1 | 0.13 | 3.2e-5 | 1.2e-3 | 17.1 |
| 797 | 1.56 | 31.5 | 0.91 | 1.22 | 112.1 | 0.39 | 4.3e-5 | 3.8e-4 | 12.9 |

Checks. On the long lines (759, 768, 787) the model gives 2.5-2.7 % per 100,000 km, close to the 2.4 %
of 105 European buses (B10) and below the 7.8 % of one LFP bus in a cold Chinese city (B9). Years to
EOL (8.5-17 years) bracket the makers' 6-8 years (C5) and the probable 8-12 years (C8). The energy
per km (0.91-1.30) brackets the field mean of about 1.2 kWh/km with heating (C8).

**Calendar fade on depot periods.** The tool adds damage only on mission periods. Calendar fade on
depot periods (about 2.7e-5 per day) is then lost; it is small against the uncertainty of c_cal.

**Seasons.** For monthly or weekly periods, use the seasonal P_HVAC in e_j (a transitory profile
can carry the winter months). The fade coefficients stay at their 25 °C values, because the eCitaro
keeps its pack near 25 °C (B1a, B1b); cold mainly raises the energy per km.

**The 18 m eCitaro G (line 768).** No source read gives its energy use. A first approximation
scales e_0 + k_g g_j by the empty-mass ratio 17.3 t / 11.6 t = 1.49 of diesel buses (B6 Table 1)
and uses E_nom = 550 kWh (usable, case study). This is my assumption.

### 2.3 Variance: calibration at the threshold

The only fleet spread found is a random rate per vehicle: yearly fade 4.06 ± 1.19 points over 110
vehicles, CV = 0.29 (C1). Under a random rate, the standard deviation grows like t; under independent
increments (Gamma, Wiener, Gaussian), like √t. I match the two at the threshold τ, where the
reliability row matters most: Var(D at EOL) = (CV τ)². With independent increments the variance
at mean τ is τ/β, so

  β = 1 / (CV² τ) = 1 / (0.29² · 0.20) ≈ 60.

Before EOL this overstates the spread; after, it understates it. B5 supports the growth of the spread
with time on cells, and a random rate per cell as a second term.

Example, one month (30 days) on line 759: μ = 30 · 6.2e-5 = 1.9e-3; Gamma shape βμ = 0.11, rate 60;
variance 3.1e-5. After 8.9 years of the same use: mean 0.20, standard deviation 0.058.

### 2.4 Repair by partial renewal: my derivation of ρ (no source gives a number)

- **What is done in practice.** Makers replaced "often only some cells" after several repair attempts
  (Shenzhen, one fleet: C7). Daimler Buses offers to repair and overhaul the modular packs (B1c,
  B1d) and reconditions NMC1 packs, which "significantly increases the capacity" (B1d), without a
  number. A laboratory study shows that a new cell works well inside a pack of cells aged to 90 %,
  and that the "traditional approach" was to replace all cells (B3). So partial renewal is offered
  by one maker and practised by one fleet; it is not shown to be established, and its gain is not
  measured in any source read.
- **Parallel packs.** The eCitaro carries 4-6 packs of 98 kWh (NMC3) on a 650-750 V system (B1a,
  B1e). A pack of 5,400 cells of 4.93 Ah holds about 96 kWh, so each pack is likely a full-voltage
  unit and the packs work in parallel (my inference; Daimler does not say so). Then replacing k of n
  equally aged packs gives the fade D (n − k)/n, so **ρ = k/n**: 0.17 for one of six packs, 0.25 for
  one of four. This matches the ARD∞ law D → (1 − ρ) D.
- **Modules in series.** Inside a series string, the weakest element limits the usable capacity
  (mechanism used in B3). Replacing the weakest module lifts the pack only to the next weakest one.
  With a CV of 0.10 between modules (the 10 % difference between replicate cells at 0 °C, B7 §II)
  and 50 modules, ρ is about 0.03 for one module and 0.1 for six (my estimate from expected normal
  order statistics: largest of 50 at 2.25 σ, second at 1.85 σ, seventh at 1.13 σ). For a module that failed early, ρ can be much larger.
- **Cost.** One pack of six: about C_rep/6 ≈ 10,000 EUR plus labour (labour: no source). At D = 0.15
  it removes 0.025, so C_R ≈ 400,000 EUR per unit of fade. The tool's C_R is linear in the fade
  removed, while the real cost is per pack; choose C_R at the D where repairs are expected.

## 3. Source entries

### B1. Daimler Buses, maker documents on the Mercedes-Benz eCitaro battery (2022-2026)

- **References** (all downloaded 9 Nov 2026; dates from the page metadata):
  - B1a. Daimler Truck newsroom, press release *Even more capacity, even more range: Mercedes-Benz
    eCitaro ahead of the game with the next generation of high-performance batteries*, 2 May 2022.
    https://www.daimlertruck.com/en/newsroom/pressrelease/even-more-capacity-even-more-range-mercedes-benz-ecitaro-ahead-of-the-game-with-the-next-generation-of-high-performance-batteries-51922838
  - B1b. Same newsroom, *Fully electric Mercedes-Benz eCitaro with innovative technologies: New NMC 3
    high-performance batteries [...]*, 8 May 2023.
    https://www.daimlertruck.com/en/newsroom/pressrelease/fully-electric-mercedes-benz-ecitaro-with-innovative-technologies-new-nmc-3-high-performance-batteries-and-new-thermal-management-increase-capacity-extend-range-and-reduce-energy-consumption-52214919
  - B1c. Same newsroom, *The new NMC4 battery in the Mercedes-Benz eCitaro: More capacity, longer
    service life*, 2 Oct 2025.
    https://www.daimlertruck.com/en/newsroom/pressrelease/the-new-nmc4-battery-in-the-mercedes-benz-ecitaro-more-capacity-longer-service-life-53205023
  - B1d. Same newsroom, *New offers for high-voltage batteries: Daimler Buses wants to use e services
    to continue growing*, 2 Oct 2025.
    https://www.daimlertruck.com/en/newsroom/pressrelease/new-offers-for-high-voltage-batteries-daimler-buses-wants-to-use-e-services-to-continue-growing-53205041
  - B1e. Daimler Buses, *The eCitaro. Technical data and equipment at a glance*, data sheet dated
    04/2026, 13 pages. https://www.mercedes-benz-bus.com/int/en/models/ecitaro/facts-ecitaro.pdf
  - B1f. Mercedes-Benz Buses, product page *eCitaro, battery technology*.
    https://www.mercedes-benz-bus.com/en_DE/models/ecitaro/technology/battery-technology.html
- **Evidence basis.** All six read in full (web pages as extracted text; data sheet text layer, all
  13 pages).
- **Nature.** Maker statements. They show what the maker designs, guarantees and offers, not
  measured field fade.

Findings.

| Item | Value | Location |
|---|---|---|
| Chemistry, all generations | lithium-ion, NMC cathode, graphite anode, liquid electrolyte (NMC3) | B1a, B1b ("new high-energy NMC cathode") |
| Generations | NMC1 (series from 2018), NMC2, NMC3 (from end of 2022), NMC4 (from 2026); a solid-state option existed in between | B1a, B1b, B1c |
| NMC3 pack | cylindrical 21700 cells, 4.93 Ah; 600 cells per module, 9 modules per pack (5,400 cells); 98 kWh per pack; cells kept at about 25 °C by the cooling circuit; maker Akasol | B1a, B1b |
| NMC4 pack | prismatic cells by BMZ Poland; 12 cells per module, 15 modules per pack; 111 kWh per pack (+13 % energy density over NMC3); "a higher share of the stored energy can actually be used" | B1c |
| Packs per bus | 12 m solo: 4-6 packs (NMC3 up to 588 kWh; NMC4 444/555/666 kWh). 18 m G: 4-7 packs (NMC3 up to 686 kWh; NMC4 up to 777 kWh) | B1a, B1c, B1e p. 10 |
| Charging | depot plug (CCS2) up to 150 kW; pantograph or roof rails up to 300 kW | B1a, B1c, B1f |
| Cycle life claim (NMC3) | "Manufacturer Akasol assumes 4,000 charging cycles, depending on factors like the depth of discharge and the individual usage profiles"; with daily use and depot charging "a service life of up to ten years" | B1a, B1b |
| Guarantee (NMC4) | "guarantee a service life of ten years", optional customer-specific warranty up to 15 years | B1c |
| Warranty (NMC4, in new-vehicle sales) | "warranties of up to twelve years" | B1d |
| Range wording | the maximum configuration gives a "reliable range [...] throughout the entire service life of the battery" (NMC3: 280 km solo, 220 km G; NMC4: 600 km solo, 450-480 km G) | B1a, B1b, B1c, B1e p. 8 |
| Fast charging and life | the NMC4 chemistry "demonstrates its advantages, especially during regular quick charging processes with up to 300 kW", with "an extended service life" | B1c |
| Repair (module level) | "The modular concept allows the batteries to be repaired and overhauled during their first service life" (B1c); "a modular design of its batteries, enabling simple repairs" (B1d) | B1c, B1d |
| Reconditioning offer | Daimler Buses "is now reconditioning first-generation NMC1 batteries as required, which significantly increases the capacity"; extension to NMC2 and NMC3 announced; no number for the capacity gain | B1d |
| Full replacement offer | from 2026, NMC1-NMC3 packs replaced by NMC4 packs, "depending on the customer's planned operating period"; fewer packs may give the same capacity | B1d |
| Workshops | more than 120 eBus service centres in Europe, with roof stands and battery cranes "for repairing or replacing complete batteries" | B1c, B1d |
| Heating | heat pump; heating, ventilation and air conditioning need about 40 % less energy than on the diesel Citaro | B1b, B1f |
| Fleet size | more than 3,000 eCitaro delivered since 2018 (B1c); almost 700 in German service in 2023, a quarter articulated (B1b) | B1b, B1c |

- **Not given in any B1 document:** the empty mass, the energy use per km, the end-of-life capacity
  threshold, the price of a pack, and the measured fade of any bus.
- **Case-study link.** The 264 kWh of the VBG eCitaro of line 759 (case_study/zurich_depot_and_lines.md)
  fits 8 packs of the NMC2 generation at about 33 kWh each. That pack size comes from a search snippet
  of an Italian copy of the 2021 data sheet (tplitalia.it), not read; see section 6.

### B2. ICCT 2024, survey of ten Chinese e-bus operators

- **Reference.** Mao, S.; Rodríguez, F. (2024). *Electrifying public transit: Zero-emission bus
  operations, charging, and maintenance in Chinese cities.* Working paper ID 137, International
  Council on Clean Transportation (ICCT), October 2024, 13 pages.
  https://theicct.org/wp-content/uploads/2024/10/ID-137-–-China-e-buses_final2.pdf (downloaded
  9 Nov 2026).
- **Evidence basis.** Read in full (13 pages).
- **Data.** Questionnaire of 2023 to 10 operators in 8 Chinese cities with 9,277 e-buses, 6.4-12 m,
  batteries 57-350 kWh (Table 1, p. 3). Answers are the operators' own statements. Cell chemistry
  not stated.
- **Battery replacement (Table 3, p. 5).** Five operators replaced batteries; replacement came
  "between the third and seventh year of operation" (p. 5).

  | Fleet | Buses with battery replaced | Years before replacement | Capacity lost at replacement | SOH at replacement (my computation) |
  |---|---|---|---|---|
  | Tianjin fleet B | 1 of 24 | 6 | 17 % | 83 % |
  | Dalian | 50 of 90 | 7 | 40 % | 60 % |
  | Harbin | no reply | 6 | 17 % | 83 % |
  | Haikou | 4 of 55 | 3 | 15 % | 85 % |
  | Neijiang | no reply | 6 | 26 % | 74 % |

- **Replacement cost and practice.** 7 of 10 operators "tolerated greater battery degradation and
  delayed battery replacements", for lack of funds; a replacement "typically cost 30%-50% of the
  purchase cost" (p. 5). In 2024 the government announced a subsidy of ¥42,000 for battery
  replacement of e-buses over 8 years old (p. 8, p. 9).
- **Warranty rule.** The official battery warranty requirement of 2015 is 5 years or 200,000 km
  (p. 9). Some fleets reach 200,000 km in 3 years (p. 9).
- **Energy use.** 80-120 kWh/100 km across the surveyed fleets (p. 8). Below 0 °C or above 35 °C
  (daily mean), up to +20 % against 20-25 °C; an earlier ICCT study found +30 % to +50 % (p. 3).
- **Use.** 130-260 km per bus and day (Figure 2, p. 4). Charging power mostly 120 kW DC, 60-300 kW
  (Table 4, p. 6).
- **Repair by module.** Not discussed.
- **Reading.** The capacity lost at replacement spreads from 15 % to 40 %. So in practice the end of
  life is an operator's decision, driven by range needs and money, not one fixed threshold.

### B3. Nenadic, Trabold, Thurston 2020, cell replacement in a small pack (laboratory)

- **Reference.** Nenadic, N.G.; Trabold, T.A.; Thurston, M.G. (2020). Cell Replacement Strategies for
  Lithium Ion Battery Packs. *Batteries* 6(3), 39. https://doi.org/10.3390/batteries6030039 (open
  access).
- **Evidence basis.** Read in full as the publisher's HTML text, through the browser pane (MDPI and
  the RIT repository refused script downloads, HTTP 403). Figures and Tables 1-4 are images and were
  not rendered; numbers below come from the text.
- **Data.** Laboratory. Lithium cobalt oxide (LiCo) 18650 cells, 2.33 Ah nominal; full cycles, 1C
  discharge, 0.7C constant-current constant-voltage charge to 4.2 V, room temperature not controlled
  (§2.1). 16 single cells; packs of 9 cells, 3 in series and 3 strings in parallel (3S3P), passive
  balancing (§2).
- **Findings.**
  - Fade rate to 90 % capacity: 0.11 % per cycle for 16 single cells, 0.10 % per cycle for 9 cells
    in a pack (§2.1, Figure 3). Cells in a matched pack age like single cells (§4).
  - Fade is "not strictly monotonic": the capacity moves 0.46 % per °C of ambient temperature, more
    than four times the fade of one cycle (§2.1, Figure 4b). A few degrees of room temperature
    (about ±3 °C) give about ±1.5 % of the new capacity (abstract).
  - "The traditional approach in pack maintenance is to replace all cells at once to control the
    mismatches. This approach is clearly untenable for very large battery packs" (§1).
  - Repair test (Scenario 1, §2.2): 9 cells aged to 90 % were put in a pack and one was replaced by
    a new cell. The new cell "nicely coexists with the aged cells and did not introduce any obvious
    increase in the overall rate of capacity fade" (§2.2, Figure 8). The pack's coulombic efficiency
    stayed close to that of a new pack, with more frequent drops (§3.1, Figure 12). Pre-aged spare
    cells were not needed (§2.3, §4).
  - EOL wording: "In demanding applications, such as electric vehicles, the cells are considered
    usable only when the capacity is higher than 80% of the nominal capacity" (§2.2). Less demanding
    (stationary) uses tolerate below 70 % (§4).
  - A pack rebuilt from mismatched used cells lost about 8 points of coulombic efficiency (§4); active
    balancing recovered part of it (§3.3).
- **What it does not give.** The pack capacity before and after the repair (so no rho), any cost or
  repair time, and any vehicle data.
- **Relevance.** Laboratory evidence that a pack-level repair by replacing a few cells is technically
  sound. It is not evidence that bus operators do it.

### B4. Patrizi et al. 2024, review of battery degradation models for maintenance

- **Reference.** Patrizi, G.; Martiri, L.; Pievatolo, A.; Magrini, A.; Meccariello, G.; Cristaldi, L.;
  Nikiforova, N.D. (2024). A Review of Degradation Models and Remaining Useful Life Prediction for
  Testing Design and Predictive Maintenance of Lithium-Ion Batteries. *Sensors* 24(11), 3382.
  https://doi.org/10.3390/s24113382 (open access; PDF from the University of Florence repository,
  https://flore.unifi.it/retrieve/29221a2c-7be6-400f-a8cf-b9d61d152c0d/sensors-24-03382-1.pdf).
- **Evidence basis.** Read in full (15 pages; reference list checked for the items cited below).
- **Findings.**
  - The stochastic models used for lithium-ion batteries are "general path models [...], and
    stochastic processes like [...] ARIMA [...], the Wiener process [...], the Brownian motion
    process [...], the gamma process [...], and the inverse Gaussian process" (§2, p. 3).
  - **Caution on this claim.** The four references given for the Wiener, Brownian, Gamma and inverse
    Gaussian processes (refs. 9-12) are works on accelerated degradation test planning in general,
    not fits to battery data (reference list, p. 13). So the review names the families; it does not
    show that any of them is the established model of battery fade.
  - Definitions used (§3.2, p. 6): Wiener increments N(µ k, δ² k); Gamma increments Gamma(α k, β);
    the Gamma process "has the attractive feature of monotonicity"; the Wiener process does not.
  - The general trend of SoH "is always characterized by an exponential decay" (§1, pp. 1-2;
    Figures 1-2, NASA and Toyota data). Double-exponential fits are common (§3.3, eq. 3).
  - Fade can be written against the cumulative charge moved through the battery ("moved charge"),
    whatever the cycle shape, at constant temperature and current and 20-80 % SOC (§3.4, refs. 43-44).
    This supports charge throughput (kWh or EFC) as the time scale of the damage.
  - Failure threshold: "70% or 80% of the rated capacity", by application field (§3.3, p. 8).
  - All the models are developed on single cells; "publicly available datasets of battery pack
    degradation are not available", and pack degradation "is still an open research question"
    (§2, p. 4).
  - The three most used public datasets: NASA (4 cells of 2 Ah, 24 °C, 1C discharge, to 70 %),
    Toyota (140 LFP/graphite cells of 1.1 Ah, fast charge, 30 °C, to 80 %), CALCE (prismatic cells of
    1.1 and 1.35 Ah, to 80 %) (§2, pp. 3-4).
- **No fitted parameters** of any stochastic process are reported.

### B5. Omichi et al. 2026, Honda: Wiener diffusion and random effects fitted to cell ageing tests

- **Reference.** Omichi, K.; Suzuki, K.; Nagatsuka, H.; Yasuda, M. (2026). Technology for Lower Limit
  Prediction of Lithium-ion Batteries Performance after Degradation Employing Diffusion Process
  Model. *Honda R&D Technical Review* 2026, pp. 48-55.
  https://global.honda/en/technical-review/PDF/038/38_47e.pdf (downloaded 9 Nov 2026). An in-house
  technical journal of the maker, not a peer-reviewed journal.
- **Evidence basis.** Read in full (8 pages). Pages 50 and 52 rendered to read Tables 1-2, Eq. (7)
  and Figure 4, which the text layer does not carry.
- **Data.** Laboratory, Honda's own cells, chemistry not stated. 16 cylindrical cells of the 5 Ah
  class cycled between 4.1 V and 3.4 V at 55 °C (§2.1, Figure 2). Training set: 4.5 Ah cylindrical
  cells, 4 months; storage tests at 16 conditions (SOC 30, 55, 80, 95 %; 25, 45, 50, 65 °C) and cycle
  tests at 27 conditions (centre SOC, SOC range, temperature, charge and discharge current); 430
  points. Validation: 12 months, 38 other conditions, 710 points (§3.4).
- **Model.** X(t) = capacity fade, 1 - capacity retention (§2.2). Candidates (Tables 1-2, p. 50):
  measurement error only (M1, least squares); a Wiener diffusion term σ_B dW(η(t)) whose variance
  grows with time ("interindividual variation"); a random effect λ ~ N(µ_λ, σ_λ²) that multiplies
  the trend ("model variation"); and their combinations. Time is transformed as η(t) = t^b (Eq. 1).
  The trend rate depends on SOC, temperature, ΔSOC, charge and discharge current (Table 3).
- **Model choice by AIC (Akaike information criterion; Figure 4, p. 52).** Every model with a
  diffusion term or a random effect beats M1 by a large margin (AIC about -1,390 to -1,427 against
  about -993 to -1,035). Best: M3 = random effect on the trend plus Wiener diffusion, with a
  quadratic trend in SOC and temperature (AIC -1,426.7). Adding measurement error did not improve
  the AIC (§4.1).
- **Fitted values (Eq. 7, p. 52).** dX = λ·µ dη + σ_B dW(η), with µ dη = (a1 SOC + a2 Temp +
  a11 SOC² + a22 Temp² + a0) dη, η = t^b. Values: a0 = -5.13, a1 = 2.6, a2 = -1.7E-3, a11 = -3.2E-1,
  a22 = 1.0E-4, b = 0.55, µ_λ = -1.7E-2, σ_λ = 1.6E-3, σ_B = 7.3E-4. The paper does not state the
  units of SOC, temperature, t and X in this equation, so these values cannot be transferred to a
  bus pack. Only b (time exponent 0.55, close to the square-root law of calendar fade) is
  unit-free.
- **Validation.** Standardized residuals of the 710 validation points: 93.6-94.0 % inside the
  theoretical 95 % band, 98.7-99.0 % inside 99.7 % (§4.3, Figure 6). The predicted band widens over
  10 years (Figure 8).
- **Relevance.** One maker's evidence, on cells, that (a) the spread of fade between nominally equal
  cells grows with time, and (b) a Wiener diffusion plus a random rate per unit describes it better
  than a deterministic curve with noise. It is a single study, not an established standard.

### B6. Göhlich et al. 2018, design of urban electric bus systems (TU Berlin)

- **Reference.** Göhlich, D.; Fay, T.-A.; Jefferies, D.; Lauth, E.; Kunith, A.; Zhang, X. (2018). Design
  of urban electric bus systems. *Design Science* 4, e15. https://doi.org/10.1017/dsj.2018.10 (open
  access, CC BY 4.0; PDF from cambridge.org, downloaded 9 Nov 2026).
- **Evidence basis.** Read in full (28 pages; pp. 25-28 are references, scanned only).
- **Nature.** Simulation and cost model, validated on two 12 m battery buses in Berlin (19 round
  trips of a heavy urban cycle; state of charge predicted within 1 %, §5.2). Energy values below are
  model outputs, except the auxiliary powers, which the authors measured.
- **Energy model (§5.1, Eqs. 1-3).** Resistance force
  F = λ_m m dv/dt + m g f_R cos α + ½ ρ c_W A v² + m g sin α; battery power
  P = F v / η_drive + P_aux; P_aux = P_HVAC + P_others. The values of f_R, c_W, A and λ_m are not
  given in the paper (it points to Ly et al. 2016).
- **Measured auxiliary powers (12 m bus).** Heating to 17 °C at -10 °C ambient: about 24 kW
  (§2.6.1, p. 10). Air compressor up to 5 kW, steering pump up to 2.75 kW, both rarely on; battery
  cooling up to 2.75 kW electric, constant on hot days; battery heater 500 W on cold days (§2.6.2,
  p. 12). Average auxiliary power used in the study: 8 kW; winter: 24 kW (§4.1).
- **Effect of heating on energy per km (§2.6.1, p. 11).** With 1.2 kWh/km for traction and non-HVAC
  auxiliaries ("a plausible value according to our measurements"), a constant 24 kW heater adds
  1.3 kWh/km at 18 km/h (SORT 2) and 2 kWh/km at 12 km/h (SORT 1). Cooling at 35 °C adds about
  12.5 kW, so 0.7 kWh/km at 18 km/h and 1.0 kWh/km at 12 km/h. SORT = standardised on-road test
  cycles of UITP (International Association of Public Transport): SORT 1 heavy urban 12 km/h,
  SORT 2 easy urban 18 km/h, SORT 3 suburban 25 km/h (§3.1).
- **Simulated energy use, 12 m bus, heavy urban Manhattan cycle, 6.32 stops per km (Table 4,
  Table 9).** Average conditions (20 % occupancy, 8 kW auxiliaries): 1.69-1.75 kWh/km with
  opportunity-charged NMC or LTO packs, 1.83 kWh/km with a 300 kWh depot-charged LFP pack. Winter
  (65 % occupancy, 24 kW): 3.25-3.33 kWh/km.
- **Masses (Table 1, p. 5).** Typical empty mass of diesel buses: 12 m 11.6 t (gross 19.5 t);
  18 m articulated 17.3 t (gross 28 t, 29 t for alternative fuels). The authors state that an electric
  bus without traction battery and charging equipment has about the same empty mass (p. 6).
- **Chemistry and depth of discharge.** LFP, LTO (lithium titanate) and NMC are the common bus cells
  (§2.4). Catalogue cycle life at 100 % DoD: LFP 3,000; LTO 10,000-20,000; NMC 1,000-5,000 (Table 2).
  Opportunity-charged buses use a narrow window, "e.g., 40%", depot-charged buses 85-90 % (§2.4,
  p. 9, from interviews with makers); the simulation uses 30-70 % SOC for opportunity charging and
  10-95 % for depot charging (p. 18). LFP suits slow depot charging; NMC suits both (§2.4).
- **Costs (Table 7, 2017 prices).** Battery storage system: NMC 800 €/kWh, LFP 430 €/kWh, LTO
  950 €/kWh, from Kunith (2017), literature and expert interviews. Battery life assumed 6 years, bus
  life 12 years (§6.2).

### B7. Smith et al. 2017, NREL semi-empirical life model of an NMC/graphite cell

- **Reference.** Smith, K.; Saxon, A.; Keyser, M.; Lundstrom, B.; Cao, Z.; Roc, A. (2017). *Life
  Prediction Model for Grid-Connected Li-ion Battery Energy Storage System.* Preprint, 2017 American
  Control Conference, Seattle. NREL/CP-5400-67102, August 2017, 9 pages.
  https://docs.nlr.gov/docs/fy17osti/67102.pdf (the old host www.nrel.gov no longer resolves).
- **Evidence basis.** Read in full (9 pages). Pages 3-4 of the paper (PDF pp. 5-6) rendered to read
  Eqs. 4-11, which the text layer garbles.
- **Data.** Laboratory. 11 Kokam 75 Ah NMC/graphite pouch cells, 9 conditions, about 300-450 days:
  cycling at 0, 23, 30, 45 °C with 80 % or 100 % DOD at 1C; storage at 30-55 °C and 65-100 % SOC
  (Table I, p. 2). Monthly capacity test at C/5.
- **General statements.**
  - "the industry mainly uses semi-empirical lifetime models" (§I, p. 1).
  - "Typically, end-of-life (EOL) is defined when the battery degrades to a point where only 70-80%
    of beginning-of-life (BOL) capacity is remaining" (§I, p. 1).
  - Replicate cells: cells 1 and 2 (23 °C) "aged nearly identically"; cells 6 and 7 (0 °C, severe fade)
    "showed around 10% difference in fade rate" (§II, p. 2).
- **Model (§III).** Capacity Q = min(Q_Li, Q_neg, Q_pos) (Eq. 1). Two parts matter for a bus:
  - Lithium loss (Eq. 4): Q_Li = d0 [b0 - b1 t^½ - b2 N - b3 (1 - exp(-t/τ_b3))], t in days, N in
    cycles. b1 = b1,ref · exp[-(E_a,b1/R)(1/T - 1/T_ref)] · (SOC term through the negative-electrode
    potential U-(t)) · exp[γ_b1 DOD_max^β_b1] (Eq. 5). b2 = b2,ref · exp[-(E_a,b2/R)(1/T - 1/T_ref)]
    (Eq. 6). T_ref = 298.15 K.
  - Negative-electrode site loss by cycling (Eqs. 8-10): dQ_neg/dN = -c2/Q_neg, so
    Q_neg = (c0² - 2 c2 c0 N)^½, with c2 = c2,ref · exp[-(E_a,c2/R)(1/T - 1/T_ref)] · DOD^β_c2.
- **Fitted values (pp. 3-4).** b1,ref = 3.503e-3 day^-½, E_a,b1 = 35,392 J/mol, α_b1 = 1.0,
  γ_b1 = 2.472, β_b1 = 2.157; b0 = 1.07, b3,ref = 2.805e-2, τ_b3 = 5 days; b2,ref = 1.541e-5 per
  cycle, E_a,b2 = -42,800 J/mol; c2,ref = 3.9193e-3 Ah per cycle, β_c2 = 4.54, E_a,c2 = -48,260 J/mol;
  c0,ref = 75.64 Ah, d0,ref = 75.10 Ah.
- **Fit quality.** Capacity RMSE 1.05 Ah (1.4 % of nameplate), R² = 0.99, all cells within ±5 %
  (§III.B, Figure 4).
- **What follows for a bus (my computation from the stated coefficients).**
  - DOD multiplier of the square-root (calendar-type) fade, exp(2.472 · DOD^2.157): 1.41 at 40 % DOD,
    4.6 at 80 % DOD, 11.9 at 100 % DOD. Deep daily cycling (depot charging) thus ages this NMC cell
    about three times faster than a 40 % window (opportunity charging), at equal temperature and SOC.
  - Negative activation energies of b2 and c2: cycling at low temperature ages faster; c2 at 0 °C is
    about 5.9 times its value at 25 °C.
  - The SOC term needs the curve U-(SOC) of the negative electrode, which the paper does not give.
    So b1 cannot be evaluated for a given SOC from this paper alone. Use this model for its
    structure and its DOD and temperature factors, not for absolute rates.
- **Simulated grid application (§V).** 32 °C mean, 45 % mean SOC, 74 % maximum DOD, 69 Ah per day
  (0.92 EFC per day): 7.3 years to 70 % capacity; with seasonal 5-28 °C: 4.9 years; with thermal
  management (20-30 °C): 7.0 years at 74 % DOD and 10 years at 54 % DOD (pp. 6-7, Figure 9).

### B8. NREL BLAST-Lite, coded fade models of a large LFP cell and of an NMC cell

- **Reference.** National Renewable Energy Laboratory (NREL), *BLAST-Lite* (Battery Lifetime Analysis
  and Simulation Toolsuite, light version), open-source Python package, GitHub repository
  NREL/BLAST-Lite (now served from the organisation NatLabRockies), branch `main`, files read on
  9 Nov 2026:
  - B8a. `blast/models/lfp_gr_250AhPrismatic_2019.py` (author P. Gasper, NREL): large-format prismatic
    LFP/graphite cell, more than 250 Ah, "from a large manufacturer"; test data reported in
    https://doi.org/10.1016/j.est.2023.109042 (not read).
  - B8b. `blast/models/nmc111_gr_Sanyo2Ah_2014.py`: the model of Schmalstieg et al. 2014 (J. Power
    Sources 257, 325-334, https://doi.org/10.1016/j.jpowsour.2014.02.012, not read) for Sanyo
    UR18650E NMC cells, 2.15 Ah, "replicated as reported".
  - B8c. `blast/models/degradation_model.py`: shared functions (power-law state update, anode
    potential Ua(SOC), extraction of stressors by rainflow counting).
- **Evidence basis.** The three files read in full as text; nothing was executed. The numbers below
  marked "my computation" come from my own script, which retypes the printed formulas
  (scratchpad `scripts_bat\blast_numbers.py`). The two original data papers were not read; NREL's
  code carries their models.
- **Common form (B8c).** Capacity q = 1 - qLoss_t - qLoss_EFC. Each loss follows a power law,
  y = k x^p, with x = days (calendar part) or EFC or Ah throughput (cycling part); in a varying
  regime the state is advanced with dy = k p (y/k)^((p-1)/p) dx. Stressors per cycle are found by
  rainflow counting of the SOC history: EFC = Σ|ΔSOC|/2, DOD = max SOC - min SOC, C-rate = mean
  absolute rate.
- **LFP, 250 Ah prismatic (B8a).**
  - qLoss_t = k_cal t^0.526 (t in days), k_cal = 8.37e4 · exp(-5210/T) · exp(-3560 Ua(SOC)/T).
  - qLoss_EFC = k_cyc EFC^0.828, k_cyc = (4.38e-8 + 1.55e-8 DOD + 1.68e-7 C-rate) ·
    (exp(2190/T) + exp(-1.55e5/T)), T in K.
  - Valid range stated in the file: cycling 10-45 °C, DOD 0.8-1, charge at most 0.65C, discharge at
    most 1C. "Degradation at higher charging or discharging rates will not be simulated
    accurately." The cycling fade rate "was nearly identical for all cells, despite varying
    temperature, average SOC, and dis/charge rates", so extrapolation outside 10-45 °C is poor.
  - My computation: calendar loss at 25 °C and 50 % SOC 1.1 % after 1 year, 3.7 % after 10 years
    (at 70 % SOC: 1.3 % and 4.4 %; at 10 °C and 50 %: 0.4 % and 1.4 %). Cycle loss at 25 °C,
    DOD 0.8, 0.3C: 5.0 % after 1,000 EFC, 12.5 % after 3,000, 15.9 % after 4,000; at 0.5C: 6.6 %,
    16.4 %, 20.9 %; at 10 °C, 0.3C: 7.4 %, 18.4 %, 23.4 %. The marginal loss at 25 °C, DOD 0.8, 0.3C
    is about 4.2e-5 per EFC at 1,000 EFC and 3.4e-5 per EFC at 3,000 EFC.
- **NMC, Sanyo 2.15 Ah (B8b, Schmalstieg 2014).**
  - qLoss_t = α t^0.75, α = (7.543 V - 23.75) · 1e6 · exp(-6976/T), V = open-circuit voltage at the
    storage SOC.
  - qLoss_Q = β Q^0.5, Q = Ah throughput of one 2.15 Ah cell (charge plus discharge, so Q = 2 · 2.15 ·
    EFC), β = 7.348e-3 (V_rms - 3.667)² + 7.6e-4 + 4.081e-3 DOD.
  - Stated limits: cycle fade "NOT SENSITIVE TO TEMPERATURE OR C-RATE", accurate only near 1C and
    35 °C; knee points seen at high DOD were not fitted.
  - My computation: calendar loss at 25 °C, 50 % SOC: 2.4 % after 1 year, 13.4 % after 10 years
    (35 °C: 5.1 % and 28.6 %). Cycle loss, window 30-70 % SOC (DOD 0.4): 15.8 % after 1,000 EFC,
    31.6 % after 4,000; window 10-90 % (DOD 0.8): 26.7 % and 53.4 %.
  - Reading: this 2014 consumer cell fades far faster under cycling than the maker claim for the
    eCitaro NMC3 pack (4,000 cycles, B1a). Use it only for the shape (square root of throughput,
    DOD factor 4.08e-3 against a base 7.6e-4), not for the size, of NMC bus fade.

### B9. Xu et al. 2022, fade of one LFP city bus from one year of operating data (Changchun, China)

- **Reference.** Xu, N.; Xie, Y.; Liu, Q.; Yue, F.; Zhao, D. (2022). A Data-Driven Approach to State of
  Health Estimation and Prediction for a Lithium-Ion Battery Pack of Electric Buses Based on
  Real-World Data. *Sensors* 22(15), 5762. https://doi.org/10.3390/s22155762 (open access, CC BY;
  full text from Europe PMC, PMC9370969).
- **Evidence basis.** Read in full (text of the Europe PMC XML; Figures 1-31 not seen).
- **Data.** One battery-electric bus, model North BFC6109GBEV5, from the Chinese national supervision
  platform; May 2019 to September 2020; odometer 4,799-68,645 km; sampling every 10 s (§2). Pack:
  372 cells, 186 in series and 2 in parallel, 404 Ah, 598.92 V nominal (about 242 kWh, my
  computation); LFP (§3, §4.2). Charging mostly at constant current near 100 A (about 0.25C), some at
  200 A; discharge peaks within 250 A; currents below 1C (§4.1). Daily operation 5:00-23:00 (§2).
- **Measured fade (§4.2).** Capacity at 25 °C equivalent: 403.91 Ah at 6,482 km (20 May 2019) and
  389.30 Ah at 52,862 km (24 May 2020); initial capacity 405.95 Ah. Fade ΔSOH = 3.601 % in one year
  and 46,380 km. My computation: 3.6 % per year, or 0.078 % per 1,000 km.
- **Shape.** The authors use an "approximate linear degradation law of the lithium iron phosphate
  battery" to back out the initial capacity (§4.2).
- **Temperature.** Fade per cycle was lowest near 20 °C, higher at 40 °C than at 0 °C (§3,
  Figure 10). The weights of their approximate model assume, from the literature, that the fade
  rate doubles every 10 °C away from 20 °C (Eq. 7, §4.1); this is an assumption, not a finding.
- **Capacity measured in the field moves with temperature** (a capacity offset of 4.25 Ah, about
  1 %, between cold and warm charging; §3, Eq. 4). So month-to-month SOH readings are not monotone
  even when the true fade is.
- **Limits.** One bus, one year, one city; partial-charge capacity estimates (65-80 % SOC window).

### B10. Simão, Prytz, Nowaczyk 2022, SOH of 105 battery-electric buses in two European fleets

- **Reference.** Simão, M.; Prytz, R.; Nowaczyk, S. (2022). Long-term Evaluation of the State-of-Health
  of Traction Lithium-ion Batteries in Operational Buses. *International Journal of Prognostics and
  Health Management* 13(1), technical brief. https://doi.org/10.36001/IJPHM.2022.v13i1.3115 (open
  access, CC BY 3.0 US; PDF from papers.phmsociety.org).
- **Evidence basis.** Read in full (7 pages). Figures 2-4 not rendered; numbers from the text.
- **Data (§3.1-3.2, Tables 1-2).** 105 urban battery-electric buses of two European operators, makers
  and chemistries not named (the authors thank the bus maker Caetanobus). Data from the CAN bus
  through a third-party logger. FLT0: June-November 2021; opportunity fast charging, start SOC median
  73 %, ΔSOC median 14 %, charging power median 125 kW (peak 250 kW), daily distance median 266 km.
  FLT1: January 2019 - November 2021; depot charging only, start SOC median 48 %, ΔSOC median 52 %,
  power median 41 kW (peak 125 kW), daily distance median 170 km.
- **Inconsistency in the paper.** §3.2 calls FLT0 the fast-charging fleet and FLT1 the depot fleet;
  §3.3 says the opposite ("Fleet FLT1 uses mostly fast-charging"). The numbers of §3.2 are
  consistent with each other, so I use §3.2.
- **Fade rate (§3.3).** Linear regression of SOH on odometer: mean fade 2.4 % per 100,000 km with the
  authors' method, 7.1 % per 100,000 km with the method of Vichard et al. 2021; "strong variance
  within the fleets", not quantified. The fleet-mean SOH declines roughly linearly (Figure 4).
- **SOH definition.** S(t) = Q(t)/Q(0), usable Ah capacity; Q(0) extrapolated linearly from the data,
  because data start after the beginning of life (§2.1, §2.1.2). The true SOH could not be measured,
  so the accuracy is not known (§2.1.2).
- **Relevance.** The largest European bus sample read here. It gives a fade per km of the right
  order (2.4-7.1 % per 100,000 km), but the method uncertainty is a factor of three, and the paper
  gives no split by charging regime.

### B11. BloombergNEF 2025, battery pack price survey (press release)

- **Reference.** BloombergNEF (BNEF) (2025). *Lithium-Ion Battery Pack Prices Fall to $108 Per
  Kilowatt-Hour, Despite Rising Metal Prices.* Press release, New York, 9 December 2025.
  https://about.bnef.com/insights/clean-transport/lithium-ion-battery-pack-prices-fall-to-108-per-kilowatt-hour-despite-rising-metal-prices-bloombergnef/
- **Evidence basis.** Press release read in full. The full survey is for BNEF clients only; not read.
- **Findings (2025, real US dollars).** Average pack price, all uses: $108/kWh (-8 % against 2024).
  Battery-electric car packs: $99/kWh. Stationary storage: $70/kWh. LFP packs, all segments: $81/kWh;
  NMC packs: $128/kWh. China: $84/kWh; North America and Europe 44 % and 56 % higher (Europe about
  $131/kWh, my computation). 2026 expected lower.
- **Not given.** A price for e-bus packs (the survey covers e-buses, but the segment value is not in
  the press release), and the price of a fitted replacement pack to an operator, which includes the
  pack housing, cooling, integration, margin and labour.

### Carried entries (read by other agents of this case study, 9 Nov 2026)

These entries come from the two files `case_study/maintenance_focus_literature.md` (S-numbers) and
`case_study/maintenance_focus_reports.md` (R-numbers). Their evidence basis is the one stated there;
I did not re-read the sources, except where said. Only battery facts are repeated here.

- **C1 = S4. Malozyomov et al. 2026**, *World Electric Vehicle Journal* 17(7), 377,
  https://doi.org/10.3390/wevj17070377. Read in full by the other agent. 110 trolleybuses with
  traction batteries ("autonomous range", in-motion charging), one Russian city, 12 months; 24.5 % of
  the km on battery; 55,987 km per vehicle and year (Tables 4, 9). Mean SOH 89.98 % → 85.94 % in
  12 months; mean EFC 37.6 → 392.3 (Table 10). **Per vehicle: SOH drop 4.06 ± 1.19 points per year,
  range 1.70-6.90** (Table 9); lowest final SOH 78.9 %. A condition-based replacement scenario uses
  SOH ≥ 80 % as gate (Table 3). Chemistry not stated in the entry. My computation: coefficient of
  variation of the yearly fade between vehicles 1.19/4.06 = 0.29; fade per EFC about
  4.06 %/354.7 = 1.1e-4 per EFC (calendar fade included).
- **C2 = S6. Li et al. 2025**, *Scientific Reports* 15, 24871,
  https://doi.org/10.1038/s41598-025-09108-6. Read in full by the other agent. Nine LFP buses, three
  pack types, about four years (China). LFP604 (about 1,340-1,370 charge segments): SOH about 70 %
  after four years; LFP456: about 80 % after four years, "the three buses age differently"; LFP280
  (about 610-720 segments): above 90 % after three years (p. 9, Figures 7-8). **Capacity is not
  monotone**: it "regenerates" after rest periods (p. 9).
- **C3 = R2. VBZ 2019**, *Elektrisch in die Zukunft: Batteriebusse im Test bei den VBZ*,
  https://vbzonline.ch/content/uploads/sites/3/2019/04/VBZ_eBus_Erkenntnisbericht_Web.pdf. Read in
  full by the other agent; I re-read printed pp. 32-37 (text) and rendered pp. 34-35. Findings:
  - Depth of discharge ("Lade- bzw. Entladehub", charge or discharge swing) has an "überproportional
    negativ" (disproportionately negative) effect on ageing (p. 33).
  - Design split of the installed capacity (figure, p. 34): usable traction energy 50-60 %,
    auxiliaries 10-15 %, residual buffer 10-15 %, **degradation reserve 20 %**.
  - Standard bus Caetano e.City Gold (12 m, 150 kWh), measured through the CAN-FMS interface, May-
    November 2018, diesel auxiliary heater: total consumption "deutlich unter 1 kWh/km" on all lines;
    constant electric auxiliaries 0.38 kW (p. 35). **Chart values (my reading of the bar chart,
    ±0.02 kWh/km): line 37 about 0.91, line 66 about 0.74, line 67 about 0.83, lines 77 and 78 about
    0.76 kWh/km.** Line 66 has an average gradient of 1.2 %, line 37 about 3.8 % (75 m over about
    2 km); VBZ reads this as "den deutlichen Einfluss der Topografie" (the clear influence of
    topography) (p. 35).
  - Neighbourhood bus SOR EBN 8 (8 m, 172 kWh): about 1 kWh/km, drive about 0.85, auxiliaries about
    0.14 kWh/km (0.35 kW); uphill section of line 39 above 1.75 kWh/km, downhill practically zero; total
    of lines 39 and 40 practically equal despite 115 m against 54 m of height difference (p. 36).
    Electric heating in February 2018 would have needed almost 0.8 kWh/km, so about 2.1 kWh/km in
    total (p. 37).
- **C4 = R4. PwC et al. 2024**, BMWK evaluation of the German e-bus funding programme (read in part by
  the other agent). Operators prefer depot charging at lower power, among other reasons because it
  is "positiv auf die Lebensdauer der Batterie" (good for battery life) (p. 121). One LMP
  (lithium-metal-polymer) battery generation was replaced fleet-wide (p. 124). The cost comparison
  assumes one battery replacement at half of a 12-year holding period, 55,000 km per year (p. 238);
  it adds about 56,000 EUR (discounted at 2 %) to a 12 m depot-charged bus in 2023 (table p. 377,
  difference computed by the other agent). Pack size of that case not stated in the entry.
- **C5 = R5. Kollegger et al. 2020**, ESöV 2050 project P-196 for the Swiss Federal Office of
  Transport (read in full by the other agent). EOL at 80 % of the initial capacity "by market
  definition"; "most bus makers" expect EOL after 6-8 years; plan one battery replacement per bus
  life of 12-14 years (p. 26). Planning value 1.7 kWh/km and 70,000 km per year for a 12 m bus on
  mountain lines (p. 35). NMC typical for opportunity charging, LFP for depot charging (Figure 11,
  p. 39). Ageing models must be calibrated per battery type and maker (pp. 25-26).
- **C6 = R6. Eudy and Jeffers 2020**, NREL Long Beach Transit report (read in full by the other agent).
  BYD 40-foot buses, LFP 324 kWh; battery warranty: 70 % of a base capacity for 12 years, the base
  being 80 % of the original 324 kWh (p. 3).
- **C7 = R8. World Bank 2021**, Shenzhen Bus Group case study (read in part by the other agent;
  printed p. 60 = PDF p. 70). "Some buses experienced battery degradation as early as at their
  50,000-kilometer mileage"; most BYD K8 batteries needed replacement after 2-2.5 years, other models
  after about 3-4.5 years; **makers replaced the battery only after several repair attempts, and
  often only some cells**; the report assumes one replacement every four years (p. 60). Battery
  replaced when capacity falls below 80 % (p. 60). Later batches moved from air to liquid cooling
  "to prolong battery life" (p. 50, keyword search).
- **C8 = R11. ThEGA 2025**, real data of the Thuringian zero-emission bus fleet (read in full by the
  other agent). 32 buses, mostly 12 m, among them Mercedes eCitaro, Sileo S12, MAN Lion's City 12E;
  energy use about 1.2 kWh/km (range 1.0-1.4), heating included (p. 8); winter +20 % to +35 %, summer
  +5 % to +10 % (p. 10). Battery: "Geplante Lebensdauer: 6-8 Jahre, 4.000 Zyklen, Wahrscheinliche
  Lebensdauer: 8-12 Jahre (bei einer minimalen Restkapazität von 70%)" (planned life 6-8 years and
  4,000 cycles; probable life 8-12 years at a minimum remaining capacity of 70 %); only moderate
  degradation so far, fewer than 2,000 cycles (pp. 9-10).

## 4. Gaps

| Value | Status | How to fill it |
|---|---|---|
| Risk level ε | no source | a modelling choice; or ask the operator what share of buses may fall below the range need |
| Measured fade of the eCitaro (NMC) in service | no source read | VBG or EUROBUS BMS data (Omniplus On battery monitoring, B1d); German operators with many eCitaro (B1b: about 700 in 2023) |
| Spread of fade between buses of one type and one line | only C1 (trolleybuses, chemistry unknown) and qualitative C2 | per-bus SOH from BMS data of the case-study fleet; B10's 105-bus data set is not public |
| Effect of opportunity fast charging against depot charging on fade | no field split; models disagree (B7, B8a); operators' belief (C4) | per-bus SOH of line 759 (opportunity) against line 768 (depot) |
| Gain ρ of a partial repair (pack or module) | no measurement | ask Daimler Buses / Omniplus for SOH before and after reconditioning (B1d) |
| Cost and labour of a partial repair; downtime of a full or partial pack replacement | no source | Omniplus price list or operator workshop records; FTA Report 0253 (section 5) |
| Pack price for buses, by kWh | only all-segment BNEF values (B11) and 2017 values (B6) | BNEF full survey (clients only); operator tender documents |
| Empty mass of the eCitaro 12 m and G | not in B1 documents | Swiss vehicle registration data or a test report (for example the Omnibusspiegel comparison test named in B1f, not read) |
| Energy use of the eCitaro and eCitaro G on Zurich lines | no source; the formula is calibrated on a lighter 12 m bus (C3) | VBG or EUROBUS energy data per line; SORT test results |
| Stop term in the energy formula | not separated in any source read | per-line energy data with different stop densities |
| Chemistry-specific fade of the eCitaro NMC cells (Akasol, BMZ) | only the 4,000-cycle claim (B1a) | maker data sheet under non-disclosure; NREL BLAST-Lite NMC large-format models (section 6) |

## 5. Could not read

- **FTA Report No. 0253** (2023), *Procuring and Maintaining Battery Electric Buses and Charging
  Systems*, https://www.transit.dot.gov/sites/fta.dot.gov/files/2023-08/FTA-Report-No-0253.pdf:
  HTTP 403 to a script on 9 Nov 2026 (scratchpad downloads\bat\21 holds nothing). Worth reading for
  battery replacement time and cost in US practice. Can you retrieve it?
- **Gao et al. 2017**, *Battery capacity and recharging needs for electric buses in city transit
  service*, Energy 122: the OSTI link https://www.osti.gov/servlets/purl/1342660 served a different
  paper (ORNL trucks, TRR 2628), not used. Worth reading for a bus-specific fade model tied to
  drive cycles.
- **Journal of Energy Storage 2023, https://doi.org/10.1016/j.est.2023.109042**: the test data behind
  the NREL LFP 250 Ah model (B8a). ScienceDirect, not tried (CAPTCHA policy). B8a carries the model.
- **Schmalstieg et al. 2014**, J. Power Sources 257, 325-334 (NMC holistic ageing model): ScienceDirect,
  not tried. B8b carries the model.
- **Naumann et al. 2020**, J. Power Sources 451, 227666 (LFP cycle ageing): no open copy found in the
  TUM repository record; ScienceDirect not tried.
- **Nenadic et al. 2020** (B3): the MDPI and RIT PDF links answered HTTP 403 to scripts; the article
  was read as publisher HTML in the browser pane instead.

## 6. Leads not followed

- **Tang, S. et al. 2014**, *Remaining Useful Life Prediction of Lithium-Ion Batteries Based on the
  Wiener Process with Measurement Error*, Energies 7(2), 520-547, https://doi.org/10.3390/en7020520
  (open access): Wiener fits to cell data with drift and diffusion values. Not read.
- **Xu, X. et al. 2021**, Wiener process under time-varying temperature, Reliability Engineering &
  System Safety 214, 107675 (cited by B5). Not read.
- **Cui, Lu, Han 2024**, two-phase Wiener model, IEEE Access (search snippet: Gamma and inverse
  Gaussian limited to monotone data). Not read.
- **Xu, B. et al. 2018**, *Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment*, IEEE
  Transactions on Smart Grid: rainflow counting with a DOD stress function, the closest battery
  analogue of the tool's rainflow model. Not read.
- **NREL BLAST-Lite, other models**: `nmc_gr_75Ah_A_2019`, `nmc_gr_50Ah_B1_2020`, `nmc_gr_50Ah_B2_2020`,
  `nmc622_gr_DENSO50Ah_2021` (large-format NMC cells, closer to bus cells than B8b). Seen in the
  repository listing only. Not read.
- **Chalmers 2021**, *On the Role of Battery Capacity Fading Mechanism in the Lifecycle Cost of Electric
  Bus Fleet*, https://research.chalmers.se/en/publication/524341: bus fleet management with capacity
  fade; a snippet says a low, narrow SOC range could extend battery life by up to 3 years. Not read.
- **Zenobē**, battery leasing for UK buses: a snippet says replacement usually follows a fall below
  about 70 % SOH (https://www.zenobe.com/insights-and-guides/electric-bus-battery-degradation-what-causes-battery-degradation-and-how-to-manage-it/).
  Commercial web page. Not read.
- **Bern University of Applied Sciences (BFH) 2025**, news story on laboratory tests of bus batteries
  for a degradation model (https://www.bfh.ch/en/news/stories/2025/electric-bus-battery-optimisation/).
  Swiss and relevant; not read.
- **Mercedes eCitaro data sheet 2021 (NMC2)**, Italian copy at
  https://www.tplitalia.it/produttori/Mercedes-Benz/MB_eCitaro_schede_tecniche_2021_09.pdf: a search
  snippet gives 8 NMC2 packs = 264 kWh, about 33 kWh per pack. Not read.
- **Attia et al. 2022**, review of "knees" in lithium-ion ageing trajectories, J. Electrochem. Soc.:
  relevant to late, fast fade after a knee, which a Gamma process with constant rate misses. Not read.
- **ICCT (Mao et al. 2023)**, real-world performance of battery-electric heavy-duty vehicles in China
  (energy use +30 % to +50 % in extreme temperatures, cited in B2). Not read.
