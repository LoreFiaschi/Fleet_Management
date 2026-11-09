# Component: traction inverter power modules (IGBT or SiC MOSFET)

Status: done (9 Nov 2026). Eight sources read in full (S1-S8). Parameter table, recommended model, formula, calibration limits, gaps, unread items and leads are written. No bus-specific mission-profile lifetime study was found; the absolute damage level rests on a calibration to a maker's design target (Section 2.4). Brief as changed by the coordinator on 9 Nov 2026: the bus type is free (choose the one with the best public inverter data); the lines near Zurich stay. Next, if work resumes: the items under "Could not read" (Amirpour et al. 2021; Kundu et al. 2020; Semikron "Pimp your electric drive"), which could replace the calibration.

## 1. Parameter table

Abbreviations: IGBT = insulated-gate bipolar transistor; SiC = silicon carbide; MOSFET = metal-oxide-semiconductor field-effect transistor; ΔT_j = junction-temperature swing of one cycle; T_j,min = lowest junction temperature of the cycle (about the coolant temperature); t_on = heating time of one cycle; N_f = cycles to failure; EOL = end of life. "My computation" = computed from the formulas and inputs stated in this file (the S5 model with the sintered-chip parameters, and the line data of `zurich_line_features.csv`); no number comes from an unread source.

| Parameter | Value | Unit | Source and location | Evidence basis | Note |
|---|---|---|---|---|---|
| Reference bus and inverter | Battery-electric city bus, 12 m (line 759) and 18 m (line 768), with an IGBT inverter of the Semikron SKAI 2 HV family (sintered chips, AlCu bonds, up to 800 V DC, 300 kVA, 350 A rms) | - | S2; S8 | read in full | No bus model is named with SKAI 2 HV in the sources read. Vehicle data (masses, drive) come from the eCitaro (S1, S7), whose ZF inverter is SiC with no public module data. See Section 2. |
| Damage variable | Miner sum D = Σ n_i / N_f,i over the junction-temperature cycles; the chip (IGBT or diode) with the largest D decides | - | S3 Eq. (15)-(16), p. 252; scoping section "Inverter power modules" | read in full | Failure of the module at D = 1 on the chosen N_f curve |
| Lifetime model N_f | N_f = A0 · A1^β · ΔT_j^(-β) · ΔT_j^α · exp(E_a/(k_B T_jm)) · (C + t_on^γ)/(C + 2^γ) · k_th, β = exp(-(ΔT_j - T0)/λ) | cycles | S5 Sec. 2.1, p. 3 | read in full | Curves at **15 % failure probability** (S5 Sec. 1.2) |
| Model parameters (sintered chips) | A0 = 2.05e11; A1 = 60; T0 = 38 K; λ = 17 K; α = -4.3; E_a = 2.54e-20 J; C = 1.44; γ = -1.21; k_th = 1 (IGBT ≤ 1200 V), 0.33 (SiC ≤ 1200 V) | - | S5 Table 3, p. 8; Sec. 2.5, p. 5 | read in full | Fitted to Al bond wires; SKAI 2 HV has AlCu bonds (S8), so likely conservative |
| EOL criterion behind N_f | +20 % of V_CE (or V_DS), or of R_th, or of ΔT_j | - | S5 Sec. 1.2, p. 2 | read in full | S6 used +5 % V_CE,on; "5 %-20 %" is typical (S6 Sec. IV) |
| ΔT_j,ref per stop (design case) | 40 | K | my computation: calibrated so that D = 1 after 60,000 operating hours (S2) at 46 stops per hour | derived | **Assumption-driven.** No source gives ΔT_j of a bus inverter per stop |
| Design life used for the calibration | 60,000 | operating hours | S2 (Semikron design target for city-bus inverters: "60000 and more") | read in full | A maker's design target, not a measured life |
| T_j,min | 60 | °C | assumption; S3 p. 251 used a heatsink at 60 °C for a car | read in full (S3) | ±10 K changes the damage by a factor 1.15 / 0.86 (my computation, S5 model) |
| t_on per cycle | 10 | s | assumption (one bus acceleration) | - | 5 s or 20 s change the damage by -5 % / +2 % (my computation) |
| Cycles per stop n_c | 2 | cycles | assumption; S3 Fig. 19, p. 253-254 shows one cycle in acceleration and one in braking | read in full (S3) | |
| Damage per stop, design case, flat road | 3.6e-7 | per stop | my computation (n_c / N_f at ΔT_j,ref) | derived | |
| Grade factor G | 1.16 (line 759), 1.38 (line 768) | - | my computation from ascent and length in `zurich_line_features.csv` | derived | Half the stops uphill, half downhill, acceleration 1.0 m/s² (assumption) |
| **Mean increment per km** (design case) | 8.3e-7 (759); 1.16e-6 (768) | per km | my computation: 3.6e-7 × stops/km × G | derived | Car cycles give 4.6e-7 (CLTC) to 1.5e-6 (WLTP) per km (S3 Table VII); same order |
| Mean increment per trip (direction 0) | 1.2e-5 (759, 28 stops); 6.5e-6 (768, 13 stops) | per trip | my computation | derived | |
| **Mean increment per weekday** (design case) | 2.0e-4 (759, 241.9 km); 2.65e-4 (768, 229.2 km) | per day | my computation | derived | Life at this rate every day: 13.6 y (759), 10.3 y (768) |
| Effect of a lighter actual load | operating mass 0.8 × design mass: increments × 0.23 (759) or × 0.24 (768); 0.6 ×: × 0.02 to 0.03 | - | my computation (ΔT_j scales with mass) | derived | The absolute level is uncertain by about two orders of magnitude |
| Support b of the increment | about 1.7 × the mean (b = 3.4e-4 for 759, 4.5e-4 for 768, per weekday) | per day | assumption: +50 % unplanned stops and +10 K coolant on the worst day (1.5 × 1.15) | - | No source gives the largest increment |
| Variance v of the increment | ≤ b²/4 (upper bound for any variable on an interval of width b) | per day² | Popoviciu bound; spec condition v ≤ b²/4 | - | **No source** gives the variance of the damage under the same use |
| Bound method | Hoeffding (needs mean and support only) | - | my recommendation | - | Daily periods, design case, τ = 0.57 to 0.81, ε = 0.01 to 0.001: Hoeffding uses only 4-7 % of τ (my computation) |
| Unit-to-unit scatter (Weibull shape k) | 13.5 to 24 (one industrial IPM, ΔT_j = 81 K); a spread of about 3× at ΔT_j = 30 K suggests k ≈ 4-5 | - | S6 Table IV (my computation of k); S5 Fig. 3, p. 5 (my reading; a factor 3 between the 1 % and the 63 % life means k ≈ 4.2) | read in full | No value for traction modules found. The S5 points come from several test conditions, so k ≈ 4-5 is a rough reading |
| **Threshold τ** (on the 15 % curve) | 0.57 (k = 5, unit-failure probability 1 %); 0.81 (k = 13.5) | - | my computation: τ = (ln(1/(1-p)) / ln(1/0.85))^(1/k) | derived | Puts the unit-to-unit scatter into τ, not into the increments (Section 2) |
| ε | 0.01 (my choice) | - | **no source** suggests a risk level | - | Sources only use lifetime definitions: B1, B10 (S6), 15 % curves (S5) |
| Initial damage μ0 | 0 for a new inverter; else days in service × mean daily increment (for example 0.27 after 3.75 years on line 759 in the design case) | - | my computation | derived | |
| Replacement value | μ_new = 0 (`replacement_as_new = true`) | - | standard for a new part | - | Replaced unit: one inverter (the ZF axle has two stand-alone inverters, S7); module-level replacement not documented |
| Replacement cost | **no source** | CHF | - | - | Gap |
| Downtime of a replacement | **no source** | days | - | - | Gap |
| Repair (imperfect) | **none found**: set ρ = 0 | - | S5, S6 and S3 describe only wear-out to EOL | read in full | Active thermal control lowers future increments; it does not lower D |

## 2. Recommended model and formula from line features to the damage increment

### 2.1 Which bus and which inverter (question 1)

| Candidate | Inverter maker and type | Module data public | Lifetime model public | Bus mission-profile study |
|---|---|---|---|---|
| Mercedes eCitaro 2026 (S1) with ZF AxTrax 2 LF (S7) | ZF, two 800 V **SiC** inverters, 520 A DC | no (maker, module, chip attach unknown) | no | none found |
| City buses with Semikron **SKAI 2 HV** (S2, S8) | Semikron, **IGBT**, sintered chips, AlCu bonds, 800 V DC, 300 kVA, 350 A rms | chip-attach and bond technology, ratings | **yes**, same maker, sintered-chip parameter set (S5) | none found; > 53,000 city buses equipped, 800 million field hours (S2) |
| Passenger EV with Infineon HybridPACK 2 (S3) | Infineon FS820R08A6P2B, IGBT, 750 V / 820 A | module named (its datasheet not read) | yes (S3 Table V) | car cycles only (CLTC, WLTP) |

- **Choice**: the inverter is a **Semikron SKAI 2 HV-type IGBT inverter with sintered chips**; the vehicle is a 12 m or 18 m battery-electric city bus with the masses and drive of the eCitaro (S1, S7). Reason: it is the only city-bus traction inverter found whose maker publishes the semiconductor type, the chip-attach technology, the ratings, the bus field use and a power-cycling model with parameters for that technology.
- The bus is therefore a composite: no source read names a bus model that carries SKAI 2 HV.
- A SiC variant (the eCitaro's ZF inverter) can be run with the same model and k_th = 0.33 (S5 Sec. 2.5). At the same ΔT_j this triples the damage (my computation). SiC chips may also see a different ΔT_j; no source read gives it.

### 2.2 Model family and reasons

- **Family: rainflow** (Palmgren-Miner damage), as the scoping document fixes for power modules.
- **Bound: Hoeffding.** It needs only the mean and the support of the increment. No source gives the variance of the damage under the same use, while a support follows from a worst-case day. Over the thousands of periods before replacement, the bound costs only 4-7 % of τ (Section 1).
- **Two kinds of randomness, kept apart** (my construction):
  1. **Usage randomness** from period to period (passenger load, extra stops, coolant temperature). It is roughly independent between periods, as assumption A1 of the spec requires. It enters the increments (mean, support) and ε.
  2. **Unit-to-unit scatter** of the module strength (the Weibull scatter of N_f). It is **one random factor per module**, common to all its periods. Put into the increments, it would break assumption A1. So it goes into τ: with the damage computed on the 15 % curve of S5 and a Weibull shape k, the probability that the module fails before D reaches τ is p = 1 - exp(-ln(1/0.85) · τ^k). Solving for τ gives the table entry τ = (ln(1/(1-p)) / ln(1/0.85))^(1/k). The total risk is then at most p + ε (union bound).
- **No repair**: no source describes an action that lowers the damage of a power module. The only action is the replacement of the inverter (or of its module).

### 2.3 Formula from line features to the increment per period

For a bus serving line ℓ in period P (day, week or month):

- μ_inc(ℓ, P) = d_ref · G_ℓ · s_ℓ · km_ℓ(P) · F_T(P)
- d_ref = n_c / N_f(ΔT_j,ref, T_j,min, t_on) = 2 / 5.5e6 = **3.6e-7 per stop** (design case, flat road).
- s_ℓ = stops per km: column `stops_per_km` of `zurich_line_features.csv` (1.99 for 759, 2.31 for 768).
- km_ℓ(P) = km driven on line ℓ in the period: `km_per_vehicle_wd` (and the Saturday and Sunday columns) times the days.
- G_ℓ = grade factor = ½ [N_f(ΔT_ref) / N_f(ΔT_ref (1 + x_ℓ)) + N_f(ΔT_ref) / N_f(ΔT_ref (1 - x_ℓ))], with x_ℓ = g · ḡ_ℓ / a, ḡ_ℓ = (ascent_d0 + ascent_d1) / ((length_d0 + length_d1)/2), a = 1.0 m/s². It assumes half the stops uphill and half downhill on the mean uphill grade. Result: G = 1.16 (759, ḡ = 1.06 %), 1.38 (768, ḡ = 1.67 %).
- F_T(P) = exp(E_a / k_B · (1/T_jm,ref - 1/T_jm(P))): the effect of a coolant temperature different from 60 °C. Per ±10 K: × 1.15 / × 0.86 (sintered set). The link between ambient and coolant temperature has no source here.
- Optional mass factor: replace ΔT_ref by r · ΔT_ref, with r = operating mass / design mass. It assumes that the chip losses, and so ΔT_j, scale with the traction force (current proportional to torque, below the torque limit). With r = 0.8 the damage falls by a factor of about 4; with r = 0.6, by a factor of 30 to 50.
- Support: b(ℓ, P) ≈ 1.5 × 1.15 × μ_inc(ℓ, P) (assumption, see the table). Hoeffding input: `support` = b; R grows by b² per period.

Worked numbers (design case, weekday, F_T = 1):

| Line | Stops/km | G | Damage per km | Damage per trip (dir. 0) | Damage per weekday | Days to D = 0.57 |
|---|---|---|---|---|---|---|
| 759 (12 m) | 1.99 | 1.16 | 8.3e-7 | 1.2e-5 | 2.0e-4 | about 2,800 (7.7 years) |
| 768 (18 m) | 2.31 | 1.38 | 1.16e-6 | 6.5e-6 | 2.65e-4 | about 2,150 (5.9 years) |

### 2.4 How reliable these numbers are

- The **ratio between lines** (stop density, grade) rests on the structure "damage per stop" and on the slope of the S5 model. It is the most reliable part.
- The **absolute level** rests on one calibration: a maker's design target of 60,000 operating hours (S2), met exactly at the design load, at 46 stops per hour. The real ΔT_j per stop is unknown. A ΔT_j 10 % higher or lower changes the damage by × 1.8 or × 0.5 (my computation); a lighter load can lower it by one to two orders of magnitude.
- A low damage rate agrees with the fleet evidence in `maintenance_focus_reports.md`: inverters cause about 0.1-0.3 % of all defects (Toronto evaluation, R3), and no source in that file names wear-out of the power modules.
- For the scheduling study, I recommend the design case as the **conservative case** and r = 0.8 as a **sensitivity case** in which the inverter rarely limits the schedule.

### 2.5 Calibration limits of the lifetime models (question 5)

| Model | Range of the test data | Use here | Source |
|---|---|---|---|
| CIPS08 (Bayerer et al.) | ΔT_j 45-150 K; T_j,max 80-205 °C; "can not be used for extrapolation"; inaccurate at low current per bond and low voltage class | City-cycle swings are mostly below 40 K (S3 Fig. 16): **extrapolation**. Not used for the numbers above | S4 Secs. 2.2, 3.1, 3.5 |
| Semikron Danfoss AN 21-001 | DC tests ΔT_j 50-120 K; inverter tests ΔT_j 30-63 K; T_jm 32-127 °C; Al bond wires | ΔT_j,ref = 40 K lies inside the inverter-test range. Downhill stops (ΔT_j ≈ 33-36 K) are near its lower end. Below 30 K the "low ΔT_j extension" grows fast (× 24 at 20 K, × 213 at 15 K; my computation) and is **extrapolated** | S5 Secs. 1.4, 2.4, Fig. 3 |
| Infineon HybridPACK 2 model used by Lu et al. | Not stated in S3; the coefficients come from an Infineon tutorial (Thoben 2016), not read | Used only as a cross-check | S3 Sec. IV-A |
| Choi et al. scatter | One 600 V / 30 A industrial IPM, ΔT_j ≈ 81 K, 6-8 samples per condition | Shape k is likely too high for traction modules at low ΔT_j | S6 Secs. III-IV |
| Any accelerated-test model | Tests at large ΔT_j, where plastic deformation dominates, are applied to small field swings, where elastic deformation grows | General warning | S5 Sec. 2.4, p. 4; scoping section (Alavi et al. 2024, Sec. 4) |

Other limits:
- The calibration to 60,000 hours (S2) is a design target. S2 does not say at which failure probability, load or stop density it holds.
- Miner's linear sum ignores the order of the cycles; no source read quantifies this for buses.
- SiC: S5 covers SiC with one factor (k_th = 0.33); no SiC field data were read.

## 3. Source entries

### S1. Daimler Buses, *The eCitaro. Technical data and equipment at a glance*, 04/2026

- URL: https://www.mercedes-benz-bus.com/int/en/models/ecitaro/facts-ecitaro.pdf (downloaded 9 Nov 2026, 2,130,058 bytes, 13 pages). The mirror on `showroom.mercedes-benz-bus.com` failed the TLS name check.
- Evidence basis: read in full (13 pages).
- Findings:
  - Electric drive of every variant (eCitaro K, 12 m with 2 or 3 doors, G with 3 or 4 doors): **ZF AxTrax 2 LF** (p. 6, table "Transmission").
  - Power, peak / duration: 440 / 360 kW. Maximum torque 2 x 525 Nm. Transmission ratio 2 x 11.896; final ratio 22.66. Nominal voltage 650 / 750 V (p. 6).
  - Gross vehicle weight: 20,000 kg for the 12 m bus, 30,000 kg for the G with 3 doors (p. 9).
  - Battery: NMC4 modules of 111 kWh, 4 to 7 modules (p. 10).
  - Date of the document: 04/2026 (p. 13).
  - The data sheet says nothing on the inverter, its maker or its semiconductors.
- Limits: the data sheet describes the bus sold in 2026. Buses delivered earlier (for example the VBG buses of line 759, delivered from 2023) may have the older ZF AVE 130 axle. Search snippets say so (Daimler Truck press releases, not read), so this is a lead, not a fact.

### S2. Backhaus, K. (Semikron), *Durable Traction Inverters for Electrified Heavy Duty Vehicles*, Semikron technical article, 3 Jan 2019

- URL of the original: https://www.semikron.com/de/ueber-semikron/fachartikel/durable-traction-inverters-for-electrified-heavy-duty-vehicles.html. It now redirects to a landing page of Semikron Danfoss without the article. Read in the Wayback Machine copy of 23 Oct 2019 (http://web.archive.org/web/20191023011453/...), opened in the browser pane; the plain download was blocked by the archive as "suspected abusive bot traffic". The PDF version (516.79 KB) linked on the page was not opened.
- Evidence basis: read in full (the web text; the two figures are product photos).
- Findings (all from the one page of text):
  - City buses and delivery trucks have "much higher frequency of start and stop, accelerate and decelerate" than passenger cars. This raises the power-cycling requirement of the traction inverter.
  - Design target: the inverter "needs to be designed to sustain 60000 and more operating hours during life".
  - Product: SKAI2HV, an inverter for commercial vehicles with **sintered** power semiconductor dice. Continuous motor current up to 350 A rms; battery voltage up to 800 V DC. (The article does not say "IGBT"; the maker's product page, S8, does.)
  - Field use: since 2013, "tens of thousands" of commercial vehicles, mainly city buses, use SKAI2HV. They accumulated over 800 million operating hours. With a "typical city-bus driving profile of 10 km/hour", this is about 8 billion km. Semikron equipped over 53,000 city buses in total.
  - Power range of the vehicles shown with SKAI2HV at the IAA show: 80 to 300 kW.
  - The article gives **no** failure count, failure rate or lifetime result for the field fleet.
- Use here: a maker's design target for bus inverters (60,000 operating hours), and the conversion 10 km per operating hour that the maker assumes for a city bus. Evidence of one maker's practice, not an industry standard.

### S3. Lu, Y., Xiang, E., Zhu, L., Gao, H., Yang, H., Zhao, R., "Mission Profile-Based Lifetime Estimation and Its System-Controlled Improvement Method of IGBT Modules for Electric Vehicle Converters", *CPSS Transactions on Power Electronics and Applications* 8(3), 246-256, Sep 2023, DOI 10.24295/CPSSTPEA.2023.00023

- URL: https://file.cpss.org.cn/uploads/allimg/230928/10.24295CPSSTPEA.2023.00023.pdf (open access, 11 pages).
- Evidence basis: read in full (text layer; pages 6 and 7 rendered for Eq. (13), Eq. (14), Fig. 16 and Tables V-VII).
- Case: a passenger electric car, mass 1.9 t, transmission ratio 12.9 (Table I, p. 247); motor 115 kW (Table II, p. 249); DC bus 350 V; heatsink temperature fixed at 60 °C (p. 251). Module: Infineon FS820R08A6P2B, a HybridPACK 2 IGBT module, 750 V / 820 A, with 66 bond wires per phase part (p. 249; Table III, p. 250).
- Cycles: CLTC (China light-duty vehicle test cycle) and WLTP (worldwide harmonised light vehicle test procedure), each about 1800 s long (time axes of Figs. 2, 15 and 19).
- Chain: vehicle dynamics -> motor current -> losses -> Foster thermal network (4th order) -> junction temperature -> rainflow count -> lifetime model -> Miner sum (Fig. 18, p. 253).
- Lifetime model (Eq. (13), p. 251, coefficients Table V, p. 252; attributed to Infineon, Thoben 2016, their ref. [6]):
  - N_f = a1 (ΔT_j / 1 K)^(-b1) exp(E_a1 / (k_b T_j,max)) f(t_on) if ΔT_j <= 45 K; a1 = 5.4e14, b1 = 7, E_a1 = 0.215 eV.
  - N_f = a2 (ΔT_j / 1 K)^(-b2) exp(E_a2 / (k_b T_j,max)) f(t_on) if ΔT_j > 45 K; a2 = 2e9, b2 = 3.7, E_a2 = 0.215 eV. k_b = 86 µeV/K.
  - f(t_on) = 2.25 if t_on < 0.1 s; (t_on / 1.5 s)^x if 0.1 s <= t_on < 60 s; 0.33 if t_on >= 60 s (Eq. (14), p. 252). The printed equation has **no exponent x**. Continuity at both ends gives x = -0.30 (my computation: (0.1/1.5)^-0.3 = 2.25 and (60/1.5)^-0.3 = 0.33). Table V also lists "c = 4.5", which no equation uses.
  - The model counts damage of the bond wires, the metallization and the die-attach solder (p. 251).
- Rainflow result: in both cycles almost all cycles have ΔT_j below about 40 K and T_jm between 65 and 80 °C; a few tens of cycles per class (Fig. 16, p. 251).
- Damage and life:
  - Table VI (p. 252), damage "under specific MPs (equivalent driving distance is around 10000 kilometers)" (p. 252): IGBT 0.59 % (CLTC), 1.23 % (WLTP); diode 2.11 % (CLTC), 2.59 % (WLTP).
  - Table VII (p. 252), life of the module (shortest-lived chip), by DC-bus voltage: CLTC 300 / 350 / 400 V: 9.45e4 / 7.53e4 / 5.68e4 h, 2.7e6 / 2.17e6 / 1.41e6 km. WLTP: 3.67e4 / 2.31e4 / 1.58e4 h, 9.82e5 / 6.79e5 / 4.46e5 km.
  - Life formula: T_life = T_MP / (3600 D_MP), L_life = L_MP / (1000 D_MP) (Eq. (16), p. 252).
  - **Inconsistency in the paper**: Table VI gives 2.11 % per about 10,000 km for the diode under CLTC, which means a life of about 0.47e6 km. Table VII gives 2.17e6 km. The two tables do not agree; the paper does not explain the gap.
- Derived damage per km (my computation from Table VII at 350 V): CLTC 1 / 2.17e6 km = **4.6e-7 per km**; WLTP 1 / 6.79e5 km = **1.5e-6 per km**. From Table VI (diode, CLTC): 2.1e-6 per km.
- Sensitivities: DC-bus voltage 300 -> 400 V shortens life by a factor 1.9 (CLTC) and 2.3 (WLTP) (Table VII). A transmission ratio of 15.9 or 16.4 instead of 12.9 lengthens life by a factor 2.0 to 2.5 (Table VIII, p. 254). In a WLTP segment, ΔT_j of a cycle drops from 17.6 to 12.4 K with the higher ratio (p. 254, Fig. 19).
- Limits: a passenger car, not a bus. Heatsink at a fixed 60 °C, so the effect of ambient temperature is not modelled. No scatter of the result is given.

### S4. Bayerer, R., Herrmann, T., Licht, T., Lutz, J., Feller, M., "Model for Power Cycling lifetime of IGBT Modules - various factors influencing lifetime", CIPS 2008 (the CIPS08 model)

- File: `literature/Model_for_Power_Cycling_lifetime_of_IGBT_Modules_-_various_factors_influencing_lifetime.pdf` (6 pages). The scoping section already gives the model and both exponent sets; it does not give the range of the test data or the scatter, so the paper was read again for these.
- Evidence basis: read in full (text layer; page 5 rendered for Figs. 8-11).
- Findings not in the scoping section:
  - **Range of the test data**: ΔT_j from 45 K to 150 K; maximum junction temperature T_j from 80 °C to 205 °C (Sec. 2.2, p. 1-2).
  - Above ΔT_j of about 100 K (T_j,max = 125 °C) or 125 K (T_j,max = 150 °C) the curves of Fig. 11 are extrapolated in T_j,max (Sec. 3.5, p. 5). Above ΔT_j of about 100 K, fatigue of the substrate-to-baseplate solder limits life, and the model does not cover it (Sec. 3.5, p. 5-6).
  - At low current per bond foot and low blocking voltage the model gives "extremely high N_f"; chip-solder fatigue then limits life and the model "becomes inaccurate" (p. 6).
  - **Scatter**: lifetime data "follow a Weibull distribution, normally"; the residuals of the log-linear fit were checked and found normal (Sec. 3.3, p. 3-4). The paper gives **no numerical** spread: the N_f axis of Fig. 9 (test against model, p. 5) has no readable decade labels, and no standard deviation of the residuals is printed. The agreement is "accepted" given the "natural spread of degradation processes and test accuracies" (Sec. 3.5, p. 5).
  - The authors "strongly recommend not applying the model without consulting experts at Infineon Technologies" (Sec. 4, p. 6).
- Use here: the calibration range of CIPS08 (ΔT_j >= 45 K) lies **above** the swings of a city drive cycle (S3, Fig. 16: mostly below 40 K). Applying CIPS08 to bus cycles is an extrapolation.

### S5. Wintrich, A. (Semikron Danfoss), *Power Cycle Model for IGBT Product Lines*, Application Note AN 21-001, revision 02, 23 Jul 2024

- URL: https://assets.danfoss.com/documents/latest/444233/AB501650017495en-000201.pdf (10 pages, public).
- Evidence basis: read in full (text layer; pages 3, 4 and 5 rendered for the equation and Figs. 2-3).
- Failure modes covered: chip-solder fatigue, bond-wire heel crack or lift-off, and top-side metallization; **not** baseplate-solder fatigue from temperature cycling (Sec. 1, p. 1).
- **End-of-life (EOL) criterion**: a 20 % rise of the forward voltage (V_CE or V_DS), or of R_th, or of ΔT_j, each against its initial value (Sec. 1.2, p. 2).
- **Statistics**: Weibull statistics turn test results into a failure probability. **The power-cycle (PC) curves of the note are at 15 % failure probability** (Sec. 1.2, p. 2). The note gives no Weibull shape parameter.
- Data base: about 1000 devices; over 60 % tested to EOL; 600 to 1700 V; IGBT, diodes and SiC; t_on from 0.07 to 60 s (DC test) and 0.04 to 0.5 s (inverter test); **ΔT_j from 50 to 120 K (DC test) and 30 to 63 K (inverter test)**; T_jm from 32 to 127 °C (Sec. 1.4, p. 2).
- Model (Sec. 2.1, p. 3), T_jm = (T_j,max + T_j,min)/2 in kelvin (Sec. 2.3):
  - N_f = A0 · A1^β · ΔT_j^(-β) · ΔT_j^α · exp(E_a / (k_B T_jm)) · (C + t_on^γ) / (C + 2^γ) · k_thickness, with β = exp(-(ΔT_j - T0) / λ).
  - The factor A1^β ΔT_j^(-β) is the "low ΔT_j extension"; it is about 1 for ΔT_j above 60 K and grows below 40 K.
- Parameters (Tables 1-3, pp. 6-8):

  | Module family | A0 | A1 | T0 [K] | λ [K] | α | E_a [J] | C | γ |
  |---|---|---|---|---|---|---|---|---|
  | Copper baseplate, soldered chips (SEMITRANS, SEMiX) | 2.9e9 | 60 | 40 | 17 | -4.3 | 4.50e-20 | 1 | -0.75 |
  | No baseplate, soldered chips (MiniSKiiP, SEMITOP, SKiiP3, SKiM4/5) | 2.90e9 | 60 | 40 | 17 | -4.3 | 4.50e-20 | 0.38 | -0.7 |
  | No baseplate, single-side sintered chips (SKiM63/93, SKiiP4) | 2.05e11 | 60 | 38 | 17 | -4.3 | 2.54e-20 | 1.44 | -1.21 |

  k_B = 1.38e-23 J/K. Chip-thickness factor k_thickness: 1 for IGBT up to 1200 V; 0.65 for 1700 V IGBT and CAL diodes; 0.5 for rectifiers; **0.33 for SiC devices up to 1200 V** (Sec. 2.5, p. 5).
- Check of the formula (my computation): ΔT_j = 60 K, T_j,min = 40 °C, t_on = 2 s, baseplate set: N_f = 8.8e5. The note states "about 880k cycles" for this case (Sec. 2.3, p. 4).
- Effect of T_jm: at ΔT_j = 60 K, N_f is about 880k for 40 -> 100 °C but about 260k for 90 -> 150 °C (Sec. 2.3, p. 4, Fig. 2).
- **Low-ΔT_j evidence and scatter**: inverter tests over 1.5 years at low ΔT_j (Sec. 2.4, p. 4). In Fig. 3 (p. 5), the measured ratio N_f / N_f* (N_f* = model without the low-ΔT_j factor) is about 4 to 13 at ΔT_j ≈ 30 K, about 1.4 to 3.6 at 50-52 K, and about 1.1 to 1.4 at 63 K (my reading of the plotted points). So at 30 K the test results spread over a factor of about 3.
- The dependence on chip thickness between 650 and 1200 V is "much smaller than the standard deviation" of the results under one test condition (Sec. 2.5, p. 5); the standard deviation itself is not given.
- Use here: a maker's public lifetime model with numbers, a stated EOL criterion and a stated failure probability, calibrated down to ΔT_j = 30 K by inverter tests. The SKAI2HV bus inverter of S2 has sintered dice, so the third parameter set is the closest; the note does not list SKAI2HV among its product families, so this pairing is my assumption.

### S6. Choi, U.-M., Blaabjerg, F., Jørgensen, S., "Study on Effect of Junction Temperature Swing Duration on Lifetime of Transfer Molded Power IGBT Modules", *IEEE Transactions on Power Electronics* 32(8), 6434-6443, 2017, DOI 10.1109/TPEL.2016.2618917

- URL: https://vbn.aau.dk/ws/files/280878799/07820215.pdf (publisher's version in the Aalborg University repository; 10 pages plus a cover page).
- Evidence basis: read in full (text layer; the Weibull plots of Fig. 7 were not rendered, the numbers come from Table IV).
- Device and test: a 600 V, 30 A three-phase transfer-moulded intelligent power module (IPM), no baseplate, aluminium bond wires (Sec. III-A, p. 6437). Power cycling in a real inverter (two back-to-back converters). Six conditions with ΔT_j ≈ 81 K and T_jm ≈ 102 °C, and swing durations t_ΔTj from 0.59 to 10 s (Table II, p. 6437). 6 to 8 modules per condition, 39 in total (p. 6437).
- EOL criterion: +5 % of V_CE,on (p. 6437-6438). "Typically, 5 %-20 % increase of V_CE,on is considered as the end-of-life criterion" (Sec. IV, p. 6438).
- Scatter within one condition: "the number of cycles to failure is different to each other from about 20000 to 45000 cycles" (p. 6437).
- Table IV (p. 6440), nominal B1 / B10 / B63.2 / MTTF in cycles, for example t_ΔTj = 10 s: 107,413 / 127,800 / 150,930 / 145,264; t_ΔTj = 1 s: 165,927 / 183,120 / 201,252 / 196,717.
- **Weibull shape derived from Table IV** (my computation; a two-parameter Weibull fits B1, B10, B63.2 and MTTF of each row to within 10 cycles): shape β = 13.5 (10 s), 17.2 (5 s), 24.1 (2 s), 23.8 (1 s), 24.0 (0.8 s), 16.6 (0.59 s). Coefficient of variation (CV) of the life: 0.05 to 0.09. B10 / B50 = 0.87 to 0.93.
- Failure mode: bond-wire cracks in all tested modules; no visible solder degradation (Sec. V, p. 6441).
- Limits: one small industrial module type, one ΔT_j (81 K), small samples from one production lot are likely (the paper does not say). The authors ask for tests at smaller ΔT_j and longer durations (Sec. VI, p. 6442). So this scatter is a **lower bound** on the scatter of a traction module in the field.

### S7. ZF Friedrichshafen AG, *AxTrax 2 LF Low-Floor Electric Axle* (fact sheet, bus, English), 03/2026

- URL: https://www.zf.com/public/org/AxTrax2_LF_FactSheet_EN_Bus_032026_340341.pdf (8 pages).
- Evidence basis: read in full.
- Findings:
  - The axle has **two stand-alone silicon carbide (SiC) inverters** and an electronic control unit (ECU); it is for electric, hybrid and trolley buses up to 30 t (p. 2). The inverters are "800 Volt" SiC inverters (pp. 5, 7).
  - SiC inverter: HV range 450-770 V or 500-850 V; maximum HV DC current up to 520 A; ZF control board and in-house software (p. 6).
  - Two permanent-magnet synchronous (PSM) hairpin motors, up to 360 kW continuous; peak system output torque up to 36,800 Nm (p. 7).
  - Variants (p. 7): AX436, 360 kW continuous, maximum vehicle weight 30 t; AX426, 260 kW continuous, 20 t. Integrated axle ratio 21.3 or 18.7. System weight about 1,257 kg including both inverters.
  - Maintenance: "enhanced axle condition monitoring" and over-the-air updates (p. 5); ZF Data Solutions health data (p. 7).
  - The fact sheet names **no** semiconductor maker, module type, chip-attach technology or lifetime.
- Use here: with S1, the eCitaro sold in 2026 has two 800 V SiC inverters (one per motor is likely, but the fact sheet does not say so). The public data stop at the inverter level.

### S8. Semikron Danfoss, product page *SKAI High Voltage Converters*

- URL: https://www.semikron-danfoss.com/products/product-lines/skai/skai-2-hv.html (downloaded 9 Nov 2026; the page carries no date).
- Evidence basis: read in full (the visible text of the page).
- Findings:
  - SKAI 2 HV (2nd generation) is **IGBT-based**, with **sintered and AlCu-bonded** power semiconductors, a polypropylene film DC-link capacitor, liquid cooling, IP67 housing; up to 800 V DC and 300 kVA.
  - SKAI 3 HV (3rd generation) uses **SiC MOSFETs**; up to 950 V DC and 600 kVA.
  - Applications named: "Electric trucks, electric busses".
- Use here: confirms that the SKAI2HV of S2 is an IGBT inverter with sintered chips. The bond wires are AlCu, while the model of S5 is fitted to Al bond wires; AlCu bonds are meant to last longer, so S5 applied to SKAI2HV is likely conservative (my inference; no source read on AlCu power cycling).

## 4. Gaps

| Missing value | Why it matters | How to fill it |
|---|---|---|
| ΔT_j per stop of a bus inverter (and per braking) | Sets the absolute damage level; a 10 % error gives a factor 0.5-1.8 | Ask Semikron Danfoss or ZF for a bus mission-profile analysis (a search snippet, not read, says Semikron offers lifetime estimation from application profiles as a service). Or simulate with the maker's loss tool (SemiSel) and a bus drive cycle. Or log the inverter's temperature sensor on a bus (ZF Data Solutions, S7) |
| A published bus mission-profile lifetime study | Would replace the calibration of Section 2 | None found in the searches listed below; Semikron's article "Pimp your electric drive" shows bus temperature cycles (search snippet) but could not be read |
| Empty and mean operating mass of the eCitaro and eCitaro G | Sets the mass ratio r; the damage is very sensitive to it | Daimler Buses type-approval data or operator data (VBG, Eurobus); also listed as a gap in `zurich_depot_and_lines.md` |
| Coolant temperature against ambient temperature | Sets F_T per season | Maker data or bus logging |
| Weibull shape of power-cycling life of traction modules at low ΔT_j | Sets τ (0.57 at k = 5; 0.81 at k = 13.5) | Maker qualification data (AQG 324 reports), or published test series at ΔT_j ≤ 40 K |
| Variance of the damage increment under the same use | Needed for Cantelli or Bernstein | Simulate the S5 model over measured load and temperature records of the lines |
| Replacement cost of a traction inverter or module | Cost C_rep | Quote from ZF or Daimler Buses service; teardown studies (ICCT/Ricardo eTruck teardown, not read) |
| Downtime of an inverter replacement | Availability | Operator workshop records (Eurobus) |
| Which drive the VBG buses of 2023 and 2025 have (AVE 130 or AxTrax 2 LF) | The 2023 line-759 buses may have the older ZF axle | Daimler Truck press releases (search snippets only) or VBG |
| Field failure rate of bus inverters by wear-out | Validation of the low damage rate | Fleet reports name inverter defects only in small shares (`maintenance_focus_reports.md`, R3); no wear-out data |

## 5. Could not read

| Item | Reason | Why it looks worthwhile |
|---|---|---|
| Amirpour, S., Thiringer, T., Hagstedt, D., "Mission-Profile-Based Lifetime Study for SiC/IGBT Modules in a Propulsion Inverter", PEMC 2021, pp. 264-271, DOI 10.1109/PEMC48073.2021.9432583 | IEEE Xplore answered HTTP 418 (bot block) to the PDF route; the document page in the browser pane did not load (pane hidden). Chalmers lists no full text | Compares SiC and Si IGBT damage over ECE-City (urban, many stops) and WLTC; would give the SiC factor and the stop effect from one study. Can you retrieve it? |
| Kundu, A. et al., "An Approach for Estimating the Reliability of IGBT Power Modules in Electrified Vehicle Traction Inverters", *Vehicles* 2(3), 2020 (MDPI, open access) | MDPI answered "Access Denied" to the script download | EV traction inverter, two drive cycles, reliability estimate. Can you retrieve it? |
| Semikron, technical article "Pimp your electric drive" | The page redirects to a Semikron Danfoss landing page; no Wayback copy found | Per a search snippet, it shows the temperature cycles of a reinforced SKAI2HV in a large city bus: the missing ΔT_j per stop |
| PDF version of S2 (516.79 KB, linked from the archived page) | Not opened (the web text was read in full) | May hold the same text with figures |
| Journal of Semiconductors 45(5) 052301, 2024, "Reliability evaluation of IGBT power module on electric vehicle using big data" | The landing page gives no PDF link | Big-data junction-temperature estimation for EVs |

## 6. Leads not followed

- Yesgat, A., *Active Thermal Control of Power Semiconductor for High Power Electric Drive Applications*, MSc thesis, Chalmers, 2024 (https://odr.chalmers.se/bitstreams/48cb3138-fa65-4ef2-95ad-5b564a28e953/download, 82 pages). Heavy-duty truck, SiC, HHDDT cycle, CIPS08 and Infineon SiC models. Only the abstract and contents were seen; not read, so no number is used. Lower priority: a master's thesis, a truck cycle.
- Thoben, M., Infineon tutorial "Reliability and lifetime of power modules" (2016, ResearchGate): source of the HybridPACK 2 model of S3.
- Infineon application note AN2019-05 "PC and TC diagrams": Infineon's power-cycling curves and their failure-probability definition.
- ECPE guideline AQG 324 and its annex on lifetime calculation: the qualification standard for automotive power modules.
- Lutz, J. et al., "Validity of power cycling lifetime models for modules and extension to low temperature swings", EPE 2020 (S5 ref. [10]): basis of the low-ΔT_j extension.
- Scheuermann, U., Schmidt, R., PCIM 2013, SKiM63 lifetime model for sintered chips (S5 ref. [6]).
- Vernica, I., Wang, H., Blaabjerg, F., "Uncertainties in the lifetime prediction of IGBTs for a motor drive application", PEAC 2018 (S3 ref. [21]): Monte Carlo scatter of predicted life.
- ICCT/Ricardo, *eTruck Virtual Teardown* (2022): possible inverter cost.
- Daimler Truck press releases on the eCitaro with the ZF AVE 130 axle (for the buses delivered before the AxTrax 2 LF).
- Searches run without a bus-specific mission-profile result (9 Nov 2026): "city bus cycle IGBT junction temperature lifetime rainflow"; "electric bus IGBT lifetime Miner driving cycle"; "Braunschweig cycle IGBT power module"; "Manhattan / Orange County / CBD bus cycle junction temperature"; "China typical city bus / CCBC / C-WTVC IGBT"; "pure electric bus IGBT junction temperature"; "hybrid bus / transit bus IGBT thermal cycling"; "trolleybus / tram traction converter IGBT lifetime"; "electric bus SiC inverter lifetime drive cycle".
