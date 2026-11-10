# Component: tyres of a battery-electric city bus

Status: done (10 Nov 2026). Ten sources read (S1-S10), in full or in the parts stated. In progress: none.
On 10 Nov 2026 Fwa 1991 was tried again and could not be read (section 5): the ETH licence seems not
to cover the article. So the route factor stays the derived F2 of section 2.3, and the per-line values were not
recomputed; the Hagenholz lines (case_study/zurich_hagenholz_line_features.csv) have no g_j yet.
Next: fill the gaps of section 4; first G1 (battery-bus factor from tread data) and G7 (prices in CHF);
Fwa 1991 through the user (section 5).
Scope: the vehicle is free (the user's choice): any well-documented 12 m battery bus; the eCitaro is used
here. The lines are those near Zurich (case_study/zurich_line_features.csv).

## 1. Parameter table

**Vehicle.** The reference bus is the Mercedes-Benz eCitaro, 12 m, two axles (S6): tyres 275/70 R22.5,
gross weight 20.0 t, front axle at most 7.5 t, drive axle at most 13.0 t. The user allows any
well-documented 12 m battery bus; the eCitaro is the one with a full public data sheet read here, and
it runs on the case-study lines. The data sheet gives no unladen mass (gap G6).

**Damage variable.** D = tread depth lost in the main grooves, in mm, measured at the worst groove (as in
S1 §2). Two components per bus: the steer-axle tyre set (2 tyres) and the drive-axle tyre set (4 tyres;
the S1 buses have twin rear tyres; S6 does not say).

"Derived here" means computed in this file from cited numbers; it is not a statement of the source.

| Parameter | Value | Unit | Source and location | Evidence basis | Note |
|---|---|---|---|---|---|
| New tread depth h0, steer axle | 19.5 | mm | S8 p. 38 (Goodyear UrbanMax MCA 275/70 R22.5) | read in part | city-bus tyre, one maker; regional tyres 13.8-18.1 mm (S8 p. 38); S1 example 17 mm (§2) |
| New tread depth h0, drive axle | 20.4 | mm | S8 p. 38 (UrbanMax MCD Traction 275/70 R22.5) | read in part | |
| Legal minimum depth | 1.6 | mm | S4, VTS Art. 58 Abs. 4 | read in part (the article in full) | all motor vehicles, buses included; no separate winter rule in the VTS |
| Operator replacement depth | about 2 | mm | S1 §2 p. 4 | read in full | one Italian operator; 3-4 mm for winter tyres in some countries (S1 §1, S7 §1) |
| tau, steer (to 2 mm) | 17.5 | mm | derived here: 19.5 - 2.0 | - | to 1.6 mm: 17.9 mm; to 3 mm: 16.5 mm |
| tau, drive (to 2 mm) | 18.4 | mm | derived here: 20.4 - 2.0 | - | to 1.6 mm: 18.8 mm; to 3 mm: 17.4 mm |
| Recommended family | inverse Gaussian (alternative: Gamma) | - | section 2 | - | no source fits a process to tyre data (section 2) |
| Base wear rate, steer, diesel bus | 0.15 | mm per 1,000 km | S1 Table 6 (275/70 R22.5, new, front) | read in full | regional routes, 22-38 km/h; all tyres of this size 0.17; range of tyre sets 0.13-0.22 (S1 §3.2) |
| Base wear rate, drive, diesel bus | 0.16 | mm per 1,000 km | S1 Table 6 (275/70 R22.5, new, rear) | read in full | retreads on the rear: 0.19 (other subset) |
| Battery-bus factor f_BEB | 1.35 (range 1.3-1.8) | - | R8 pp. 58, 83 (tyre cost +30 % to +38.8 %); R1 pp. 38, 63 (miles per tyre 43-45 % lower) | from case_study/maintenance_focus_reports.md, not re-read | R8 is a cost ratio, R1 is confounded by route (potholes); see section 2 |
| Mean increment per 1,000 km, battery bus | steer 0.20, drive 0.22 | mm per 1,000 km | derived here: base × 1.35 | - | life to tau about 85,000-86,000 km |
| Route factor g_j | 0.92-1.35 for the case-study lines | - | derived here (section 2, formula F2) | - | optional; default g_j = 1 |
| Load factor (EMEP) | LCF = 1.41 + 1.38 × LF | - | S10 Eqs. B3-B4 (printed p. 82) | read in part | full load wears 1.98 × empty; applies "for urban buses and coaches" |
| Speed factor (EMEP) | 1.39, flat below 40 km/h | - | S10 Figure B1 (printed p. 83) | read in part | all case-study lines run at 10-32 km/h, so no line differs by speed |
| CV of the wear rate between tyres, c | 0.30 (low 0.19) | - | S1 §3.2 pp. 5-6, §4 pp. 10-11 | read in full | 0.19 between tyre sets, about 0.30 within sets and per tyre; cars 0.13-0.25 (S7 §3.2) |
| Gamma rate beta | steer 0.635, drive 0.604 (c = 0.19: 1.58, 1.50) | 1/mm | derived here: beta = 1 / (c² tau) | - | matches the spread c at the end of life |
| IG squared CV eta | 0.09 (c = 0.19: 0.036) | - | derived here: eta = c² | - | the IG row of the library needs no independence |
| Gaussian or rainflow (Cantelli) variance v | c² × tau × mu_inc | mm² per period | derived here | - | for example line 759, drive, one weekday: 0.087 mm² |
| Wiener sigma | sqrt(c² × tau × mean mu_inc per period) | mm | derived here | - | for example 0.29 mm per weekday at 0.052 mm per day |
| Regrooving: depth restored | at most 3 (Goodyear); 3-4 (Michelin: from 3-4 mm left to 6-8 mm) | mm | S8 pp. 44-45, 53; S5 slide 4 | S8 read in part; S5 read in full | regroove when 3-6 mm (S8) or 3-4 mm (S5) remain |
| rho of regrooving (ARD-infinity) | 0.2 (range 0.17-0.25) | - | derived here: rho = restored / (h0 - remaining depth) | - | rear axle in practice (S8 p. 44); normally once per tread life (S5, S8) |
| Extra life from regrooving | up to 25 % | - | S5 slides 2, 4 | read in full | manufacturer claim |
| Replacement value mu_new | 0 (new tyre or retread) | mm | definition of D | - | retread tread depth not found (gap G3) |
| Initial damage mu_0 | spread over [0, tau); mean tau/2 | mm | derived here (renewal argument, section 2) | - | no source for buses (gap G5) |
| epsilon | no source | - | - | - | modelling choice; per tyre, epsilon_set / (number of tyres) by the union bound (section 2) |
| Price of a new tyre | no source read; about 414 USD derived from R1 | USD per tyre | derived here: 0.07 USD/mile × 5,920 miles per tyre (R1 p. 38) | from maintenance_focus_reports.md | definitions in R1 not checked; US, about 2019-2020 |
| Price of a retread | about 50 % of a new tyre | - | S5 slide 7 (Michelin Remix) | read in full | 2011 sales guide |
| Price of a regroove | about 12 % of a new tyre (about 25 GBP in 2011) | - | S5 slide 7 | read in full | |
| C_R (repair cost per mm removed) | about 0.04 × price of a new tyre, per tyre | per mm | derived here: 0.12 × price / 3 mm | - | multiply by 2 (steer) or 4 (drive) per set |
| C_rep (replacement) | 1 × price (new), 0.5 × price (retread, rear only) | per tyre | S5 slide 7; S1 §2 p. 4 (retreads rear only) | read in full | plus fitting labour (gap G7) |
| Tyre cost per bus-km, battery bus | 125 (diesel 90) | CNY per 1,000 km | R8 p. 83 | from maintenance_focus_reports.md | Shenzhen |
| Labour and depot time | no source | - | - | - | gap G7 |

## 2. Recommended family and the formula from line features to the increment per period

### 2.1 What the sources show about the family

- **No source read fits a stochastic process to tyre tread data.** Two leads could not be read: a 2001
  paper with "tire wear" among its keywords (Bagdonavicius and Nikulin, paywall) and a 2022 fit of a
  Gamma *distribution* to the tread depths of 4,288 car tyres (Promet; host blocked). See section 5.
- **Established in general.** Wiener, Gamma and inverse Gaussian (IG) processes are "the three most
  commonly used processes for modeling degradation paths" (S9 §3.2 p. 12). This is a general statement,
  not one about tyres.
- **Shape of a tyre's path.** Almost linear in distance for three bus tyres on one route (S1 §3.1,
  Figure 2). Depth may fall faster during a break-in, and depth and mass do not always fall linearly
  (S7 §2.1, §3.4, §4).
- **Spread.** The main uncertainty is between tyres: coefficient of variation (CV) of the wear rate
  0.19 between tyre sets and about 0.30 per tyre (S1 §3.2, §4); route effect 20-30 % (S1 §4). For car
  tyres the CV of life is 0.13-0.25 (S7 §3.2).
- **Monotone damage.** Rubber is only lost. Gamma and IG increments are positive; Gaussian and Wiener
  increments can be negative, which is harmless only when the CV per period is small.

### 2.2 Recommendation

1. **Inverse Gaussian, eta = c² = 0.09 (c = 0.30; or 0.036 with c = 0.19).** Reasons:
   - the increments are positive;
   - the IG reliability row of the library holds for any dependence between the terms (spec section
     "Inverse Gaussian Model"). It therefore covers a wear rate that is random per tyre but fixed over
     its life ("a fast tyre stays fast"), which is the structure that S1 shows;
   - one parameter per cell, read directly from the CV of the wear rate.
2. **Alternative: Gamma, beta = 1 / (c² tau)** (0.604-0.635 per mm for c = 0.30). It is the most common
   process for monotone wear in general (S9) and the library has validators for it. Its independent
   increments make the spread grow with the square root of the mean damage. Calibrated in this way it
   matches the spread c at the threshold, where the reliability row binds, and overstates the spread
   early in life.
3. **Not recommended.** Wiener: one sigma per cell for all lines, and negative increments. Gaussian:
   negative increments. Rainflow: built for counted load histories; the tyre data are depth readings.

**Effect on the usable tread (derived here with scipy, tau = 18.4 mm).** Largest mean damage allowed by
Pr(D > tau) <= epsilon, as a fraction of tau. IG: tau / E[Z | Z > q] with Z ~ IG(1, 1/eta), the library's
row; Gamma: exact tail of Gamma(beta × mu, beta); Gaussian: normal tail with variance c² × tau × mu:

| c | epsilon | IG row of the library | Gamma (exact tail) | Gaussian |
|---|---|---|---|---|
| 0.19 | 0.01 | 0.62 | 0.60 | 0.65 |
| 0.19 | 0.05 | 0.69 | 0.72 | 0.73 |
| 0.30 | 0.01 | 0.48 | 0.42 | 0.50 |
| 0.30 | 0.05 | 0.57 | 0.58 | 0.61 |

So an open-loop plan uses only 42-73 % of the tread before replacement. The cause is the spread
between tyres, not the family. Operators measure tread depth at maintenance (S1 §2: once a year in
Varese). With a measured mu_0 per tyre and frequent re-planning, a smaller c (the noise of one tyre's
own path) would apply; no source read quantifies it (gap G4).

### 2.3 Formula from line features to the increment per period

**F1, default (every coefficient cited).** For bus i, axle set l, line j, period k:

    mu_inc(i, l, j, k) = r_l × f_BEB × L_jk × g_j        [mm per period]

- r_steer = 0.15 mm per 1,000 km and r_drive = 0.16 mm per 1,000 km: S1 Table 6 (275/70 R22.5, new
  tyres, diesel buses, regional routes).
- f_BEB = 1.35 (range 1.3-1.8): R8 pp. 58, 83 and R1 pp. 38, 63 (as recorded in
  case_study/maintenance_focus_reports.md).
- L_jk = km of one bus on line j in period k = km per vehicle per day (line table, columns
  `km_per_vehicle_wd`, `_sa`, `_su`) × number of such days in the period.
- g_j = 1. Reason: the EMEP speed factor is flat below 40 km/h (S10 Figure B1), and all case-study lines
  run at 10-32 km/h, so no cited relation separates them.
- Variance or rate per period: Gamma beta as in section 1 (independent of the period); Gaussian or
  Cantelli v = c² × tau × mu_inc; IG eta = c².

**F2, optional route factor (derived here; structure cited, two inputs assumed).**

    g_j = (1 - phi) + phi × X_j / Xbar,   X_j = s_j × a × v_j² + g² × theta_j × (A_j + D_j) / L_j

- Structure: the slip-energy model of HDM-4 (S10 Eq. C1) makes wear per km proportional to F² / N; with
  F = m × acceleration this is m × acceleration² / g. Integrated along the line: each stop has one
  acceleration and one braking at a over v² / (2a), which gives a × v² per stop; a grade theta adds
  (g theta)² over the climb and the descent, which gives g² × theta × (A + D). Unit of X_j: m²/s⁴ (s_j per
  metre, L_j in metres).
- phi = 0.3: share of wear caused by manoeuvres on the reference routes, derived here from S1 Table 5:
  routes with less stop-and-go wore 18-33 % less, so if those routes had no manoeuvre wear at all, the
  share is at least 0.18-0.33.
- Inputs from the line table: s_j stops per km, v_j mean speed (a proxy for the cruise speed),
  A_j + D_j ascent plus descent per round trip, L_j round-trip length.
- Assumed (no source): a = 1.0 m/s² at stops; theta_j = half the steepest grade of the line.
- Xbar = mean of X_j weighted by vehicle-km, so the fleet mean of g_j is 1 and the fleet mean rate
  stays the cited one.
- Mass: by the same structure, wear per km is proportional to the mass per wheel at fixed manoeuvres
  (S10, derived). For another bus of mass m', multiply by m' / m.
- Load: the EMEP factor LCF = 1.41 + 1.38 × LF (S10 Eq. B4) can scale a line by its mean load factor
  LF_j: multiply by LCF(LF_j) / LCF(mean LF). The line table has no occupancy (gap G2).

**Values for the case-study lines (derived here; weekday, one bus, battery-bus factor 1.35).**

| Line | Stops per km | Mean speed km/h | (A + D) / L m per km | g_j (F2) | km per bus per weekday | mu steer, mm per day (F1) | mu drive, mm per day (F1) | mu drive with g_j |
|---|---|---|---|---|---|---|---|---|
| 739 | 1.90 | 25.3 | 9.0 | 0.97 | 113.9 | 0.023 | 0.025 | 0.024 |
| 742 | 1.98 | 27.3 | 19.7 | 1.11 | 177.3 | 0.036 | 0.038 | 0.042 |
| 749 | 1.85 | 27.8 | 19.2 | 1.20 | 63.2 | 0.013 | 0.014 | 0.016 |
| 759 | 1.99 | 21.8 | 10.7 | 0.96 | 241.9 | 0.049 | 0.052 | 0.050 |
| 766 | 2.28 | 25.1 | 10.0 | 1.02 | 59.0 | 0.012 | 0.013 | 0.013 |
| 768 | 2.31 | 21.1 | 16.5 | 1.01 | 229.2 | 0.046 | 0.050 | 0.050 |
| 769 | 1.31 | 30.0 | 30.9 | 1.32 | 39.7 | 0.008 | 0.009 | 0.011 |
| 771 | 3.88 | 20.6 | 23.5 | 1.17 | 90.8 | 0.018 | 0.020 | 0.023 |
| 772 | 4.08 | 17.6 | 16.3 | 1.03 | 90.3 | 0.018 | 0.020 | 0.020 |
| 773 | 2.84 | 21.1 | 28.0 | 1.22 | 28.2 | 0.006 | 0.006 | 0.007 |
| 774 | 3.75 | 16.0 | 38.5 | 1.35 | 33.0 | 0.007 | 0.007 | 0.010 |
| 781 | 2.94 | 19.6 | 9.5 | 0.98 | 165.0 | 0.033 | 0.036 | 0.035 |
| 787 | 1.97 | 20.0 | 10.5 | 0.92 | 255.6 | 0.052 | 0.055 | 0.051 |
| 795 | 2.51 | 9.9 | 23.4 | 0.93 | 26.1 | 0.005 | 0.006 | 0.005 |
| 797 | 1.14 | 31.5 | 15.6 | 1.07 | 112.1 | 0.023 | 0.024 | 0.026 |

- Line 768 runs the articulated eCitaro G in the case study; the table uses the 12 m values for all lines.
- Life check: tau / rate = 17.5 / 0.202 = 86,000 km (steer) and 18.4 / 0.216 = 85,000 km (drive). On
  line 759 (242 km per weekday) that is about one year without regrooving. S1 reports buses of this tyre
  size at 42,300 km per year (Table 6).
- The range of g_j (0.92-1.35) is within the route effect of the sources: 20-30 % (S1 §4) and a factor
  1.6 between two car routes (S7 §3.3).

### 2.4 Repair, replacement and the cells

- **Regrooving as a repair.** ARD-infinity law, rho = 0.2 (0.17-0.25; section 1). Before the first
  intervention, D_last = 0, so ARD-1 gives the same first repair. In practice regrooving happens once per
  tread life, when 3-6 mm remain, and the regrooved tyre goes to the rear axle (S8 pp. 44-45); the model
  allows any number of repairs at any depth (gap G8). The VTS allows regrooving only for tyres marked
  regroovable (S4, Art. 60 Abs. 3).
- **Replacement.** A new tyre, or on the rear axle a retread (S1: front tyres always original).
  replacement_mu = 0; replacement_as_new = true, if the retread depth equals the new depth (gap G3).
- **Cells.** One cell per axle set. The legal limit holds for every tyre; by the union bound,
  Pr(some tyre of the set beyond tau) <= n × Pr(one tyre beyond tau), so a set of n tyres with target
  epsilon needs epsilon / n per tyre (n = 2 steer, n = 4 drive). This is conservative if the tyres of a set
  wear alike.
- **Rotation.** Moving tyres between positions is out of the model.

### 2.5 Initial state

No source gives tread depths in a running bus fleet (gap G5). A derivation: if tyres are replaced at
a fixed damage tau and wear at a steady rate, the damage of a tyre picked at a random time is uniform on
[0, tau) (stationary renewal process, a classical result; no source read here). Then mean tau/2 and
variance tau²/12. For the plan, set mu_0 per bus from the operator's last readings, or spread mu_0 over
[0, tau) across the buses (for example the quantiles (i - 0.5) / F × tau), with v_0 = 0 when the depth is
measured.

## 3. Source entries

### S1. Giechaskiel, Melas, Freri 2026: tread wear of C3 bus tyres around Varese, Italy

- **Reference.** Giechaskiel, B.; Melas, A.; Freri, A. (2026). Tread Wear of C3 Bus Tyres Around a City in
  Italy. *Atmosphere* 17(8), 785. https://doi.org/10.3390/atmos17080785. Open access (CC BY). Authors from
  the Joint Research Centre (JRC) of the European Commission and the bus operator Autolinee Varesine.
  "C3" is the UNECE (United Nations Economic Commission for Europe) class of tyres for heavy vehicles.
- **Evidence basis.** Read in full (13 pages, text layer); Figures 2 and 4 read from rendered pages 5-6.
  Figures 3, 5 and 6 not read in detail (their numbers are summarised in the text and in Table 6).
- **Data.** 203 two-axle low-floor rigid buses (2 front tyres, 4 rear tyres), regional and suburban
  routes around Varese, 2019-2025; 360 tyre measurements; 6 brands, 15 tyre models; bus brands include
  Mercedes Citaro U, MAN Lion's City, Iveco Crossway (§2, Table 1; §4 p. 10). Diesel buses: the paper
  names no electric bus. Unladen mass about 10-11 t, maximum laden mass 18-19 t (§2 p. 3). Mean
  timetable speed 22-38 km/h, nominal mean 25.7 km/h; one hilly route with 450 m altitude difference
  (§2 p. 2). Annual mean temperature about 14 °C (§2 p. 2).
- **Method.** Tread depth with an analogue gauge in the three main grooves, the minimum reported, once
  per year at maintenance or inspection, tyres mounted, pressure about 8 bar (§2 p. 4). Only the total
  tread reduction at replacement is used; replacement "typically when the tread depth was reaching
  2 mm"; reductions below 5 mm discarded; measurement uncertainty assumed 0.5 mm (§2 p. 4). Example
  given: new depth 17 mm, replaced at 2 mm, so 15 mm used (§2 p. 4).
- **Findings.**
  - Wear is "almost linear" in distance for three tyre sets with intermediate data (model B3,
    size 275/70 R22.5, same route) (§3.1, Figure 2). Read from Figure 2: about 12, 14 and 16 mm lost
    at about 94,000, 82,000 and 110,000 km.
  - Tread depth reduction per 10,000 km (§3.2, Table 6):

    | Size | Subset | km per year (×1000) | mm per 10,000 km |
    |---|---|---|---|
    | 265/70 R19.5 | new, all axles | 24.3-29.4 | 2.6-2.9 |
    | 275/70 R22.5 | all | 42.3 | 1.7 |
    | 275/70 R22.5 | new, front | 44.5 | 1.5 |
    | 275/70 R22.5 | new, rear | 40.8 | 1.6 |
    | 275/70 R22.5 | retreaded once, rear | 41.5 | 1.9 |
    | 295/80 R22.5 | all | 36.3 | 1.9 |
    | 295/80 R22.5 | new, front | 38.8 | 2.2 |
    | 295/80 R22.5 | new, rear | 30.0 | 1.5 |
    | 295/80 R22.5 | retreaded once, rear | 38.1 | 1.9 |

  - **Spread between tyre sets** (§3.2, pp. 5-6): 275/70 R22.5: mean 1.7 mm per 10,000 km, "one standard
    deviation 19%", range 1.3-2.2 mm, ten tyre sets. 295/80 R22.5: mean 1.9, standard deviation 31 %,
    range 1.1-3.7, 26 sets. 265/70 R19.5, brand B: 2.6, standard deviation 30 %, range 1.8-3.4.
  - **Spread within one tyre set** (same model, same bus model): 275/70 R22.5 "17-53%, typically around
    30%" (§3.2 p. 6); 295/80 R22.5 11-89 %, mean 39 % (§3.2 p. 6). The coefficient of variation of each
    tyre's mean wear rate "was around 30%" and includes routes, drivers, buses, weather and load (§4
    pp. 10-11).
  - **Effects** from subsets (§3.3, Tables 3-5; summary §4 p. 11): route about 20-30 % (routes between
    towns with less stop-and-go wore 18-33 % less, Table 5); front against rear axle below 25 % (front
    lower or equal; e.g. B3 front -26 %, Table 3); retreaded tyres below 15 %, and lower wear than new
    ones (Table 4); tyre models on the same axle and route mostly within 15 % of the average, one case
    48 % (§4 p. 11). The hilly route (450 m) gave higher wear for 265/70 R19.5 tyres (§3.2 p. 5).
  - **Mass loss** 550-800 mg/km per bus (Table 6, §3.4), computed with density 1.15 g/cm3, contact
    width 75 %, void ratio 25 % (§2 p. 4).
  - **Legal limit** quoted as 1.6 mm for summer tyres, 3-4 mm for winter tyres in some countries (§1
    p. 2); the fleet replaced at about 2 mm (§2 p. 4).
  - **Retreading:** C3 tyres "are typically retreaded at least once" (§1 p. 1, citing a US retread
    report); in this fleet retreaded tyres only on the rear axle, retreaded once (§2 p. 4; §3.3 p. 8).
  - The authors found no other bus values in the literature (§4 p. 10). Other heavy vehicles cited
    there: below 1 mm per 10,000 km for a highway truck (their ref. 16, Lupker et al. 2002); 1-2 mm per
    10,000 km for a tractor-semitrailer in central London (their ref. 17, Lepine et al. 2022).
- **Limits.** Diesel buses on regional roads, not battery buses in a city. Speeds and loads per route
  not recorded. A one-study result, not an established rate.

### S2. Lepine, Burgess, Cebon: empirical estimation of heavy goods vehicle tyre wear (HVTT paper)

- **Reference.** Lepine, J.; Burgess, S.; Cebon, D. Empirical estimation of heavy goods vehicle tyre wear.
  Paper of the International Symposium on Heavy Vehicle Transport Technology (HVTT), University of
  Cambridge. https://hvttforum.org/wp-content/uploads/2019/11/Lepine-EMPIRICAL-ESTIMATION-OF-HEAVY-GOODS-VEHICLE-TYRE-WEAR.pdf.
  The PDF has no date; the author note says one author graduates "in 2018".
- **Evidence basis.** Read in full (11 pages, text layer). Figures read only through their axis labels.
- **Findings.**
  - Test-track measurement of rubber lost by semi-trailer tyres (385/65 R22.5) at fixed slip angles,
    100 kN per axle (§3.1, §3.3). The slip angle is the angle between the wheel plane and its direction
    of travel.
  - Wear per unit area of road: sigma(alpha) = K alpha^2, K = 145 g/m2, alpha in radians (§3.4, Eq. 1).
    At 10 degrees (0.175 rad) this gives about 4.4 g/m2.
  - Tread depth from mass: delta_h = delta_M / (pi d w rho), with d the tyre diameter, w the width,
    rho the rubber density (§4, Eqs. 2-3).
  - A 270-degree roundabout of 10-24 m diameter at 5 km/h costs a trailer tyre at most about 60
    micrometres of tread (§4.1, Figure 8 axis).
  - Only one normal load was tested; the load effect is future work (§6).
- **Use here.** Background on cornering wear only: a two-axle city bus has no unsteered axle group, so
  this scrubbing mechanism is small for it. Not used for numbers.

### S3. Lupker et al. 2002: truck tyre wear assessment and prediction (EU project TROWS)

- **Reference.** Lupker, H.A.; Montanaro, F.; Donadio, D.; Gelosa, E.; Vis, M.A. (2002). Truck Tyre Wear
  Assessment and Prediction. 7th International Symposium on Heavy Vehicle Weights and Dimensions, Delft,
  16-20 June 2002, pp. 275-288. https://hvttforum.org/wp-content/uploads/2019/11/Truck-Tyre-Wear-Assessment-And-Prediction-L-upker.pdf.
  TROWS = Tyre and ROad Wear and Slip assessment, an EU 5th Framework project (TNO, Pirelli and others).
- **Evidence basis.** Read in full (14 pages, text layer; the figure pages 11-14 are poorly extracted).
- **Findings.**
  - Typical wear of a 40-t long-haul truck on motorways: front axle 30,000 to 15,000 km per mm, rear
    (drive) axle 15,000 to 10,000 km per mm (p. 279). That is 0.33-0.67 and 0.67-1.0 mm per 10,000 km.
  - "Besides the difference in front (steered) and rear (driven) axle, wear strongly depends on axle
    load" (p. 279). No load law is given.
  - Drive-axle tyres wear more than steer-axle tyres because of acceleration and braking forces;
    retarders (constant braking torque) improve km per mm (p. 279).
  - Route model: wear is taken as proportional to the work of the contact forces; each road section
    gets a "wear aggressivity", the work per unit length and unit weight (p. 278). It depends on vehicle
    mass and its distribution, drive or carried wheel, speed, bend radius and road element (p. 278).
  - On an aggressive hilly public-road course near Parma (134 km lap), the wear rate was about twice
    the carousel rate and four times the reference level; summer wear exceeded winter wear (p. 280).
  - The compound wear against frictional power per unit area is non-linear (p. 280, Figure 9).
- **Use here.** Background for the mechanism (work of contact forces) and the axle difference. Long-haul
  values are a lower bound for a city bus.

### S4. Swiss vehicle regulation VTS (SR 741.41), version in force from 1 October 2026

- **Reference.** Verordnung vom 19. Juni 1995 über die technischen Anforderungen an Strassenfahrzeuge
  (VTS; Ordinance on the technical requirements for road vehicles), SR 741.41. Consolidated German
  version of 1 October 2026, Fedlex XML:
  https://fedlex.data.admin.ch/filestore/fedlex.data.admin.ch/eli/cc/1995/4425_4425_4425/20261001/de/xml/fedlex-data-admin-ch-eli-cc-1995-4425_4425_4425-20261001-de-xml.xml
  (version list from the Fedlex SPARQL endpoint, 9 Nov 2026).
- **Evidence basis.** Read in part: Art. 58, 59, 60, 61 and 95 in full; all articles that contain
  "Reifen" (tyre) or "Profil" listed and their matching sentences read. No article sets a separate tread
  depth for buses.
- **Findings.**
  - **Minimum tread depth.** Art. 58 Abs. 4: "Die Reifen müssen auf der ganzen Lauffläche mindestens
    1,6 mm tiefe Profilrillen aufweisen" (tyres must have tread grooves at least 1.6 mm deep over the
    whole tread). The same paragraph forbids a damaged or exposed carcass ("Gewebe"). This holds for
    all motor vehicles, buses included; only slow vehicles are exempt (Art. 119, 138).
  - **Regrooving ("Nachschneiden", "Nachrillen").** Art. 60 Abs. 3 (version of 22 Dec 2023, in force
    since 1 April 2024): regroovable tyres must carry the marking of UNECE Regulation No. 54 or No. 109;
    "Das Nachschneiden anderer Reifen ist unzulässig" (regrooving other tyres is not allowed). The VTS
    sets no depth or count for regrooving; those come from the UNECE regulations and the tyre maker.
  - **Retreads ("aufgummierte Reifen").** Allowed, with the retreader's name, size, speed, load and
    construction marked on the tyre (Art. 60 Abs. 5). No axle restriction in the VTS.
  - **Weights.** Two-axle bus ("zweiachsige Gesellschaftswagen") at most 19.50 t gross (Art. 95 Abs. 1
    lit. dbis). Vehicles "nach Absatz 1 Buchstaben d, e–g und j" with zero-emission drive may be up to
    2 t heavier (Art. 95 Abs. 1bis); the text does not say whether lit. dbis is included. Axle loads at
    most 10.00 t for a non-driven single axle and 11.50 t for a driven single axle (Art. 95 Abs. 2 lit. a,
    b no. 3).
- **Use here.** Legal threshold 1.6 mm (both axles); the depot operator may replace earlier (S1: about
  2 mm).

### S5. Michelin UK: "Selling the Difference: Importance of Regrooving" (sales guide)

- **Reference.** Michelin UK, *Selling the Difference - Importance of Regrooving*, service guide (slides),
  uploaded May 2011:
  https://marketing.michelin.co.uk/ask-the-michelin-man/wp-content/uploads/2011/05/Michein-Service-Guide-Regrooving.pdf.
  A manufacturer's sales document about truck tyres in general, not about one bus tyre.
- **Evidence basis.** Read in full (9 slides, text layer).
- **Findings.**
  - Regrooving ("Nachschneiden") cuts a new pattern into the base rubber under the original tread
    (slide 2). It is done when 3-4 mm of tread remain (slides 3, 4).
  - After regrooving the tread depth is "about 6 to 8 mm, or a tread depth of about the same as that of
    a half-worn tyre" (slide 4).
  - "On average up to 25% additional mileage" (slide 4); "~25% increase in tyre life" (slide 2).
  - Cost: "about 12% (~£25) of the cost of a new tyre" (slide 7). The same chart gives the "4 lives"
    of a Michelin casing: new tyre (life 100 %, cost 100 %), regroove (25 %, 12 %), Remix retread
    (100 %, 50 %), regroove of the retread (25 %, 12 %); in total 250 % life for 174 % cost (slide 7).
  - Remix (Michelin's retread) tyres can be regrooved too (slide 7).
- **Derived here (not in the source).** With new depth h0, regroove at r1 = 3-4 mm left, depth after
  regrooving r2 = 6-8 mm: the damage D = h0 - depth falls from h0 - r1 to h0 - r2. Under the ARD-infinity
  law of the model, rho = (r2 - r1) / (h0 - r1). For h0 = 17-20 mm and r2 - r1 = 3-4 mm, rho is about
  0.2-0.3. This matches the claimed 25 % extra life, if wear is linear.
- **Limits.** A marketing document; the 25 % figure is a manufacturer claim, not a measured result.
  Regrooving is normally done once per life (once on the new tyre, once on the retread); the model
  allows any number of repairs.

### S6. Daimler Buses: "The eCitaro. Technical data and equipment at a glance" (data sheet, 04/2026)

- **Reference.** Daimler Buses GmbH (2026). *The eCitaro. Technical data and equipment at a glance*,
  international English edition, dated 04/2026.
  https://www.mercedes-benz-bus.com/int/en/models/ecitaro/facts-ecitaro.pdf.
- **Evidence basis.** Read in full (13 pages, text layer).
- **Findings for the 12 m eCitaro (2 doors and 3 doors).**
  - Length 12,135 mm; wheelbase 5,900 mm; two axles (p. 4).
  - Tyre size 275/70 R 22.5 standard; 315/60 R 22.5 as an option (p. 4).
  - Gross vehicle weight 20,000 kg; maximum front axle load 7,500 kg; maximum drive axle load
    13,000 kg (p. 9). The data sheet gives no unladen mass and does not say whether the rear tyres are
    twin tyres.
  - Drive: ZF AxTrax 2 LF electric axle, 360 kW continuous, 440 kW peak (p. 6). Batteries NMC
    (nickel manganese cobalt), 111 kWh per module, 444-666 kWh (p. 10). Up to 88 passengers (p. 8).
  - Tyre pressure monitoring is standard (pp. 7-8).
  - eCitaro G (18 m): GVW 29,000-30,000 kg, same tyre size, drive axle 13,000 kg (pp. 4, 9).
- **Use here.** The reference 12 m battery bus of this file. Its tyre size equals the 275/70 R22.5 group
  of S1, and its drive axle limit (13.0 t) is well above the 11.5 t of a diesel two-axle bus in the VTS
  (S4, Art. 95 Abs. 2 lit. b no. 3).

### S7. Giechaskiel, Ferrarese, Grigoratos 2025: tire wear, tread depth reduction and service life (review)

- **Reference.** Giechaskiel, B.; Ferrarese, C.; Grigoratos, T. (2025). Tire Wear, Tread Depth Reduction,
  and Service Life. *Vehicles* 7(2), 29. https://doi.org/10.3390/vehicles7020029. Open access.
- **Evidence basis.** Read in full, pp. 1-11 (text layer); the reference list (pp. 12-14) only scanned.
  Figures 1-3 not read; their numbers are in the text.
- **Scope.** Passenger-car (C1) tyres; no bus data. Used here for the shape of the wear path, the spread
  and the measurement error.
- **Findings.**
  - **Shape of the wear path.** 1950s road studies: tyres lost mass at a constant rate while groove depth
    fell "at a declining rate"; another found the log-log plot of tread loss against mileage linear (§2.1
    p. 2, their refs. 19, 20). Tread wear is often high during a break-in; the US grading test (UTQG,
    Uniform Tire Quality Grading) measures after a break-in of 800 miles (1,287 km) (§2.2 p. 3; §3.4
    p. 7). The authors conclude that the progression of depth and mass "is not always linear" (§4 p. 8).
  - **Spread of tyre life under similar use.** 50 identical summer tyres (ADAC test): 40,000 km with one
    standard deviation 10,000 km (coefficient of variation, CV, 25 %) (§3.2 p. 4). 5,000 tyres of one
    size on one car model: 37,000 ± 6,000 km front (CV 16 %), 64,000 ± 8,500 km rear (CV 13 %) (§3.2
    p. 4). The same tyres in different areas differed by about 20 % from the mean; within one area
    the variability was "even higher (by a factor of two)" (§2.1 pp. 2-3).
  - **Route effect.** Two routes differed by a factor 1.6 in wear rate; the higher one had more curves,
    hills and an abrasive surface (§3.3 p. 6). Urban and dynamic driving wear more than the rural and
    motorway routes behind the averages (§4 p. 8).
  - **Driven axle.** Front (driven and steered) to rear wear ratio 2.0 for summer tyres (§3.3 p. 6).
  - **Measurement error.** A tread gauge has about ±0.1 mm accuracy; extrapolation of a short test to
    the limit gives about ±13 % error in projected life (§3.4 p. 7). The legal check uses the minimum
    over all wear indicators (§3.4 p. 7).
  - **Mass per mm.** Tread mass loss ΔV = (1 - v) k b (π/4) [2 w (t/10) - (t/10)²] with void ratio v,
    contact width ratio k, section width b, wheel diameter w, depth loss t; density 1.15 g/cm3
    (Appendix A, Eqs. A1-A3). About 250-350 g per mm for car tyres (§4 p. 8).
  - Minimum depth 1.6 mm in the EU by Directive 89/459 (§1 p. 2).
- **Use here.** Support for a near-linear mean path with a possible break-in, for a spread of 15-25 %
  between tyres, and for a measurement error that matters only for short intervals.

### S8. Goodyear: Truck Tires Technical Data Book 2020-2021 (Europe)

- **Reference.** Goodyear (2020). *Truck Tires Technical Data Book 2020-2021* (file "EM Databook TTDB
  2020-2021", brochure "TTDB 2020 A4-E", 78 pages, created 12 Oct 2020).
  https://www.goodyear.eu/content/dam/goodyear/truck/pdfs/em/EM%20Databook%20TTDB%202020-2021.pdf.
- **Evidence basis.** Read in part: p. 21 (bus and coach tyres), p. 38 (technical data table for
  22.5-inch sizes, read from the rendered page), pp. 44-45 (regrooving guidelines), p. 53 (regroove guide
  for bus tyres). The other pages were only searched for "tread depth", "regroov" and "retread".
- **Findings.**
  - City-bus tyres "UrbanMax": MCA, steer or all-position, "developed to provide excellent mileage in
    'stop & go' applications", "Regroovable and retreadable"; MCD Traction, drive axle (p. 21).
  - **New tread depth, size 275/70 R22.5** (p. 38, column "Tread Depth (mm)"): UrbanMax MCA 19.5 mm
    (also the high-load version 150/145J); UrbanMax MCD Traction 20.4 mm. For comparison, regional tyres
    KMAX S 13.8 mm and KMAX D 18.1 mm. Overall diameter of the MCA 976 mm, section width 271 mm,
    rolling circumference 2,979 mm (p. 38).
  - **Regrooving rules** (pp. 44-45): regroove "when 3 to 6mm of the original design is still left"
    (p. 44); 3-4 mm for highway use, 5-6 mm where penetration damage is likely (p. 45). "The maximum
    regrooving depth for all Goodyear tires is 3mm"; 3 mm of rubber must remain under the new grooves
    (p. 44). Regrooved tyres "will usually be mounted on the rear axles" (p. 44).
  - A casing can have "multiple lives (new, regrooved, retread, regrooved retread)" (p. 44). A Goodyear
    remould may be regrooved with the same 3 mm maximum (p. 45).
  - Bus tyres MCA and MCD Traction: "Maximum regrooving depth 3mm, regrooving width 6-8mm. Retreadable
    by Goodyear TreadMax" (p. 53).
- **Derived here (not in the source).** A Goodyear regroove adds at most 3 mm of groove depth. With new
  depth h0 = 19.5-20.4 mm and regrooving at 3-6 mm left, the damage before regrooving is 13.5-17.4 mm, so
  rho = 3 / (h0 - remaining) is about 0.17-0.22 under the ARD-infinity law.

### S9. Clark et al. 2026: what quality engineers need to know about degradation models (review)

- **Reference.** Clark, J.M.; Min, J.; Li, M.; Warr, R.L.; DeHart, S.P.; King, C.B.; Lu, L.; Hong, Y.
  What Quality Engineers Need to Know about Degradation Models. arXiv:2507.14666v3 [stat.AP], 5 Feb 2026.
  https://arxiv.org/abs/2507.14666.
- **Evidence basis.** Read in part, pp. 9-13 (Section 3, "Degradation Models": general path models and
  stochastic process models). The word "tire" does not occur in the text; "wear" occurs on pp. 5 and 8
  only (keyword search of all 38 pages).
- **Findings.**
  - "The three most commonly used processes for modeling degradation paths are the Wiener process
    (Whitmore, 1995), gamma process (Lawless and Crowder, 2004), and inverse Gaussian (IG) process (Wang
    and Xu, 2010)" (§3.2 p. 12). All three have independent increments (pp. 12-13).
  - General path models (GPM): a parametric path, for example linear dD/dt = theta, with unit-specific
    random parameters for unit-to-unit variability (§3.1, Eqs. 1, 2, 5; pp. 11-12). The authors
    recommend GPMs "in most cases" (§3.2 p. 13).
  - Stochastic process models capture the deviations from the mean path as part of the distribution;
    they are biased if the mean trend is misspecified (§3.2 p. 13).
- **Use here.** Evidence that the Wiener, gamma and IG processes are the established families for
  degradation in general. It is not evidence about tyres.

### S10. Boulter et al. 2006 (TRL PPR110): non-exhaust particulate matter, final report on emission modelling

- **Reference.** Boulter, P.G.; Thorpe, A.J.; Harrison, R.M.; Allen, A.G. (2006). *Road vehicle non-exhaust
  particulate matter: final report on emission modelling.* TRL Published Project Report PPR110, for DEFRA
  (UK Department for the Environment, Food and Rural Affairs) and the devolved administrations.
  https://naei.energysecurity.gov.uk/sites/default/files/cat15/0706061624_Report2__Emission_modelling.PDF.
  Year from the file date code (0706...); the pages read show no date.
- **Evidence basis.** Read in part: §3.2 (printed pp. 16-19 = PDF pp. 21-24), Appendix B on the EMEP/
  CORINAIR method (printed pp. 82-85 = PDF pp. 87-90), Appendix C on the HDM-4 tyre model (printed
  pp. 86-91 = PDF pp. 91-96). Equation C1 and Tables C1-C3 read from rendered pages. EMEP/CORINAIR is
  the European emission inventory guidebook (European Monitoring and Evaluation Programme); HDM-4 is the
  World Bank's Highway Development and Management model.
- **Findings: EMEP/CORINAIR tyre-wear method (Appendix B).**
  - Heavy-duty vehicles: e_HDV = (N_axle / 2) × LCF × e_PC (Eq. B3), with e_PC the car factor.
    LCF = 1.41 + 1.38 × LF (Eq. B4), "derived by linear regression on experimental data"; LF = 0 for an
    empty and 1 for a fully laden vehicle; "The same equations can be used for urban buses and coaches"
    (printed p. 82). So a fully laden vehicle wears 2.79 / 1.41 = 1.98 times as fast as an empty one.
  - Speed correction S_T(V): 1 at 80 km/h; about 1.39 at and below 40 km/h, flat there "due to the
    absence of any experimental data"; obtained on cars but "to be used for all vehicle categories"
    (printed p. 83, Figure B1 read from the rendered page). Reason given: more braking and cornering in
    urban driving.
- **Findings: HDM-4 tyre model (Appendix C).**
  - Mechanistic slip-energy model of Carpenter and Cenek (1999), proposed for HDM-4 (Eq. C1, printed
    p. 87): TWT = FLV × (C0tc + Ctcte × FNC × CFT² / NFT + Ctcte × FNL × LFT² / NFT), with TWT the tread
    volume worn in dm3 per 1,000 tyre-km, CFT and LFT the mean absolute circumferential and lateral
    forces on the tyre (N), NFT the normal load (N), FNC and FNL factors for force fluctuations, FLV a
    local calibration factor (set to 1).
  - Heavy bus (and all trucks and buses): C0tc = 0.001 dm3/1000 km, Ctcte = 0.0003 dm3/MNm (Table C1).
    "The values for C0tc were found to be too low and resulted in unreasonably high tyre lives"
    (printed p. 88).
  - FNC grows with speed variation and falls with speed; heavy truck at 30 km/h: 1.7 (low variation,
    acceleration noise 0.1 m/s2), 7.6 (medium, 0.3 m/s2), 29.4 (high, 0.6 m/s2) (Tables C2, C3). FNL = 2.5
    for all routes and vehicles (printed p. 89). Default heavy truck: CFT = 500 N, LFT = 907 N, NFT =
    12,753 N per tyre (printed pp. 89-90).
  - Status: "although the mechanistic theory of tyre consumption is well founded it has not been fully
    implemented due to an inability to obtain satisfactory model parameters" (printed p. 17); HDM-4 v1.0
    uses an interim model TWT = C0tc + Ctcte × TE, TE the tangential energy (Eq. C3, printed p. 90),
    whose predictions "should be viewed with caution" (printed p. 91).
- **Derived here (not in the source).** In Eq. C1, CFT = m_w a_x and NFT = m_w g for a wheel load m_w and
  a mean absolute longitudinal acceleration a_x. Then CFT² / NFT = m_w a_x² / g: at fixed accelerations,
  speeds and curves, tread wear per km grows linearly with the mass per wheel.
- **Use here.** Load scaling (EMEP, an established inventory method) and the structure of a
  slip-energy formula (HDM-4, a model whose parameters its authors call unsatisfactory).

## 4. Gaps

| Id | Missing value | Why it matters | How to fill it |
|---|---|---|---|
| G1 | Battery-bus wear factor from tread-depth data | f_BEB rests on a cost ratio (R8) and on a route-confounded tyre count (R1) | Tread readings of the operator (EUROBUS, VBG) for battery and diesel buses on the same lines; the VTI survey (section 6) |
| G2 | Passenger load per line | EMEP load factor (S10 Eq. B4) | Passenger counts of ZVV or VBG per line |
| G3 | Tread depth and wear of a city-bus retread | replacement_mu for a retread; S1 found retreads within 15 % of new tyres | Retread data sheets (Goodyear TreadMax, Michelin Remix) |
| G4 | Noise of one tyre's own wear path | Needed for closed-loop re-planning with measured depths (section 2.2) | Repeated depth readings per tyre (S1 had them for few tyres only, Figure 2) |
| G5 | Tread depths of a running bus fleet | mu_0 | Operator's tyre records; car-fleet distribution in the Promet 2022 paper (section 5) |
| G6 | Unladen mass of the 12 m eCitaro and of a diesel twin | Mass scaling of F2 and the battery-bus factor | Certificate of conformity or Swiss type approval of the vehicles |
| G7 | Absolute prices in CHF: new tyre, retread, regroove, fitting labour, depot time | C_rep, C_R, C_M | Supplier quote or a Swiss public tender (simap.ch). Search snippets (not read, not evidence) showed about 390-400 GBP without VAT (UK) and 600-640 EUR with VAT (Germany) for 275/70 R22.5 bus tyres |
| G8 | Regroove timing and count | The model allows a repair at any depth and any number of times | Model rule: allow a repair only when D >= h0 - 6 mm, once per tread life (S8 pp. 44-45) |
| G9 | Acceleration at stops and cruise speed per line | Inputs of F2 (a = 1.0 m/s² and v = mean speed are assumptions) | Vehicle telemetry of the operator, or standard bus test cycles (UITP SORT, not read) |
| G10 | epsilon | No source gives a risk level for tread depth | Policy choice; the legal rule "must" (VTS Art. 58) suggests a small value |
| G11 | Temperature and road surface | Summer wear above winter wear (S3 p. 280); no number for buses | Seasonal profiles from operator data |
| G12 | Split of the battery-bus factor between axles | The drive axle carries up to 13.0 t (S6) and the regenerative braking force; f_BEB is applied to both axles | Operator data per axle |

## 5. Could not read

- **Fwa, T.F.; Ang, B.W. (1991).** Tire-wear characteristics of public buses. *Journal of Transportation
  Engineering* 117(3), 298-310. https://doi.org/10.1061/(ASCE)0733-947X(1991)117:3(298). 9 Nov 2026:
  ASCE answered 403 to PowerShell and showed a Cloudflare human-verification page in the browser; not
  passed. 10 Nov 2026: the article page loaded in the browser without a check, but it shows "Get
  access", and the PDF link answered HTTP 403 with an HTML page; the ETH licence seems not to cover
  this article. OpenAlex and Semantic Scholar list no open copy, and no secondary source found quotes
  its regression. It looks the most relevant missing source: tread wear of 144 bus tyres (authors at
  the National University of Singapore), with regression on route and service parameters (abstract
  only, not evidence). It would give a data-based route factor in place of F2. Can you obtain it,
  for example through the document delivery service of the ETH Library?
- **Bagdonavicius, V.; Nikulin, M. (2001).** Estimation in degradation models with explanatory variables.
  *Lifetime Data Analysis* 7, 85-103. https://doi.org/10.1023/A:1009629311100. Springer served a purchase
  page (no ETH access). Possibly a Gamma process model with covariates applied to tyre wear (keywords).
- **Pečeliūnas, R.; Žuraulis, V.; Droździel, P.; Pukalskas, S. (2022).** Prediction of road accident risk
  for vehicle fleet based on statistically processed tire wear model. *Promet - Traffic&Transportation*
  34(4), 619-630. https://doi.org/10.7307/ptt.v34i4.3997. hrcak.srce.hr answered 418 (bot block). Car
  fleet; a Gamma distribution of tread depths of 4,288 tyres (search snippet); a lead for G5 and G10.
- **Markov model of tyre wear status** (Galați University journal, article 3932,
  https://gup.ugal.ro/ugaljournals/index.php/eeaci/article/view/3932). 403.
- **EMEP/EEA Guidebook 2023, chapter 1.A.3.b.vi-vii (road tyre and brake wear).** The EEA site answered 410
  or closed the connection. S10 carries the older EMEP/CORINAIR equations with locations (a reliable
  secondary source); the current coefficients may differ.
- **Retail price pages** (Big Tyres UK) answered 403. **Hankook TBR technical manual** (host not
  resolved). **Michelin document 668311** (dxm.contentcenter.michelin.com): the download failed and was not
  retried. **Continental Urban HA3** product data: the page is built by JavaScript; no PDF found.

## 6. Leads not followed

- Lepine, J.; Na, X.; Cebon, D. (2022). An empirical tire-wear model for heavy-goods vehicles. *Tire Science
  and Technology* 50, 211-229 (cited by S1; a version may be in the Cambridge repository,
  https://www.repository.cam.ac.uk/handle/1810/372114). In-service data of a semitrailer in London.
- Bennett and Greenwood (2001), HDM-4 Volume 7, and Carpenter and Cenek (1999): the primary sources of
  the slip-energy model carried by S10.
- VTI (Swedish National Road and Transport Research Institute) survey on tyre wear of electric vehicles,
  https://core.ac.uk/works/136216984 (a snippet: 8 of 11 bus companies saw faster tyre wear than on diesel
  buses; not evidence). A lead for G1.
- Krishnappa, Matthews, Liu (2019), *Transportation Research Record* 2673, 517-528: 600,000 inspection
  records of tread depth (cited by S7, cars).
- Liang et al., tread wear of truck and bus radial tyres (cited in the reference list of S7).
- Goodyear UrbanMax 2022 brochure, https://www.goodyear.eu/content/dam/goodyear/truck/pdfs/de_urbanmax_2022.pdf.
- Daimler Truck press release "Technical data for Mercedes-Benz eCitaro with NMC 3 batteries": keyword
  search only (not read in full, not cited). It lists 8,000 / 13,000 kg axle loads, while S6 lists 7,500 kg
  for the 12 m front axle.
- Ghotge et al. 2025, *World Electric Vehicle Journal* 16, 464 (bus TCO in Europe) and the GIZ 2023
  "Comparative analysis of bus technologies": keyword search found no tyre cost item; not read, not cited.
- UITP SORT (Standardised On-Road Test) cycles, for the stop accelerations of G9.
