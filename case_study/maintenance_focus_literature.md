# Maintenance focus of battery-electric buses: peer-reviewed literature

Status: done (10 Nov 2026). Eight sources entered, S1-S8, all read in full except S3 and S8 (read
in part, see their entries). Summary table, ranking, "Could not read" and leads are filled. Read and
excluded: five papers (section 5.1). Downloads in the session scratchpad, downloads\lit\01-13:
S6 in \09, S7 in \12 (file nan2022_energies_b.pdf); \12\nan2022_energies.pdf and
\13\manzolli2022.pdf are HTML error pages, not papers; S8 in downloads\fetch\03. Added on 10 Nov
2026: S8, the PhD thesis of H. A. Taha (2022), which holds the ETFA 2021 paper and a later
manuscript on the same bus drive. Taha et al. 2023 and Manzolli et al. 2022 remain unread
(section 4). Next, if more work is wanted: those two through the user, and the grey literature of
section 5.2.

Scope: journal and conference papers and peer-reviewed reviews on reliability, failures,
maintenance or predictive maintenance of battery-electric buses (BEB). Grey literature appears
only as leads (section 5.2).

## 1. Summary table

"My computation" marks numbers derived here from the sources' own figures. Shares from S7 are
shares of the nine fault types that S7 lists, not of all faults.

| Component | Weight in BEB maintenance (evidence, source, location) | Usage driver | Maintenance actions and interval | Sources |
|---|---|---|---|---|
| **Whole vehicle, against diesel** | Lublin: mean km between failures 1714 (BEB) vs 2029 (diesel), 15.5 % lower; Weibull shape 0.92 vs 1.41 (S1, Table 2-3). Availability 0.89 (BEB) vs 0.79 (diesel) over 33 months (S2, Table 2). Portland: 2124 vs 4630 planned miles to breakdown, but +464 miles for BEB once routes and use are controlled (S5, Tables 1, 3). The sources do not agree on the sign. | Ramp use, stops, passenger load, temperature (S5, Table 3); winter (S1, §2) | Corrective repair at the depot, at once or next shift (S1, §2; S2, §3.1). Inspection every 6 months in Lublin (S2, §4). Recommended check every 2 weeks, about 3700 km (S5, p. 9; model-based) | S1, S2, S5 |
| **Traction battery** (cells, contactors, BMS) | Among the 3 least reliable systems by Weibull scale, b = 15,957 km (S1, Table 8, §4.2.1). Battery degradation (F1): the largest downtime, about 310 h of the trolleybus fleet's failure downtime, and about 21 % of failure-plus-downtime cost, my computation (S4, Figure 12, Tables 12-13). Cell imbalance is the top fault type, 12,506 mentions; battery items 20,202 of 49,673 listed mentions, 41 %, my computation (S7, Table 1). | Equivalent full cycles, depth of discharge, cell temperature, calendar age (S4, §2.4, §4.4); cycles per day and pack layout (S6, p. 9); charging events for contactors (S1, §2) | Contactor and module repair (S1, §2). Pack replacement after 5 years, or condition-based to 6.5 years with SOH ≥ 80 % and resistance gates (S4, §2.5, Table 3; scenarios, not the operator's practice). Monthly BMS review, yearly capacity test proposed (S4, §4.7) | S1, S4, S6, S7 |
| **Charging system and interface** (pantograph, plug, bus-charger communication) | 506 of 3747 recorded failures, 13.5 %, my computation (S1, p. 14, Table 2); Weibull scale 16,424 km, third lowest (S1, Table 8). First-month availability 0.28, caused by charging adaptation (S2, §3.2). HV insulation or charging-contact fault F4: the highest rate per km, 0.552 per 100,000 km (S4, Table 12). BMS-charger communication and charging-base temperature: 10,303 mentions, 21 %, my computation (S7, Table 1). | Number of charging events and contact quality (S1, §2); not quantified | Mechanical repair of pantograph and plug (S1, §2). No interval given | S1, S2, S4, S7 |
| **High-voltage system and traction power electronics** (inverter, converters, insulation, HV contactors) | Traction power-electronics failure F3: 20 events, 0.325 per 100,000 km, but the second-largest downtime (about 165 h) and costly events (median about 160,000 RUB); about 16 % of failure-plus-downtime cost, my computation (S4, Table 12, Figures 12-13). Lublin HVS: Weibull scale 102,600 km, the most reliable of the 8 systems (S1, Table 8). HV open circuit and overcurrent: 10,904 mentions, 22 % (S7, Table 1; may include battery faults). **No field source splits the inverter into power modules (IGBT or SiC) or DC-link capacitors.** Model only, handbook rates: inverter PCB 1.12, IGBTs 0.085 failures per 10⁶ h for an in-wheel-motor bus; inverter PCB importance 1.55, after the control software (1.82) and the controller board (1.57) (S8, Table 8.1, §8.4) | Not quantified for buses. Wet weather for braking-resistor insulation, wire abrasion (S1, §2) | Corrective repair; no interval given by a fleet. Proposed: inverter inspection every 2 years or 30,000 miles (S8, Table C.1) | S1, S4, S7, S8 |
| **Doors** | Lowest Weibull scale of the BEB systems, b = 14,154 km; 40 % lower than on the diesel buses, b = 23,713 km (S1, Tables 8-9, p. 15) | Winter: freezing slush, frozen gaskets (S1, §2) | Corrective repair; no interval | S1 |
| **Low-voltage (24 V) system** | Mean 1438 km between failures, Weibull scale 23,254 km (S1, Tables 7-8); the lowest scale on the diesel buses too, b = 10,895 km (S1, Table 9). 24 V undervoltage at the motor controller: 4690 mentions (S7, Table 1) | Parking: 24 V batteries discharge (S1, §2) | Corrective; no interval | S1, S7 |
| **Battery cooling and thermal management** | Mean 1434 km between failures, the lowest mean of the 8 systems; Weibull scale 40,657 km (S1, Tables 7-8). Thermal derating F2: 33 events, 0.536 per 100,000 km, about 9 % of failure-plus-downtime cost, my computation (S4, Tables 12-13). Heating or water-cooling relay: 3574 mentions (S7, Table 1) | Hot summers and cold winters (S4, Table 10); refrigerant and pump faults (S1, §2) | Corrective; no interval | S1, S4, S7 |
| **Brakes** | Weibull scale 35,674 km, middle of the 8 systems (S1, Table 8). Failures are sensors, slack adjusters and leaking cylinders, not pad wear (S1, §2). Model only: about 7 % more brake particulate from the extra weight, minus an assumed 25-75 % from regenerative braking (S3, §3.2) | Weight, stops, braking per km (S3, Eqs. 3-8) | Corrective; no pad-life data for buses | S1, S3 |
| **Heating and air conditioning (HVAC)** | Weibull scale 31,952 km (S1, Table 8) | Refrigerant leaks, fans (S1, §2) | Corrective; no interval | S1 |
| **Tyres** | **No peer-reviewed field data on tyre wear or tyre life of battery-electric buses found.** Model only: about 4 % more tyre particulate for a bus about 2000 kg heavier (S3, §3.3, Table 3). Not tracked in S1, S4 or S7 | Weight, stop-and-go, acceleration, slope (S3, §3.6, §5) | None given | S3 |
| **Traction motor** | **Not tracked as a separate system in any BEB field source read.** S4 groups it with "traction power electronics". Model only: handbook rates for an in-wheel motor, bearing seal 0.4465 and windings 0.277 failures per 10⁶ h each (S8, Table 8.1) | Bearing failure rate grows linearly with motor speed (S8, Eq. 9.11, from a US Navy handbook) | Proposed: bearing inspection every 2 years or 22,000 miles (S8, Table C.1) | S8 |
| **DC-link capacitors** | **No source found on DC-link capacitors in buses** (search terms in section 2) | - | - | - |

## 2. Ranking of the components by the evidence

**Overall.** The peer-reviewed field evidence is thin. Five field studies (S1, S2, S4, S5, S7)
and one field degradation study (S6) were read; S1 and S2 come from the same operator and
authors, and S4 is a trolleybus fleet. Only S1 compares systems of the same battery-electric bus
with each other; S4 and S7 cover only electrical faults. So the ranking below is by breadth and
consistency of evidence, not by a shared metric.

1. **Traction battery.** The only component that every component-level source ranks high: S1
   (low Weibull scale, contactor failures), S4 (largest downtime, costliest events, the
   life-limiting part), S7 (top fault type). It is also the only component with field degradation
   data (S4: SOH loss 4.06 ± 1.19 points per year; S6: down to 70-80 % after four years).
2. **Charging system and interface.** High in S1 (13.5 % of failures), S2 (first-month
   availability), S4 (highest rate per km) and S7 (21 % of listed mentions). Part of it is
   infrastructure, not the bus. No degradation data.
3. **High-voltage system and traction power electronics.** Few events but long downtime and high
   cost per event in S4; most reliable system in S1. The two sources disagree, and neither splits
   the inverter.
4. **Doors.** Least reliable BEB system in S1 and 40 % worse than on the diesel buses there. One
   source only; S4 and S7 exclude mechanical items.
5. **Low-voltage (24 V) system and battery cooling.** Frequent but small faults in S1 and S7;
   thermal derating in S4. Weather-driven.
6. **Brakes and HVAC.** Middle of the S1 ranking. Brake failures are sensors and adjusters, not
   wear. Brake wear of buses appears only in a model (S3).
7. **Tyres.** No peer-reviewed field evidence. One emission model (S3) predicts slightly more
   tyre wear from the extra weight. A search snippet of a grey-literature survey (section 5.2)
   mentions faster tyre wear on electric buses; the survey was not read, so this is not evidence.
8. **Traction motor.** No field evidence: no BEB source tracks it separately. One model (S8) uses
   handbook rates and names the motor bearings among the critical parts.
9. **DC-link capacitors and inverter power modules (IGBT or SiC).** No field evidence in buses. The
   same model (S8) uses handbook rates for the inverter board and IGBTs; it has no DC-link
   capacitor.

**About the four candidates named by the user.**

- *Traction battery*: strong evidence, and the only one with degradation data and a
  replacement decision (S4, S6).
- *Inverter power modules (IGBT or SiC)*: no bus source separates them. Search terms used: "IGBT
  module lifetime electric bus traction inverter mission profile", "electric bus OR city bus power
  module lifetime IGBT SiC bond wire solder fatigue real driving data", "electric bus motor
  controller IGBT junction temperature reliability", Crossref "electric bus inverter reliability
  lifetime". Hits were car, rail, photovoltaic or generic studies. The only bus-level study read,
  S8, takes its inverter and motor failure rates from handbooks and a van study, because
  "operational failure data are unavailable" (S8, §8.1).
- *DC-link capacitors*: no source found in buses. Search terms: "DC-link capacitor lifetime
  electric bus traction inverter film capacitor degradation vehicle". The nearest hit is a
  heavy-duty traction inverter design study (Chalmers, 2026), not read.
- *Tyres*: no peer-reviewed field data on bus tyres. Search terms: "tyre wear electric buses vs
  diesel buses measured tread wear mileage tire life city bus study", Crossref "tyre wear
  electric bus". Also searched: "brake pad wear electric bus regenerative braking measured
  service life km operator data paper" (no field data).

**Evidence that one fleet did something, against established practice.** All maintenance
intervals above are single-fleet practice (S2: inspection every 6 months; S4: quarterly service)
or a paper's own scenario or recommendation (S4: battery replacement at 5 or 6.5 years; S5: a
check every 2 weeks). None of the sources states an established interval for battery-electric
buses.

## 3. Source entries

### S1. Niewczas et al. 2025, reliability of electric buses in Lublin (Poland)

- **Reference.** Niewczas, A.; Rymarz, J.; Ślęzak, M.; Kasperek, D.; Hołyszko, P. (2025). Reliability
  Study of Electric Buses in the Urban Public Transport System. *Energies* 18(14), 3863.
  https://doi.org/10.3390/en18143863 (open access, CC BY).
- **Evidence basis.** Read in full (22 pages, all text, tables and figure captions).
- **Data.** Field data of one operator, MPK Lublin (Municipal Transport Company Lublin). 32 Solaris
  Urbino 12 electric buses (E12), new, 2021-2024, 36 months, 3747 failures. Reference: 20 diesel
  Solaris Urbino 12 (S12, Euro VI), 36 months, 2104 failures, from earlier work of the same group
  (§2, Table 2). The E12 has a 120 kWh lithium titanate oxide (LTO) battery, usable 96 kWh, a 240 kW
  asynchronous central motor, and 450 kW pantograph charging at 8 terminal stops plus 40-80 kW depot
  chargers (§2, Table 1). Mean daily mileage 238 km (electric) and 251 km (diesel) (§2).
- **Failure definition.** A "failure" is any recorded state of unfitness that sends the bus to the
  depot; source: internal damage reports (§2, §1). So the counts include small defects.
- **Findings, whole vehicle.**
  - Mean mileage between failures: 1714.3 km (electric) vs 2029.3 km (diesel), 15.5 % lower for
    electric (Table 2; Eq. 3).
  - Weibull fit: shape c = 0.923, scale b = 1647 km (electric); c = 1.410, b = 2256 km (diesel)
    (Table 3). The authors read c < 1 as random failures and early defects, c > 1 as wear (§4.1.1).
  - Kruskal-Wallis test says the difference is significant (p = 0.0248, Table 4). The median test
    says it is not (p = 0.6024, Table 5).
  - Repair time (quoted from other works, not measured here): 1.95 h per electric-bus failure
    (ref. 45 = source S2) vs 6.4 h for diesel buses (ref. 13, a 1993 Polish book) (§4.1.2, p. 14).
- **Findings, systems of the electric bus.** Eight systems were chosen by expected failure rate and
  operator experience (§4.2.1). Traction motor, inverter, tires and suspension are **not** tracked
  as separate systems for the electric bus. The high-voltage system (HVS) includes "a number of
  converters" (§2, p. 7).

  | System (E12) | Mean km between failures (Table 7) | Weibull c, b [km] (Table 8) | km at R = 0.8 / R = 0.5 (Table 10) |
  |---|---|---|---|
  | Traction battery (TB) | 2172 | 0.909, 15,957 | 3024 / 10,032 |
  | Battery cooling (BCS) | 1434 | 0.859, 40,657 | 6100 / 26,464 |
  | Charging system (ChS) | 1449 (506 failures) | 0.856, 16,424 | 2835 / 10,277 |
  | High-voltage system (HVS) | 1603 | 0.862, 102,600 | 10,800 / - |
  | Electrical (low-voltage) system | 1438 | 0.838, 23,254 | 3870 / 15,060 |
  | Doors | 1586 | 0.888, 14,154 | 2610 / 9430 |
  | Brakes | 1619 | 0.910, 35,674 | 6630 / 23,636 |
  | Heating and air conditioning (HVAC) | 1670 | 0.911, 31,952 | 6156 / 8020 |

  - Lowest Weibull scale, so the least reliable systems: doors, traction battery, charging
    system (§4.2.1, p. 14; §5).
  - Diesel reference, lowest scale: electrical system 10,895 km, engine 22,878 km, doors 23,713 km
    (Table 9; the caption says "S12 electric buses", but S12 is the diesel sample). Door scale is
    40 % lower for electric than for diesel (p. 15).
  - **Caveat (my check).** Table 7 and Table 8 do not agree. With c ≈ 0.9 the Weibull mean is about
    1.05 b, so about 16,700 km for the battery, not the 2172 km of Table 7. The paper does not
    explain this. Use the ranking, not the absolute numbers.
- **Typical failure modes reported by the operator (§2, pp. 5-8).**
  - Traction battery: contactors on both battery terminals fail often, linked to charging
    interruptions (bus-charger communication, pantograph contact). Electrical connections and
    microprocessor parts are the most fail-safe.
  - Battery cooling: refrigerant (R407) loss and glycol pump defects.
  - Charging: mechanical failures of the pantograph current collector and of the Combo 2 plug;
    bus-charger communication; weak insulation of the bus.
  - High-voltage: contactors of auxiliary devices; insulation wear from wire abrasion; braking
    resistor insulation in wet weather.
  - Low-voltage electrical: on-board 24 V batteries discharged while parked; the most common cause
    of unfitness of this system.
  - Doors: freezing slush and frozen gaskets in winter.
  - Brakes: caliper area: wheel speed sensors, pad wear sensors, blocked slack adjusters, moisture
    from leaking brake cylinders.
  - HVAC: refrigerant leaks, heater fans, rooftop units.
- **Usage drivers.** Not quantified. Weather (cold, slush) is named for doors; charging events for
  battery contactors and the charging system.
- **Maintenance actions.** Corrective repair at the depot, immediately or at the next shift (§2).
  Electric-bus repairs "generally involve the rapid replacement of entire structural modules",
  quoted from Malozyomov et al. 2023 (p. 14). The cited paper does not say this (section 5.1), so
  this is the claim of S1's authors only. No intervals given.
- **Degradation data.** None. Only failure counts and mileage between failures. No battery
  state of health.

### S2. Niewczas et al. 2025, availability of electric vs diesel buses in Lublin

- **Reference.** Niewczas, A.; Mórawski, Ł.; Dębicka, E.; Rymarz, J.; Kasperek, D.; Hołyszko, P.
  (2025). A Comparative Study of the Availability of Electric Buses in the Public Transport System.
  *Applied Sciences* 15(3), 1212. https://doi.org/10.3390/app15031212 (open access, CC BY).
- **Evidence basis.** Read in full (14 pages; Figures 1-5 are images, read through captions and
  the numbers in Table 2).
- **Data.** Same operator as S1 (MPK Lublin). 18 new Solaris Urbino 12 electric vs 18 new Ursus
  City Smile CS 12 LFD diesel buses, 33 months, 2021-2024 (§3.1, Table 1). Daily mileage about
  238 km (electric) and 243 km (diesel) (§3.1). Source: daily technical availability reports of the
  operator's IT system (§3.1).
- **Metric.** Technical availability K_G = available buses / (available + unavailable), per day
  (Eq. 1). A bus with a failure on a day counts as unavailable that day (§3.1).
- **Findings.**
  - Mean availability over 33 months: 0.89 (electric) vs 0.79 (diesel) (Table 2; §3.2).
    Electric is 11.2 % higher; by year 0.9 %, 19.3 % and 13.5 % higher (§4).
  - Electric buses: lowest month 0.28, in the first month, caused by the adaptation of the
    charging system to local conditions (§3.2, §4). No trend over three years (§4).
  - Diesel buses: availability drops after the first year (0.85 to 0.74); the authors attribute
    this to mechanical wear (Table 2; §4). This attribution is the authors' reading, not a
    measured cause.
  - The difference is significant in years 2 and 3 and over 33 months (ANOVA, p < 0.001), not in
    year 1 (p = 0.065) (Table 6).
- **Components.** None. The paper gives no breakdown by component or failure type. The authors
  list "a powertrain failure comparison between diesel and electric vehicles" as future work (§4).
- **Maintenance actions and intervals.** Electric buses have periodic inspections every 6 months
  in this fleet; the authors link a 6-month dip of availability to these inspections (§4,
  Figure 3). This is one fleet's practice. Corrective repairs at the depot, at once or at the
  next shift (§3.1).
- **Degradation data.** None.
- **Caveats (my reading).** Table 3 (normality tests) repeats the numbers of Table 2, so it looks
  like an editing error. One sentence of §3.1 says a bus "without faults ... was counted as an
  unavailable vehicle", which contradicts the definition; it looks like a typo. The two sources
  S1 and S2 come from the same authors and operator; they are not independent evidence.

### S3. Tivey et al. 2023, tyre and brake wear emissions of battery-electric vs diesel buses (model)

- **Reference.** Tivey, J.; Davies, H.C.; Levine, J.G.; Zietsman, J.; Bartington, S.;
  Ibarra-Espinosa, S.; Ropkins, K. (2023). Meta-Analysis as Early Evidence on the Particulate
  Emissions Impact of EURO VI on Battery Electric Bus Fleet Transitions. *Sustainability* 15(2),
  1522. https://doi.org/10.3390/su15021522 (open access, CC BY).
- **Evidence basis.** Read in part: pp. 1-22 read in full (main text, Tables 1-11, Appendix
  A.1-A.3). Pp. 23-28 (Appendix A.4 onward, more tables of emission results) only scanned for
  the words wear, maintenance, measure, replace, life; no hit beyond table notes. Pp. 29-30
  are the references.
- **What it is.** A desk model of particulate emissions, not a maintenance study. It contains
  **no measured tyre or brake wear**. The authors say there is "little direct surveillance data"
  on battery-electric bus brake and tyre emissions (§3.2, §3.3). Tyre and brake wear enter only as
  airborne particulate per km, from weight-based emission factors of Beddows and Harrison (§2.2,
  Eq. 3).
- **Fleet described.** First Bus, York depot (UK): 5 Volvo B9TL Euro VI diesel and 5 Optare
  Metrodecker battery-electric double-deckers, with regenerative braking on the electric buses
  only (§2.1, Table 1).
- **Findings relevant to tyres and brakes.**
  - The battery-electric bus is 11 % (with 50 passengers) to 15 % (empty) heavier; about
    2000 kg more (§2.1, Table 1).
  - The weight model gives about 4 % more tyre particulate and about 7 % more brake particulate
    for the heavier electric bus, before regenerative braking (§3.3, Table 3; §3.2, Table 2).
  - Regenerative braking: the authors assume a 25-75 % cut of friction-brake emissions for buses,
    not the 90 % used for cars, because friction brakes still stop the bus and a heavy vehicle
    uses them more (§3.2, Eq. 4). This range is an assumption, not a measurement.
- **Usage drivers named.** Vehicle weight (Eq. 3); urban stop-and-go driving and low average
  speed (§3.6, §4, Table 9); acceleration and braking per km, as proxies for tyre and brake work
  (Eqs. 7-8); road slope, named as a likely driver outside the study (§5).
- **Maintenance actions and intervals.** None given.
- **Degradation data.** None measured. The paper announces a follow-up with on-road data, the
  "First Bus E6DV-to-BEV Bus Fleet Transition Evaluation" (§5, Acknowledgments); see section 5,
  Leads.

### S4. Malozyomov et al. 2026, battery trolleybuses with autonomous range (Russia), field data

- **Reference.** Malozyomov, B.V.; Martyushev, N.V.; Tynchenko, V.S.; Gladkikh, V.A.; Panfilova,
  T.A.; Govorkov, A.S.; Tynchenko, V.V.; Modina, M.A. (2026). Field-Based Reliability and Battery
  Lifetime Assessment of Autonomous-Range Trolleybuses. *World Electric Vehicle Journal* 17(7), 377.
  https://doi.org/10.3390/wevj17070377 (open access, CC BY).
- **Evidence basis.** Read in full (43 pages; pp. 39-43 are notation and references). Figures 12-14
  read from rendered pages 25-26.
- **Scope decision.** In scope as an adjacent vehicle class, with care. An "autonomous-range
  trolleybus" runs on the overhead line and on its traction battery where the line is missing
  (in-motion charging). It shares the battery, power electronics and charging-contact risks of a
  battery-electric bus. It is not a depot-charged or opportunity-charged bus.
- **Data.** One operator, one city (not named), 12 months. 110 trolleybuses of the "Gorozhanin"
  family; 40,150 vehicle-days, 3960 pack-month battery records, 584 maintenance, failure and
  downtime events (Table 1). Total 6.16 million km; 24.5 % of the km on battery (Table 4). Mean
  daily mileage 153.4 km (Table 4); mean annual mileage 55,987 km per vehicle (Table 9). Winter
  down to about -15 °C monthly mean (Table 5).
- **Important limit of the event data.** The 584 events hold only five failure modes and one
  scheduled-maintenance group (§2.5, Figure 12): 144 failures and 440 scheduled-maintenance events.
  No tyre, brake, door, suspension or body events appear. So the paper says nothing about the
  weight of the battery against the rest of the vehicle.
- **Failure modes (Table 12, p. 25).** Events in 12 months; rate per 100,000 km; cost.

  | Mode | Events | Per 100,000 km | Cost [RUB/km] | Median cost per event (Figure 13) |
  |---|---|---|---|---|
  | F1 Battery degradation | 28 | 0.455 | 0.682 | about 155,000 RUB |
  | F2 Battery thermal derating | 33 | 0.536 | 0.284 | about 55,000 RUB |
  | F3 Traction power-electronics failure | 20 | 0.325 | 0.513 | about 160,000 RUB |
  | F4 High-voltage insulation or charging-contact fault | 34 | 0.552 | 0.235 | about 45,000 RUB |
  | F5 Auxiliary or high-voltage control fault | 29 | 0.471 | 0.279 | about 60,000 RUB |
  | F6 Scheduled maintenance | 440 | - | - | about 18,000 RUB |

  - Downtime (Figure 12): battery degradation (F1) has the largest downtime, about 310 h; power
    electronics (F3) next, about 165 h; the others about 90-115 h. The authors: battery degradation
    and traction-system failures "contribute disproportionately to downtime" (p. 25).
  - Battery degradation and power electronics have the most expensive single events (Figure 13).
    The power-electronics failure is not split into IGBT modules, capacitors or other parts.
  - Maintenance and downtime together cost 3.235 RUB/km; energy 9.066 RUB/km; the battery
    replacement share 16.718 RUB/km under a 5-year replacement (Table 13). **My computation:** the
    five failure modes give 1.99 RUB/km, so about 62 % of the maintenance-and-downtime cost; F1
    alone about 21 %, F3 about 16 %.
- **Battery degradation data (Table 9, Table 10).** Mean state of health (SOH) fell from 89.98 % to
  85.94 % in 12 months; mean internal resistance rose from 14.92 to 15.83 mΩ; mean equivalent full
  cycles (EFC) rose from 37.6 to 392.3 (Table 10). Per vehicle, SOH drop 4.06 ± 1.19 percentage
  points per year, range 1.70-6.90 (Table 9). Lowest final SOH 78.9 %. The authors call the battery
  "the principal service-life-limiting component" (p. 22).
- **Usage drivers.** Named, not separated causally: EFC (cycling), depth of discharge, cell
  temperature and thermal history, calendar age (§2.4, §4.4). Thermal-derating events peak in
  summer and winter months (Table 10: about 100 per month in Dec-Feb and Jul-Aug, 11-17 in spring
  and autumn). Battery-mode km share raises energy per km by about 15 % (Table 4).
- **Maintenance actions and intervals.**
  - Scheduled maintenance: Figure 14 shows 110 events in each of March, June, September and
    December, so one quarterly service per vehicle (my reading). Downtime about 1 h per event
    (Figure 12, about 440 h for 440 events; my reading).
  - Battery: a fixed 5-year replacement (strategy B1) against a condition-based extension to
    6.5 years (B2) (§2.5). Both are scenarios of the paper; it does not say which one the operator
    uses. B2 gates (Table 3): SOH ≥ 80 %, internal resistance ≤ 18 mΩ and ≤ +20 %, SOH spread
    between packs ≤ 3 points, EFC ≤ 1500, monthly BMS (battery management system) review,
    quarterly decision, capacity test at least once a year (§4.7).
- **Degradation data.** Yes for the battery: monthly SOH, internal resistance and EFC per pack.
  None for other components.
- **Caveats (my reading).**
  - The authors state that the validation data set of the energy model is "generated", and that
    the B2 effects on FMEA (failure mode and effects analysis) scores are assumptions (§4.6).
  - Table 12 gives downtime per 10,000 km that is about 2.5 times the totals of Figure 12 (for F1,
    1.229 h per 10,000 km over 6.16 million km is about 757 h, but Figure 12 shows about 310 h).
    The paper does not explain this. Use the ranking, not the hours.
  - The text says thermal-derating failures (F2) cluster by season (p. 26). The heat map
    (Figure 14) shows at most 6 events in a month, in April. The derating counts of Table 10 do
    cluster in summer and winter.
  - The 8 routes of Table 8 have almost the same battery share (28.8-31.2 %). This looks too
    uniform for field data.

### S5. Kapatsila et al. 2024, breakdowns of battery-electric, hybrid and diesel buses in Portland (USA)

- **Reference.** Kapatsila, B.; Grisé, E.; Crumley, M.; El-Geneidy, A. (2024). Empirical analysis of
  battery-electric bus transit operations in Portland, OR, USA. *Transportation Research Part D* 128,
  104120. https://doi.org/10.1016/j.trd.2024.104120 (open access, CC BY; PDF from the authors' lab,
  tram.mcgill.ca/Research/Publications/Ebus_run.pdf).
- **Evidence basis.** Read in full (11 pages; pp. 10-11 are references).
- **Data.** TriMet (the Portland transit agency). 8 battery-electric 40-foot buses (New Flyer and
  Gillig) from one garage, on 4 urban routes, against 611 diesel and 2 hybrid buses (§3, Table 1).
  9 months, 5 Sep 2021 to 6 Jun 2022. Breakdowns are road calls as defined by the National Transit
  Database (NTD), from the agency's failure records (§3). Distances are planned miles, not odometer
  readings (§6).
- **Breakdown definition.** Excludes human-related events (passenger requests, vandalism),
  "non-integral parts" (fare system, wiper and washer) and body repair (§3, p. 5). A
  "maintenance" is an interruption of service of one day or more that is not a breakdown (§3).
- **Findings (Table 1, p. 5; §4.1).**

  | Measure | Diesel | Battery-electric | Hybrid |
  |---|---|---|---|
  | Planned miles from maintenance to breakdown, mean (SD) | 4630 (4618) | 2124 (2196) | 3512 (3625) |
  | Breakdowns per bus in 9 months | 4 | 5.6 | 7 |
  | Times out of service for maintenance (≥ 1 day) | 8.8 | 3.6 | 5.7 |
  | Breakdown records | 2414 | 39 | 14 |

  - The diesel-electric difference in miles to breakdown is significant (t = 6.9, p = 0) (§4.1).
  - The multilevel model (Table 3) says that, with other factors fixed, electric buses run
    464 miles more than diesel buses before a breakdown (90 % confidence, p < 0.1). Each extra ramp
    use per 100 stops cuts this distance by 205 miles, about 9 % (p. 9). Each extra 100 stops adds
    about 6 miles (Table 3). So the raw gap comes mostly from route and use differences, by the
    authors' reading (§4.3).
- **Components.** None. The paper gives no breakdown by component or failure cause. The authors'
  explanation of the ramp effect ("drains the battery") is a hypothesis, not a finding (p. 9).
- **Usage drivers named.** Ramp deployments, number of stops, average passenger load (-126 miles
  per extra passenger), daily temperature (-7.9 miles per °F, for all buses) (Table 3, §4.3).
- **Maintenance actions and intervals.** The authors recommend a preventive check every 2 weeks,
  or about 2300 miles (about 3700 km), for the electric buses (p. 9). This is a model-based
  recommendation, not the agency's practice.
- **Degradation data.** None.
- **Caveats (my reading).** Only 7 electric buses and 39 breakdowns enter the model (Table 1). The
  electric buses ran shorter routes with fewer stops (Table 1). The period follows the COVID-19
  pandemic, with low loads (§4.1).

### S6. Li et al. 2025, battery state of health of nine battery-electric buses in China, field data

- **Reference.** Li, H.; Zhuo, S.; Zhou, Y.; Bin Kaleem, M.; Jiang, Y.; Jiang, F. (2025). Robust SOH
  estimation for Li-ion battery packs of real-world electric buses with charging segments.
  *Scientific Reports* 15, 24871. https://doi.org/10.1038/s41598-025-09108-6 (open access).
- **Evidence basis.** Read in full (18 pages; pp. 16-18 are conclusion and references). Figures 6-8
  (capacity and SOH curves) read through the text that describes them, not rendered.
- **What it is.** A method paper on estimating the state of health (SOH, capacity relative to
  nominal) of the traction battery from charging data. It is not a maintenance study. It is used
  here only for its field degradation data.
- **Data.** Nine battery-electric buses of three types, three buses each, lithium iron phosphate
  (LFP) cells; China, city and operator not named (authors from Changsha). About four years,
  March or July 2020 to March-June 2024 (Table 2). Sampling every 10 s from the battery management
  system (BMS) (p. 7, p. 10). Pack types: LFP604 (large bus, 2 strings in parallel), LFP456 (medium
  bus), LFP280 (small bus, 1 string) (Table 1; pp. 12-14).
- **Degradation data (p. 9, Figures 7-8).**
  - LFP604 (most cycles, about 1340-1370 charge segments each, Table 2): SOH down to about 70 % by
    the end of the four years.
  - LFP456: about 80 % of the original capacity after four years; the three buses age differently.
  - LFP280 (fewest cycles, about 610-720 charge segments): above 90 % after three years.
  - Capacity is not monotone: it "regenerates" after rest periods, for example when fewer buses
    are in service (p. 9).
- **Usage drivers named.** Number of charge-discharge cycles and time per cycle; vehicle weight;
  pack layout (strings in parallel); temperature, season and driving habits (p. 9). These are the
  authors' explanations of the differences between bus types, not tested causes.
- **Maintenance actions.** None. The authors name SOH estimation as a basis for "timely
  maintenance" and battery reuse (abstract, p. 16), without intervals.
- **Caveats (my reading).** Three buses per type only. The cell counts in the text (p. 12, p. 13)
  swap the series numbers of Table 1 between LFP604 and LFP456. SOH labels come from
  ampere-hour counting over partial charges (50-100 % state of charge), smoothed over 30 cycles
  (pp. 5-7); initial SOH values above 100 % appear (Figure 8 caption).

### S7. Nan et al. 2022, fault records of battery-electric buses in Zhenjiang (China)

- **Reference.** Nan, J.; Deng, B.; Cao, W.; Hu, J.; Chang, Y.; Cai, Y.; Zhong, Z. (2022). Big
  Data-Based Early Fault Warning of Batteries Combining Short-Text Mining and Grey Correlation.
  *Energies* 15(15), 5333. https://doi.org/10.3390/en15155333 (open access, CC BY).
- **Evidence basis.** Read in full (20 pages; the text layer is fragmented, so page 9 with Table 1
  was rendered and read as an image).
- **What it is.** A method paper on early warning of one battery fault. Its first step counts
  fault types in the maintenance and service records of an electric-bus fleet. That count is the
  evidence used here.
- **Data.** A batch of battery-electric buses built in 2017 and operating in Zhenjiang, China;
  data of March 2021, with April 2021 for validation; the number of buses is not given (§3, p. 7;
  §4.3, p. 17). Sources: maintenance and repair documents filled in by service-station staff,
  alarm data, and on-board monitoring at 10 s (§3, §3.1-3.2). Test system of the bus maker Higer
  (§4.3).
- **Fault counts (§3.2, p. 9, Table 1).** 580,000 original fault records; 100,000 after word
  segmentation; 103 key fault items. Before counting, the authors removed "faults in non-pure
  electric systems" (p. 9). So mechanical items (tyres, brakes, doors, suspension) are **not** in
  the ranking. The table is titled "Top 10" but lists 9 rows (word frequency = number of
  mentions):

  | Fault (wording of the source) | Mentions |
  |---|---|
  | Poor consistency of battery cells (cell imbalance) | 12,506 |
  | Open circuit (high voltage) | 6979 |
  | Communication failure between BMS and charger | 5781 |
  | State of charge (SOC) value too high | 5194 |
  | Motor controller undervoltage 24 V | 4690 |
  | Over-temperature fault of charging base | 4522 |
  | Overcurrent (high voltage) | 3925 |
  | Heating or water-cooling relay failure | 3574 |
  | Battery cell voltage too high | 2502 |

  - My grouping of these rows: traction battery and its management (cell imbalance, SOC too high,
    cell voltage too high) 20,202 mentions; charging interface (BMS-charger communication, charging
    base temperature) 10,303; high-voltage circuit (open circuit, overcurrent) 10,904; auxiliary
    and low-voltage items (24 V undervoltage at the motor controller, heating or cooling relay)
    8264. "Open circuit" and "overcurrent" may also belong to the battery or the inverter; the
    source does not say.
- **Comparison with diesel.** None.
- **Usage drivers named.** For cell imbalance, the authors blame "the inconsistency of the
  production and use environment" (p. 9). Battery current and voltage have the highest grey
  correlation with the fault one hour later (Table 3: 0.90 and 0.71); vehicle speed, pedal
  travel and brake state about 0.47-0.49.
- **Maintenance actions.** Not described; the paper aims at a one-hour early warning (§4).
- **Degradation data.** None (alarm and fault events only).
- **Caveats (my reading).** Mentions in free text are not independent failures; one event can be
  written many times. The 52.5 % share of "battery, motor, and electronic control" failures in the
  introduction (p. 1, ref. 6) is about recalls of cars in China in 2021, not buses.

### S8. Taha (Hussein) 2022, PhD thesis with two papers on the drive of an in-wheel-motor electric bus

- **Reference.** Hussein, H. A. T. (2022). *Autonomous Machine Maintenance and Control in Digital
  Twin Environment.* PhD thesis, Polytechnique Montréal (advisors S. Yacout and Y. Shaban). Open
  access: https://publications.polymtl.ca/10367/ (PDF:
  https://publications.polymtl.ca/10367/1/2022_HusseinAdelTahaHussein.pdf). The author publishes as
  H. A. Taha. A thesis by articles. Two chapters treat a bus:
  - Chapter 8 (printed pp. 85-93) reprints Taha, H.A.; Sakr, A.H.; Yacout, S.; Serafin, P. (2021),
    *Failure Reasoning and Uncertainty Analysis for Wheel Motor Electric Bus*, IEEE ETFA 2021,
    https://doi.org/10.1109/etfa45728.2021.9613671 ("Published in", p. 85).
  - Chapter 9 (printed pp. 94-115) is the manuscript *Autonomous Uptime-Improvement for Electric Bus
    in Digital Twin Environment* (Taha, Yacout, Shaban), "Submitted to: IEEE Transactions on
    Intelligent Transportation Systems" (p. 94).
- **Evidence basis.** Read in part: chapters 8 and 9 in full (PDF pp. 102-132), the list of articles
  in chapter 2, Appendix C (Table C.1 read from the rendered page 145; Table C.2 through the text
  layer), and references [1]-[4] of the list. The other chapters (machining, aircraft engines)
  were not read. The IEEE version of the ETFA paper was not read; the thesis version is the
  author's text and may differ in layout.
- **Relation to Taha et al. 2023 (section 4).** Chapter 9 has the same scheme as the title of the
  2023 *Engineering Failure Analysis* paper: failure analysis linked to on-road driving, and
  autonomous actions. Whether the 2023 paper is the published form of chapter 9 is not established:
  the titles differ, and the 2023 paper was not read.
- **Data.** None from the field. The bus is a two-axle in-wheel-motor bus: one three-phase induction
  motor and one DC/AC inverter per driven wheel, one control unit (§8.2, §9.4). "Since operational
  failure data are unavailable, reliability prediction and maintenance planning are challenging
  tasks" (§8.1, p. 85). The failure rates come from handbooks and other studies: references [1]-[4]
  (a US Navy handbook of mechanical reliability, 1992; a fault-tree study of electric vans, Shu et
  al. 2019; a 2016 work on reliability prediction of electronic products; a 2011 study of embedded
  software in vehicles)
  (Table 8.1, p. 89).
- **Findings (model results, not measurements).**
  - Failure rates per 10⁶ hours (Table 8.1): embedded software 17.6; controller board (PCB) 3.06;
    inverter PCB 1.12; inverter IGBTs 0.085; optocoupler 0.243; motor bearing seal 0.4465; bearing
    0.083; rotor winding 0.277; stator winding 0.277.
  - Fuzzy fault tree, 10 % uncertainty, 10,000 working hours: the most critical part is the
    embedded software (importance 1.82), then the controller board (1.57), then the inverter board
    (about 1.55); among the motor parts, the bearings (§8.3-8.4, pp. 92-93).
  - Chapter 9: per-wheel inverter failure rate 1.54E-06 and bearing 1.74E-06 at 11,000 rpm (Table
    9.3, p. 108). The column is labelled "FPMh" (per 10⁶ hours), but the values read as per hour;
    the source is not consistent here.
  - The bearing failure rate grows linearly with the motor speed (Eq. 9.11, p. 111, from the Navy
    handbook). An expert from "an e-mobility maintenance center" set the data uncertainty at 17 %
    (p. 106). The bus is inspected every 600,000 miles, taken as 33 % system reliability (p. 112).
    Lowering the mean cycle speed from 15.11 to 9.63 miles per hour adds 8,603 miles before that
    limit (p. 115).
  - Proposed inspection intervals (Table C.1, p. 145, with literature references): inverter every
    2 years or 30,000 miles; motor bearings every 2 years or 22,000 miles; speed and temperature
    sensors every 3 years or 45,000 miles; control unit hardware yearly.
- **Use here.** It confirms that this group's bus work has no field data on inverters or motors.
  Its rates are handbook values, usable only as a prior. It does not change the ranking.

## 4. Could not read

- **Taha, H.A.; Yacout, S.; Shaban, Y. (2023).** Online failure analysis and autonomous risk
  control scheme for electric buses. *Engineering Failure Analysis*, December 2023, 107629.
  https://doi.org/10.1016/j.engfailanal.2023.107629. Reason: ScienceDirect, not open access; the
  brief rules out ScienceDirect because of its CAPTCHA (a CAPTCHA was met on 9 Nov 2026 for
  another ScienceDirect item). 10 Nov 2026: OpenAlex and Semantic Scholar list it as closed; its
  PolyPublie record (https://publications.polymtl.ca/56585/) holds only an external link, no file.
  Why worthwhile: a case study of the drive system of an electric bus (Polytechnique Montréal) that
  links driving conditions to failure analysis. Its likely precursor, chapter 9 of the author's
  thesis (S8), uses handbook failure rates, not field data; so the paper is now less likely to hold
  field data on the inverter and motor. Can you retrieve it?
- **Manzolli, J.A.; Trovão, J.P.; Antunes, C.H. (2022).** A review of electric bus vehicles
  research topics - Methods and trends. *Renewable and Sustainable Energy Reviews* 159, 112211.
  https://doi.org/10.1016/j.rser.2022.112211. Open access (CC BY-NC-ND), but the ScienceDirect
  PDF link answered with a CAPTCHA (9 Nov 2026; not passed). 10 Nov 2026: the article page
  https://www.sciencedirect.com/science/article/pii/S1364032122001344 answered in the browser with
  HTTP 403 and a Cloudflare challenge; not passed. OpenAlex reports no repository copy
  ("any_repository_has_fulltext": false); Semantic Scholar lists only the ScienceDirect PDF. Why
  worthwhile: a peer-reviewed map of e-bus research topics; it would show how much of the literature
  treats maintenance. Can you download it from ScienceDirect in your browser?
- **Wang, Z.; Liu, P.; Xin, T. (2011).** Study on a fault-analysis method for electric buses based
  on AHP. *Gaojishu Tongxin (High Technology Letters)* 21(11), 1196-1200.
  https://doi.org/10.3772/j.issn.1002-0470.2011.11.014. Reason: no full text found; probably in
  Chinese. Why worthwhile: weights faults by subsystem from the real fault statistics of one
  electric bus (Beijing Institute of Technology). Old (2011) and one bus only.

## 5. Leads not followed

### 5.1 Read in full and excluded

- **Malozyomov, B.V. et al. (2023).** Mathematical Analysis of the Reliability of Modern
  Trolleybuses and Electric Buses. *Mathematics* 11(15), 3260.
  https://doi.org/10.3390/math11153260. Read in full (25 pages). Excluded: the data are failures of
  the electrical equipment of conventional trolleybuses at Novosibirsk depot No. 2 in 2022 (§2,
  Table 1): circuit breaker, DC traction motor, contactors, starting resistors, current collectors.
  No battery-electric bus data. One unsourced claim (p. 3): maintenance and repair of trolleybuses
  and electric buses take "up to 75%" of operator funds, and the electrical complex about 60 % of
  maintenance cost. Source S1 (p. 13-14) cites this paper for more frequent but less
  labour-intensive repairs of electric buses, by "rapid replacement of entire structural modules".
  The paper does not say this. Its closest sentence (p. 6) says only that trolleybuses and
  electric buses see constant replacement of failed units and subsystems with new or
  reconditioned ones.
- **Ekici, Y.E. (2026).** A Decision Tree-Based Dynamic Maintenance Interval Model Incorporating
  Driver Behavior and Traffic Congestion: An Application to an Electric Bus Fleet. *Engineering
  Perspective* 6(3), 415-427. https://doi.org/10.64808/engineeringperspective.1904906. Read in full
  (13 pages). Excluded: the vehicle is a 24.7 m dual-mode trolleybus in Malatya, Türkiye (§4.1);
  failures are only "electrical" or "mechanical", with no component split (§3.3); the reported
  metrics are not consistent (MAE larger than RMSE for the decision tree, Table 4, which cannot
  happen); power and energy units are mixed (energy consumption in kW, Table 5).
- **TCO (total cost of ownership) model for Kielce** (*Eksploatacja i Niezawodność* 24(1), 2022;
  downloads\lit\05). Excluded by an earlier agent as a cost model without field maintenance data.
- **Ibrahim, H.; Ali, A.M.; Attia, T. (2025).** Optimizing electric bus performance via
  predictive maintenance: a combined experimental and modeling approach. *Frontiers in Future
  Transportation* 5, 1506866. https://doi.org/10.3389/ffutr.2024.1506866. Read in full (13 pages).
  Excluded: one bus (345 kWh) on two test drives in Egypt (67.8 km; a 3 km WLTP-low run), §2-§5.1.
  Battery ageing comes from a semi-empirical model taken from the literature (§4.2), not from
  measurement. Its headline (battery life 6.69 instead of 5.35 years, +25 %, with gentler
  driving; §5.2) is a model output. No maintenance records.
- **Potkány, M.; Hlatká, M.; Debnár, M.; Hanzl, J. (2018).** Comparison of the Lifecycle Cost
  Structure of Electric and Diesel Buses. *Naše more* 65(4), 270-275.
  https://doi.org/10.17818/NM/2018/4SI.20. Read in full (6 pages). Excluded: a life-cycle cost
  model for one Bratislava line with costs set by the authors (Table 1, "Source: authors"); for
  example, tyre cost 3300 €/year for both bus types.

### 5.2 Grey literature seen in passing (for the other agent)

- NREL (US National Renewable Energy Laboratory) and FTA (US Federal Transit Administration)
  fleet evaluations of battery-electric buses, with road calls by system and miles between road
  calls: Foothill Transit progress reports (https://docs.nrel.gov/docs/fy21osti/76259.pdf;
  https://afdc.energy.gov/files/u/publication/foothill_transit_beb_demo_results_2nd_rpt.pdf),
  County Connection
  (https://afdc.energy.gov/files/u/publication/zero-emission_evaluation_county_connection_bec.pdf),
  King County Metro (cost per mile, cited by the US DOT ITS knowledge base,
  https://www.itskrs.its.dot.gov/2019-b01361): not read. FTA Report No. 0253, "Procuring and
  Maintaining Battery Electric Buses and Charging Systems" (2023): read on 10 Nov 2026, entered as
  R14 in case_study/maintenance_focus_reports.md and B12 in case_study/component_battery.md.
- University of Washington, "Statistical Failure Analysis of Hybrid Bus Batteries: Using A
  Three-Year Maintenance Dataset from King County" (thesis, hybrid buses):
  https://digital.lib.washington.edu/researchworks/items/a7e57ae0-03d8-44e2-b0ec-43e048428d34.
  Not read.
- Survey and interview study on tyre wear of electric vehicles, with 28 professional users,
  among them bus companies (probably a VTI, Swedish National Road and Transport Research
  Institute, report; not in Crossref): https://core.ac.uk/works/136216984. A search snippet says
  8 of 11 bus companies saw faster tyre wear than on diesel buses; this snippet is not evidence.
  Not read.
- WRI India working paper, "Real-world Electric Bus Operation: Trend in Technology, Performance,
  Degradation, and Lifespan of Batteries":
  https://wri-india.org/sites/default/files/Real-world%20Electric%20Bus%20Operation_Working%20Paper%20final%20_revised.pdf.
  Not read.
- INL (Idaho National Laboratory), "EV-EVSE Fault Study" (2025):
  https://inl.gov/content/uploads/2023/07/EV-EVSE-Fault-Analysis-Report-Final-8-27-25-1.pdf.
  Not read.
- More FTA and NREL evaluations with maintenance cost per mile by system: King County Metro,
  FTA Report No. 0118
  (https://www.transit.dot.gov/sites/fta.dot.gov/files/docs/research-innovation/115086/zero-emission-bus-evaluation-results-king-county-metro-battery-electric-buses-fta-report-no-0118.pdf);
  FTA Report No. 0163
  (https://www.transit.dot.gov/sites/fta.dot.gov/files/2020-05/FTA-Report-No.-0163.pdf); Duluth
  Transit Authority BEB evaluation, with unavailability by cause
  (https://afdc.energy.gov/files/u/publication/duluth_transit_authority_beb_evaluation.pdf).
  Not read.
- OC Transpo (Ottawa) e-bus pilot, engineering findings, with mean distance between failures:
  https://bulldogottawa.com/wp-content/uploads/2023/05/Engineering-Findings-E-Bus-Performance.pdf.
  Not read.
- EDDA Bus project (Leipzig), operation page; a search snippet says that failures came mostly
  from door control, heating and the air compressor, not the electric drive:
  https://www.edda-bus.de/en/Operation.html. The snippet is not evidence. Not read.
- CARB (California Air Resources Board), Appendix G, literature review on transit bus maintenance
  cost:
  https://ww2.arb.ca.gov/sites/default/files/2020-06/Appendix%20G%20Literature%20Review%20on%20Transit%20Bus%20Maintenance%20Cost.pdf;
  CTE (Center for Transportation and the Environment), transit bus life-cycle maintenance cost
  project: https://cte.tv/tvidc/maintenance-cost-comparison. Not read.
- A National Academies (TCRP) chapter that lists propulsion-related cost categories of
  zero-emission buses: https://www.nap.edu/read/25842/chapter/11. Not read.

### 5.3 Peer-reviewed or preprint leads not followed

- Ercevik, A.I.; Ozbayoglu, A.M. (2025). Artificial Intelligence Based Predictive Maintenance for
  Electric Buses. arXiv:2510.23879 (preprint, no journal reference on 9 Nov 2026). Alarm
  prediction from two years of CAN (controller area network) bus data. Not peer-reviewed; not read.
- Liu et al. (2024). Reliability evaluation of IGBT power module on electric vehicle using big
  data. *Journal of Semiconductors* 45(5), 052301. https://doi.org/10.1088/1674-4926/45/5/052301.
  IGBT (insulated-gate bipolar transistor) lifetime from vehicle data and an aging test; the
  Crossref abstract does not say the vehicles are buses. Not read.
- Taha, H.A.; Mammeri, A.; Yacout, S.: later works on powertrain fault diagnosis (IEEE WF-IoT 2024)
  and on battery degradation monitoring of a 40-foot electric bus (*Neural Computing and
  Applications*); PolyPublie records https://publications.polymtl.ca/62731/ and
  https://publications.polymtl.ca/80455/. Not read.
- Agredano, F.; Taha, H.A.; Yacout, S. (2023). Maintenance 4.0 opportunistic strategy for electric
  buses. CIGI Qualita MOSIM 2023, Trois-Rivières (5 pages). Open PDF (seen in a search list, 10 Nov
  2026): https://collection-numerique.uqtr.ca/2064/1/CIGI_QUALITA_MOSIM_2023_paper_4719.pdf. Not
  read.
- Alkassas, O.; Taha, H.A.; Le Ny, J.; Yacout, S. (2026). Logical analysis of survival data for
  remaining driving range prediction in battery electric bus operations. *Computers & Industrial
  Engineering* 220, 112264 (PolyPublie search list, 10 Nov 2026). Range, not maintenance. Not read.
- Połom, M.; Wiśniewski, P. (2021). Implementing Electromobility in Public Transport in Poland in
  1990-2020. *Sustainability* 13(7), 4009. https://doi.org/10.3390/su13074009. Policy review; the
  Crossref abstract shows no maintenance content. Not read.
- Meishner, F.; Sauer, D.U. (2020). Technical and economic comparison of different electric bus
  concepts based on actual demonstrations in European cities. *IET Electrical Systems in
  Transportation*. https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-est.2019.0014.
  A search snippet says it assumes equal maintenance cost for diesel and electric buses. Not read.
- "State of Health Estimation of a Lithium-Ion Battery Used in a Trolleybus Under Real Operating
  Conditions" (a search snippet says about 6.5 years of a 200 kWh LFP pack; authors and venue not
  checked):
  https://www.researchgate.net/publication/408372805. A second field battery-degradation source
  for an in-motion-charging vehicle. Not read.
- Sensors (2022), A Data-Driven Approach to State of Health Estimation and Prediction for a
  Lithium-Ion Battery Pack of Electric Buses Based on Real-World Data:
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9370969/. Field battery degradation of buses. Not
  read; S6 already covers this kind of evidence.
- Battery aging from four years of maintenance data of a large hybrid-electric bus fleet, ECS
  Meeting Abstracts (2026), https://doi.org/10.1149/ma2026-015857mtgabs. Conference abstract,
  hybrid buses. Not read.
- "Predictive Maintenance Framework for Electric Bus Braking Systems Based on Regenerative Braking
  Data Analytics", *The American Journal of Engineering and Technology* (2025),
  https://doi.org/10.37547/tajet/volume07issue12-03. The journal's review standard is unknown.
  Not read.
- Chalmers (2026), DC-link cooling of a 6-phase 500 kW heavy-duty traction inverter with respect
  to lifetime: https://research.chalmers.se/en/publication/553311. Design study, not field data,
  not buses. Not read.
