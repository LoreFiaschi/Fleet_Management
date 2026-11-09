# Maintenance focus of battery-electric buses: evidence from grey literature

Status: done (9 Nov 2026). Eleven sources entered: R1 (NREL Foothill 2021), R2 (VBZ 2019), R3 (TTC
Toronto 2022), R4 (PwC/BMWK 2024), R5 (ESöV P-196, Graubünden 2020), R6 (NREL Long Beach 2020), R7 (AC
Transit 2026), R8 (World Bank Shenzhen 2021), R9 (NREL Duluth 2022), R10 (NHTSA recall 23V-697, 2023),
R11 (ThEGA Thuringia 2025). Summary table, ranking, "Could not read" and "Leads not followed" filled. If
work resumes: the first leads are AC Transit ZETBTA volume 3 and TCRP Synthesis 130 (section 4). Notes:
www.nrel.gov no longer resolves, use docs.nlr.gov; pwc.de and thega.de refuse scripts, the Wayback
Machine serves both PDFs. Scratchpad downloads used: rep\01-19 (rep\04, 05, 11 hold only HTML or
"Access Denied" pages).

Scope: reports of agencies, laboratories, operators, transport associations, consultancies and funded
projects. Peer-reviewed papers are only leads here. Abbreviations: BEB = battery-electric bus;
NREL = US National Renewable Energy Laboratory; HVAC = heating, ventilation and air conditioning;
IGBT = insulated-gate bipolar transistor; SiC = silicon carbide; ESS = energy storage system (the
traction battery); MBRC = miles between road calls (in-service failures); CNG = compressed natural gas;
PMI = preventive maintenance inspection; SOC = state of charge; HV = high voltage. "Planned days" = days
a bus was planned for service. Shares "of cost" are shares of the maintenance cost per mile or km
without warranty work, unless stated.

## 1. Summary table

Caveat for every row: in all fleets with cost data, the battery and drive were under warranty, and
warranty work is excluded (R1, R3, R6, R7, R8, R9). So the shares understate the traction battery and
the electric drive.

| Component | Weight in BEB maintenance (source, location) | Usage driver | Maintenance actions and interval | Sources |
|---|---|---|---|---|
| Body, doors, interior, HVAC and other "common bus systems" | The largest share everywhere. Body interior and exterior 50 % of all defects (R3 p. 34); cab, body, accessories 52 % of cost against 28 % CNG (R6 Table 10 p. 29), 15-18 % (R1 Tables 14, 27); general bus issues 51 % of unavailable days (R1 Table 5), 17.4 % (R6 Table 4) and 29.7 % (R9 Table 5) of planned days; "other defects" (air conditioning, doors, signs) 50 % of unavailable days (R4 Fig. 110 p. 218); common systems 46 % of road calls (R7 Fig. 23) | Not quantified. Named: cold (heater faults, R9), road salt (R3, R9) | Corrective repair; long downtime from parts lead times (R2 p. 30; R3 p. 19; R9 pp. 31-32) | R1, R2, R3, R4, R6, R7, R9 |
| Traction battery (ESS) | Small in cost and in-service failures under warranty: 1.6-2.2 % of cost (R1 Tables 15, 28); ESS MBRC 70,656-196,898 (R1 Table 11; R6 Table 8; R9 Table 7). Downtime 15 % of unavailable days (R1 Table 5), 0.2 % and 5.3 % of planned days (R6, R9). Exceptions: HV system 49 % of road calls of one fleet, and battery lead times the main downtime cause (R7 pp. 22-23); a battery generation replaced fleet-wide (R4 p. 124); 11 packs replaced (R3 p. 36) | Depth of discharge, charge power, temperature (R2 pp. 33-34); cycle count, depth, power, calendar age, temperature (R5 p. 25); low SOC (R9 p. 20); driving behaviour and route (R8 p. 84); cold raises the ESS share of downtime (R9 pp. 25-26) | Replacement at end of life, 80 % (R5 p. 26; R8 p. 60) or 70 % capacity (R6 p. 3; R11 p. 9). Planned life 6-8 years (R5 p. 26); one replacement in 12 years assumed (R4 p. 238); observed 2-4.5 years, early Shenzhen fleet (R8 p. 60); expected 8-12 years, 0.6-1 million km (R11 pp. 4, 9); partial cell replacement (R8 p. 60) | R1-R9, R11 |
| Electric drive: motor, gearbox, axle, controls | Electric drive 13 % and 5 % of cost, 18 % of unavailable days (R1 Tables 5, 15, 28); "Defekt im Antriebsstrang" 29 % of unavailable days (R4 Fig. 110); 9.1 % of planned days (R9 Table 5); 2.1 % (R6 Table 4). Failures of one supplier's electric axles (R4 p. 124) and of rear axle reduction gears (R6 p. 35; R3 p. 19). Propulsion below 4 % of all defects (R3 p. 34) | Not quantified | Replacement under warranty; motor about 14,600 USD (R1 p. 38) | R1, R3, R4, R6, R7, R9 |
| Traction inverter (IGBT or SiC modules) | Thin. 10 % and 3 % of propulsion defects, about 0.1-0.3 % of all defects (R3 Charts 6, 8); named inside the electric-drive downtime of 9.1 % (R9 p. 15); safety recall of 238 buses for an overheating inverter transistor (R10). No cost share anywhere. IGBT or SiC: not stated in any source | Overheating of a transistor (R10); no usage driver given | Inverter replacement (R10) | R3, R9, R10 |
| DC-link capacitors | **No evidence for the vehicle in any source.** Only a depot charger failed through an overheated capacitor (R9 pp. 31-32) | - | Charger maker replaced the capacitors of all units (R9 p. 32) | R9 (charger only) |
| Tyres | 14.6 % and 17.4 % of BEB cost against 9.3-10.4 % for CNG (R1 Tables 14, 27); tyre cost +38.8 % (R8 p. 83) or about +30 % (R8 p. 58) against diesel; 0 % in R6 (no reason given) | Vehicle weight (R5 pp. 23-24; R8 pp. 57, 83); in-wheel motors in propulsion and braking (R8 p. 83); road condition, potholes and broken curbs (R1 p. 38) | Replacement: 5,920 and 6,116 miles per tyre against about 10,700 for CNG (R1 pp. 38, 63); 125 against 90 CNY per 1,000 km (R8 p. 83) | R1, R5, R6, R8 |
| Friction brakes | Lower than conventional: 1.1-2.4 % against 5.4-5.7 % of cost (R1 Tables 14, 27); 4 % against 2 % (R6 Table 10); a frequent diesel failure, not BEB (R7 p. 23) | Regenerative braking share; regeneration high on hills (R9 p. 18), off on snow (R9 pp. 26, 32); road-salt corrosion (R9 pp. 32-33) | Brake service interval doubled (R11 p. 12); splash guards against corrosion (R9 p. 33) | R1, R2, R3, R6, R7, R9, R11 |
| Low-voltage (24 V) batteries and electrical system | About 10 replacements per 100,000 miles against 0.9 for CNG (R1 pp. 37-38); 81 % of BEB propulsion cost (R6 p. 31); replaced more often (R9 p. 31); "Electrical system problem" a most common BEB failure (R7 p. 24) | Parasitic loads while parked (R1 p. 37; R6 p. 31) | Replacement; charging as routine; protectors retrofitted (R9 p. 31) | R1, R2, R6, R7, R9 |
| HVAC and heating | HVAC 4-10 % of defects (R3 pp. 34-37); 2-4 % of cost (R1; R6); compressor defects (R8 p. 58); heater issues (R9 pp. 15, 32) | Ambient temperature (energy effect: R9 p. 25; R11 p. 10) | Corrective | R1, R3, R4, R6, R8, R9 |
| Charging interface and chargers | Charging 6 % of unavailable days and rooftop-contactor fires (R1 Table 5, p. 38); 8 % of unavailable days (R4 Fig. 110); 1.2 % and 2.6 % of planned days (R6, R9); 1.4-6 % of defects (R3 pp. 34-37); faulty plug pin (R2 p. 28) | Misalignment, loose bolts, arcing (R1 p. 38); cables run over (R3 p. 80); soft asphalt (R4 p. 358) | Charger inspections monthly, 6-monthly, yearly (R1 p. 9) | R1, R2, R3, R4, R6, R7, R9 |
| HV wiring, connectors, corrosion | Main cause of most propulsion faults (R3 pp. 34, 37); connector corrosion (R9 pp. 32-33) | Water intrusion, road salt (R3, R9) | Covers, splash guards (R9 p. 33) | R3, R9 |
| Preventive maintenance (inspection labour) | PMI 14-18 % (R1) and 23 % (R6) of cost; scheduled maintenance 8 % of unavailable days (R4) | Distance | Level I every 4,000-5,000 km, level II every 20,000 km (R8 p. 57) | R1, R4, R6, R8 |
| Whole bus against diesel or CNG | Maintenance cost per km: 75 % of diesel (assumed, R4 p. 377); < 0.20 against > 0.30 EUR/km (R11 p. 11); 17 % of diesel over years 1-4, the diesel overhaul included (R8 p. 83; Table 6-7 p. 84); lower than CNG (R6), higher or about equal (R1); 0.76-3.56 against 1.01 USD/mile (R7 Fig. 18). Availability lower than diesel in all fleets with a baseline (R1, R3, R4, R6, R7, R9) | - | - | R1, R4, R6, R7, R8, R11 |

## 2. Ranking of the components by the evidence

The ranking orders the components by how strongly the reports show that they drive BEB maintenance
cost, failures or downtime. It is not a choice of components for the model.

1. **Body, doors, interior, HVAC ("common bus systems").** The largest share of defects, cost and
   downtime in every fleet with data (R1, R3, R4, R6, R7, R9); not specific to the electric drive.
2. **Traction battery.** Small in yearly cost and road calls while under warranty, but the largest single
   life item: every cost study plans a replacement or a 70-80 % end of life (R4, R5, R8, R11), one fleet
   saw replacements after 2-4.5 years (R8), and battery faults or lead times are a main downtime cause in
   some fleets (R4, one operator; R7). The usage drivers are named by the most sources (R2, R4, R5, R8,
   R9). Measured fade curves: none.
3. **Tyres.** Higher wear than conventional buses in the two sources with numbers (R1: 43-45 % fewer
   miles per tyre, computed here; R8: +30-39 % cost), with a named driver (weight, R5, R8).
4. **Electric drive (motor, gearbox, axles).** Large downtime shares in early fleets (R1, R4, R9) and
   repeated axle and gear failures (R3, R4, R6); in R6 its share of propulsion cost halved from 2018 to
   2019 (p. 31).
5. **Low-voltage batteries.** A repeated, specific BEB problem in US fleets (R1, R6, R9), with one rate
   (R1).
6. **Charging interface and chargers.** A steady 1-8 % of downtime (R1, R4, R6, R9), with rare but severe
   events (R1, R9).
7. **Friction brakes.** Consistently less work than on diesel or CNG buses (R1, R11); relevant only where
   regeneration is lost (snow, R9) or corrosion acts (R9).
8. **Traction inverter.** Thin: small shares of propulsion defects (R3), named in one downtime category
   (R9), one safety recall (R10). No cost share, no usage driver, no interval.
9. **DC-link capacitors.** Absent for the vehicle. Only a charger capacitor failure (R9).

**Where evidence is thin or absent.** Inverter and DC-link capacitors: the eleven sources were searched
for "inverter", "IGBT", "SiC", "silicon carbide", "power electronics", "Wechselrichter", "Umrichter",
"Leistungselektronik", "capacitor", "Kondensator", "DC-link", "Zwischenkreis" (text of R4, R5, R6, R7,
R8, R9, R11 by script; R1-R3 when read in full). Web searches for fleet reports on inverter or IGBT
failures found only the recall R10 and the Duluth report R9 (9 Nov 2026). No source states whether the
inverters are IGBT or SiC. Degradation or wear data in any source: tyre life (R1, R8), battery
replacement ages and cycle counts (R8, R11), brake interval (R11); no capacity curves. Swiss, Nordic and
Dutch maintenance reports with fleet data: none found (searches on VBZ, PostAuto, BERNMOBIL, Winterthur,
BAV, BFE, VTT, HSL, TØI, TNO, CROW; 9 Nov 2026).

## 3. Source entries

### R1. NREL, Foothill Transit final report (2021)

- **Reference.** Jeffers, M.; Eudy, L. (2021). *Foothill Transit Battery Electric Bus Evaluation: Final
  Report.* National Renewable Energy Laboratory, NREL/TP-5400-80022, June 2021. Funded by the California
  Air Resources Board (CARB). URL: https://www.nrel.gov/docs/fy21osti/80022.pdf (NREL's domain now
  redirects to https://docs.nlr.gov/docs/fy21osti/80022.pdf; downloaded 9 Nov 2026).
- **Evidence basis.** Read in full (99 PDF pages, text layer; Tables 14 and 15 also read from the rendered
  page, because the text layer jumbles them).
- **Fleets.** Pomona depot: 12 Proterra BE35 fast-charge BEBs (lithium-titanate battery, 88 kWh; data
  4/2014-12/2020, 1.77 million miles) and 2 Proterra Catalyst fast-charge BEBs; baseline 8 CNG
  (compressed natural gas) buses. Arcadia depot: 14 Proterra Catalyst E2 depot-charged BEBs (440 kWh;
  1 year, 2020, still under warranty); baseline 14 CNG buses. Table 2, Table 3 (pp. 4, 6).
- **Method that matters for weights.** Work-order costs, labour at a fixed 50 USD per hour; warranty and
  accident costs removed (Section 4.6, p. 27-28; Appendix B, p. 70). So any battery or drive work done
  under warranty is **not** in the cost shares. Systems coded by ATA VMRS (American Trucking Associations
  Vehicle Maintenance Reporting Standards).

Findings.

| Item | BEB 35FC (Pomona) | BEB 40E2 (Arcadia) | CNG baseline | Location |
|---|---|---|---|---|
| Total maintenance cost, USD/mile | 0.500 | 0.364 | 0.320 (Pomona); 0.354 (Arcadia) | Table 14 p. 34; Table 27 p. 60 |
| Propulsion-related share of cost | 36.4 % | 27.8 % | 41.5 %; 34.0 % | same |
| Tires share of cost | 14.6 % (rank 3) | 17.4 % | 9.3 %; 10.4 % | same |
| Cab, body and accessories share | 15.4 % (rank 2) | 17.7 % | 16.1 %; 12.9 % | same |
| PMI (inspection labour) share | 14.2 % | 18.4 % | 16.2 %; 18.4 % | same |
| Brakes share | 2.4 % | 1.1 % | 5.7 %; 5.4 % | same |
| HVAC share | 3.2 % | 3.8 % | 3.0 %; 4.3 % | same |
| Axles, wheels, drive shaft share | 7.3 % | 1.1 % | 1.2 %; 0.4 % | same |
| Traction battery ("power plant", VMRS 45), USD/mile | 0.008 (1.6 %) | 0.008 (2.2 %) | engine 0.040; 0.032 | Table 15 p. 36; Table 28 p. 62 |
| Electric drive and motor (VMRS 46), USD/mile | 0.066 (13 %) | 0.019 (5 %) | 0 | same |
| Non-lighting electrical (low-voltage batteries, charging, cranking), USD/mile | 0.085 (17 %) | 0.037 (10 %) | 0.037; 0.035 | same |

- **Downtime (availability).** Pomona BEB 35FC: 3,800 unavailable days out of 19,619 planned. Of these,
  general bus maintenance 1,951 (51 %), electric drive 677 (18 %), ESS (energy storage system, the traction
  battery) 568 (15 %), transmission 313 (8 %), charging 241 (6 %). Table 5, p. 16 (shares of downtime
  computed here from the table). Arcadia BEB 40E2: 619 unavailable days; general bus 423, transmission
  109, electric drive 48, ESS 3. Table 18, p. 43.
- **Roadcalls (in-service failures).** BEB 35FC: 312 roadcalls, of which 132 propulsion-related and 9
  ESS-related; ESS MBRC (miles between roadcalls) 196,898. CNG: 110 roadcalls, 73 propulsion. Table 11,
  p. 27. Arcadia BEB 40E2: no ESS roadcall in 2020. Section 5.5, p. 55.
- **Traction battery.** Cost and roadcall weight small out of warranty (above). The report advises to
  "consider battery degradation over time" when planning service (Section 6.1, p. 65). No capacity-fade
  data. One BEB was lost to a "thermal event" in January 2020, cause under investigation (p. 3).
- **Inverter, DC-link capacitors.** Not named. The electric-drive cost comes from "expensive parts and
  higher labor hours for troubleshooting and repair on components such as drive motors and DC-DC
  converters" (Section 4.6.4, p. 36). Part prices: DC-DC converter over 12,600 USD, traction motor about
  14,600 USD (Section 4.6.5, p. 38). One DC-DC converter failed on the E2 fleet (p. 63).
- **Tires.** Higher wear on BEBs. Pomona: 292 tires replaced on BEBs; 5,920 miles per tire, 0.07 USD/mile,
  against 10,796 miles per tire and 0.03 USD/mile for CNG (Section 4.6.5, p. 38). Arcadia: 6,116 against
  10,640 miles per tire; tires 17 % of BEB cost against 10 % for CNG (Section 5.6.5, p. 63). Usage driver
  named by the report: local routes with "potholes and broken curbs" against mixed local and highway
  service; the report does not attribute the wear to vehicle weight or torque.
- **Low-voltage (24 V) batteries.** Premature failure; 178 replacements since January 2015, about 10 per
  100,000 miles, against 0.9 for CNG; cause: accessories drain the batteries when the master switch is
  left on (Section 4.6.5, pp. 37-38). Arcadia: 4.1 against 1.4 per 100,000 miles (p. 63).
- **Charging interface.** Three thermal events (late 2019, 2020) at the rooftop contactors of the
  overhead fast charger: loose bolts, misalignment, arcing, fire. The fast-charge fleet was parked from
  October 2020 to January 2021 (Section 4.6.5, p. 38; Section 6, pp. 64-65). Charger maintenance
  contract: monthly, 6-month and annual inspections (Section 3.1.1, p. 9).
- **Other.** Coolant pumps cracked and air compressors leaked on the E2 fleet, replaced under warranty
  (Section 5.6.5, p. 63).
- **Maintenance actions and intervals.** Minor PMI with safety inspections and filter changes, by
  mileage; no mileage values given (Section 5.6.2, p. 58). Expected bus life about 12 years ("after 6
  years ... half its expected life", p. ix).
- **Comparison with conventional buses.** CNG, not diesel. BEB scheduled cost per mile lower, unscheduled
  higher; labour 52-67 % of BEB cost against 46-57 % for CNG (Sections 4.6.2, 5.6.2).
- **Degradation or wear data.** Tire life in miles per tire (yes). No battery capacity data.
- **Scope note.** One fleet of one manufacturer (Proterra); the report warns that results "are not
  necessarily representative of a specific technology" (p. 1).

### R2. VBZ, findings report on two battery-bus trials in Zurich (2019)

- **Reference.** Verkehrsbetriebe Zürich (VBZ) (2019). *Elektrisch in die Zukunft: Batteriebusse im Test
  bei den VBZ* (Electric into the future: battery buses on trial at VBZ), "Erkenntnisbericht" (findings
  report), April 2019. Authors named: C. Böckmann, D. Sorg, A. Vogel, O. Obergfell. URL:
  https://vbzonline.ch/content/uploads/sites/3/2019/04/VBZ_eBus_Erkenntnisbericht_Web.pdf
- **Evidence basis.** Read in full (25 PDF pages, printed as spreads, report pp. 1-47). Page numbers
  below are the printed ones.
- **Fleet.** Two rented single buses. "Quartier-eBus" (neighbourhood bus) SOR EBN 8: 8 m, 172 kWh
  lithium-ion, asynchronous motors 120 kW; October 2016 to August 2018, over 60,000 km, depot charging
  only. "Standard-eBus" Caetano e.City Gold: 12 m, 150 kWh, permanent-magnet synchronous motor 160 kW;
  May to November 2018. Both with a diesel-fuelled auxiliary heater (pp. 16-17, 29).
- **No cost or failure statistics.** The report says the trials were too short for valid statements on
  maintenance effort and cost (p. 42). All findings below are qualitative.

Findings.

- **Drive and traction battery: few failures.** "Bei den Probefahrzeugen traten insgesamt nur wenige
  Störungen des Elektroantriebs und des Batteriesystems auf" (the trial buses had only few failures of the
  electric drive and the battery system) (summary, p. 9). After commissioning, failures of drive, battery
  and charging settled "auf einem sehr tiefen Niveau" (at a very low level), comparable to drive failures
  of diesel buses (p. 30).
- **What failed instead: components independent of the drive.** Doors in particular (p. 9). Repairs of
  destination displays, door system, heating and starter (low-voltage) battery took longer than on
  fleet buses, because spare parts were missing for single vehicles (p. 30). Causes were partly
  organisational: rental contract, slow after-sales of foreign manufacturers (p. 30).
- **Charging interface.** A defective pin in the special charging plug caused a long outage; a
  replacement plug took long to obtain (quote of C. Böckmann, p. 28). Software problems between charger
  and vehicle delayed the start of both trials (p. 28). The summary calls the vehicle-charger interface
  "kritisch" (critical) and asks for a robust, standard, "wartungsarm" (low-maintenance) charging system
  (p. 9).
- **Traction battery: usage drivers named.** Depth of discharge ("Lade- bzw. Entladehub") has an
  "überproportional negativ" (disproportionately negative) effect on battery ageing; charging power and
  operating temperature also matter (pp. 33-34). The battery management system and the reserve between
  usable and installed capacity strongly affect ageing (p. 34). Figure on p. 34: of the installed
  capacity, usable traction energy 50-60 %, auxiliaries 10-15 %, residual buffer 10-15 %, degradation
  reserve 20 %. VBZ names a contractual minimum battery warranty period as the operator's lever (p. 34);
  no warranty length is given.
- **Expected maintenance changes against diesel (manufacturer statements, not VBZ data).** Lower
  maintenance and repair cost because of: simpler drivetrain; no engine, gearbox or exhaust
  after-treatment; less brake wear thanks to regenerative braking; lower temperature extremes in the
  drivetrain; fewer vibrations; longer life of electronic devices (p. 42). The report marks these as
  "Gemäss Aussagen von Busherstellern" (according to bus manufacturers).
- **Workshop.** Components move to the roof; VBZ already has roof work stands from trolleybuses (p. 42).
- **Usage data relevant to a Zurich case study (energy, not wear).** Neighbourhood bus: about 1 kWh/km,
  of which drive 0.85 and auxiliaries 0.14 kWh/km (p. 36). Line 39 uphill section over 1.75 kWh/km,
  downhill near zero because of regeneration (p. 37). Heating in February 2018 almost 0.8 kWh/km
  (p. 39). Standard bus below 1 kWh/km on all lines; average gradient 1.2 % (line 66) against 3.8 %
  (line 37) (p. 35).
- **Degradation or wear data.** None.

### R3. TTC, final report of the head-to-head battery-bus evaluation (Toronto, 2022)

- **Reference.** Toronto Transit Commission (TTC) (2022). *TTC's Green Bus Program: Final Results of TTC's
  Head-to-Head eBus Evaluation.* Report to the TTC Board, 14 April 2022, 102 pages (report body pp. 1-25;
  Appendix A, technical results, pp. 26-100; Appendices B, C). URL:
  https://www.ttc.ca/-/media/Project/TTC/DevProto/Documents/Home/Public-Meetings/Board/2022/April-14/Reports/10_TTCs_Green_Bus_Program_Final_Results_of_TTCs_Head_to_Head_eBus_Evalua.pdf
  (downloaded 9 Nov 2026).
- **Evidence basis.** Read in full (102 pages, text layer; Charts 3, 4, 6, 8 also read from the rendered
  pages 33, 34, 36, 37, because the text layer does not carry them).
- **Fleet.** 60 BEBs (12 m class), depot charging, in service 1 to 2.5 years, 2.5 million km by
  31 Dec 2021: 10 BYD K9M (360 kWh), 25 New Flyer XE40 (NFI, 400 kWh), 25 Proterra Catalyst E2 (440 kWh).
  Baseline: Nova hybrid-electric buses (HEV) of similar age, not diesel. Toronto climate, road salt
  (Table 1, p. 14; pp. 38-39).
- **Warranty caveat.** All buses had a two-year "bumper-to-bumper" warranty (full-vehicle warranty); the
  vendors still did most repairs on propulsion and charging (p. 93). "Complete maintenance costs are
  unavailable" (p. 95).

Findings.

| Item | BYD | New Flyer | Proterra | Nova HEV | Location |
|---|---|---|---|---|---|
| Mean distance between failures (MDBF), km; target 30,000 | 35,000 | 70,000 | 25,000 | > 70,000 | p. 18, p. 29 |
| Distance between repairs (all minor defects), km | 323 | 527 | 375 | 811 | p. 30 |
| Fleet availability (target 80 %) | 30 % | 95 % | 95 % | > 95 % | p. 19, p. 32 |
| Average repair time (Apr 2022) | 9.03 | 3.79 | 7.49 | 1.84 | p. 31 says hours; Chart 58, p. 94, says days |
| Propulsion share of all defects | 1.6 % | 2.8 % | 3.3 % | - | Chart 3, p. 33; pp. 34-37 |
| Charging share of all defects | 2 % | 6 % | 1.4 % | - | pp. 34-37 |
| Repair and maintenance cost 2021, CAD/km (mostly under warranty) | 0.046 | 0.053 | 0.054 | 0.133 | Table 20, p. 97 |

- **Where the defects are (work-order analysis of all repair work orders, to 31 Jan 2022).** Body
  interior and exterior make 50 % of all faults. These are "common bus systems", not specific to the
  electric drive. The propulsion system is below 4 % of defects (p. 34). Top five systems per fleet
  (pp. 34-37):
  - BYD: body interior 34 %, body exterior 12 %, doors 8 %, driveline 7 % (water in the axle), HVAC 4 %.
  - New Flyer: body exterior 22 %, body interior 20 %, HVAC 10 %, doors 6 %, driveline 6 %.
  - Proterra: body exterior 25 % (cracks in the composite body), body interior 22 %, doors 8 %, HVAC 5 %,
    driveline 5 %.
  - Chart 3 (p. 33) also shows the air system (about 2-7 %) and the wheelchair ramp (about 2-5 %); values
    read from the bar chart.
- **Traction battery (ESS).** Inside propulsion defects: ESS plus propulsion controls are 61 % (New
  Flyer), 42 % (Proterra); for BYD the ESS control module, the propulsion control switch and wiring are
  64 % (pp. 34-37). New Flyer replaced 11 high-voltage batteries: battery strings went off-line, and
  performance was unstable below 15 % state of charge (SOC), because cell voltages were imbalanced
  (p. 36). Proterra: ESS cooling 15 % of propulsion defects (Chart 8, p. 37). BYD: intermittent battery
  thermal-management faults since summer 2021, no full fix (p. 35). Cause named for most propulsion
  faults: wiring harnesses, pin connections, corrosion at terminals, water intrusion (pp. 34, 37).
- **Battery capacity and degradation.** The usable capacity is 271-291 kWh out of 360-440 kWh installed
  (Table 4, p. 55). Proterra released 6 % more usable capacity after updating its battery-life
  calculations; "impacts to battery health will not be immediately noticeable" (p. 58). TTC plans a
  strategy to monitor battery health over the bus life (p. 75). No capacity-fade data.
- **Inverter.** Named only in two pie charts: "inverter and inverter wiring" 10 % of New Flyer propulsion
  defects (Chart 6, p. 36), "inverter" 3 % of Proterra propulsion defects (Chart 8, p. 37). Multiplied by
  the propulsion share, this is about 0.3 % and 0.1 % of all defects (computed here). DC-link capacitors:
  not named.
- **Traction motor.** 4 % (New Flyer) and 10 % (Proterra) of propulsion defects (Charts 6, 8).
- **Tires and brakes.** Not in the defect charts. The report states that more regenerative braking makes
  friction brakes last longer (p. 49), without wear data. Operators of BYD buses complained about braking
  feel (p. 90); this is not a maintenance finding.
- **Charging interface.** Bus side: charge-port resistor not weatherproof (BYD), charge-receptacle lock
  motors and badly crimped high-voltage cable ends (New Flyer, Proterra) (pp. 35-38). Charger side:
  27 charge-point failures in 18,648 sessions; most common were cables run over by buses and broken
  retaining clips; availability 98.7-99.4 % (Table 14, p. 79; p. 80).
- **Other.** BYD carbon-steel frame needs annual rust proofing, 3-5 days out of service each year
  (p. 94). Long repairs come from parts lead times (73 days for a rear axle reducer gear, p. 19),
  retrofit campaigns and complex propulsion faults (p. 94).
- **Comparison with conventional buses.** Against the hybrid, not diesel: repair time 4 to 8 times longer
  (p. 94); MDBF lower except New Flyer. TTC assumes 25 % lower maintenance cost for eBuses in its
  life-cycle model, against 0.52 CAD/km for a bus (Table 21, p. 98). This is an assumption, not a
  result. Bus life: 13 years (p. 95).
- **Usage drivers named.** Winter: energy use up 20-30 % (pp. 44, 69), mainly electric heat; road salt for
  corrosion (p. 16); water intrusion for wiring.
- **Degradation or wear data.** None (no battery fade, no tire or brake wear).
- **Scope note.** The report warns its findings "may not be applicable to other transit authorities"
  (p. 2). Fleets young and under warranty.

### R4. PwC and partners for the German federal ministry, evaluation of the e-bus funding programme (2024)

- **Reference.** PricewaterhouseCoopers GmbH WPG (PwC), with ifeu (Institut für Energie- und
  Umweltforschung Heidelberg), Prognos AG, EEBC and CP/COMPARTNER (2024). *Begleituntersuchung zur
  Förderung von Elektrobussen im ÖPNV. Abschlussbericht 2024* (accompanying study of the funding of
  electric buses in local public transport; final report), for the Federal Ministry for Economic Affairs
  and Climate Action (BMWK), 383 pages. URL:
  https://www.pwc.de/de/branchen-und-markte/oeffentlicher-sektor/begleituntersuchung-zur-foerderung-von-elektrobussen-im-oepnv.pdf
  (pwc.de answers 403 to scripts; read from the Wayback Machine copy of 1 Aug 2025,
  http://web.archive.org/web/20250801103511/https://www.pwc.de/de/branchen-und-markte/oeffentlicher-sektor/begleituntersuchung-zur-foerderung-von-elektrobussen-im-oepnv.pdf).
  Short version (*Kurzfassung*, 20 pages):
  https://www.prognos.com/sites/default/files/2026-01/begleituntersuchung-zur-foerderung-von-elektrobussen-im-oepnv-kurzfassung.pdf
- **Evidence basis.** Long version: read in part, pp. 1-9 (contents, glossary), 108-129 (Section 2.2.8,
  interviews with operators, authorities and manufacturers), 214-219 (Sections 3.1.2.3 and 3.1.3,
  availability data), 237-241 (Section 3.2.2, cost comparison), 335 (Santiago case), 358-359 (Oslo
  case), 372, 377 (assumptions of the cost comparison). Pages chosen by a keyword search for maintenance,
  workshop, failure, tyre, brake, inverter, power electronics, ageing. Short version: read in full; it
  holds no maintenance finding beyond the long version.
- **Fleet and data.** Programme 2018-2023: 1,489 funded BEBs and 823 charge points, 65 operators
  (*Verkehrsunternehmen*, VU) in 13 German states (Kurzfassung p. 4). Availability data: monthly
  unavailable days per bus, reported by hand by only 19 operators, January 2021 to December 2023, up to
  about 400 buses per month (pp. 215-216, Figure 111 p. 219). Interviews: 6 operators and 2
  manufacturers in 2021/22; 5 operators, 3 manufacturers and 2 authorities in 2023/24 (pp. 108, 120).

Findings.

| Item | Value | Location |
|---|---|---|
| Mean BEB availability, all 19 operators | 86 % (4.3 unavailable days per month) | p. 216 |
| Same, without the 4 worst operators | 90 % (3 days per month) | p. 216 |
| Diesel availability, older study (NOW GmbH 2021), method may differ | 93 % (2.1 days per month) | p. 216 |
| Unavailable days: "sonstiger Defekt" (other defect: air conditioning, doors, destination signs) | 50 % | Figure 110, p. 218 |
| Unavailable days: "Defekt im Antriebsstrang" (defect of the electric drivetrain) | 29 % | same |
| Unavailable days: defect of the charging infrastructure | 8 % | same |
| Unavailable days: "Wartung" (scheduled maintenance) | 8 % | same |
| Unavailable days: accident or vandalism | 5 % | same |

- **Causes at the four operators with 50-70 % availability** (asked directly by the authors, p. 216):
  one, charging infrastructure; two, causes not specific to e-buses ("Klimatisierung,
  Elektronikstörungen, Reifenventile": air conditioning, electronics faults, tyre valves); one, an early
  generation of LMP (lithium-metal-polymer) batteries and the electric drive axles of one supplier.
- **Traction battery.** One generation of LMP batteries had to be replaced across the fleets
  ("flächendeckend ausgetauscht"); electric drive axles of one supplier, used by several bus makers,
  also failed on a large scale (p. 124). Operators keep a "Restunsicherheit bei Batterielebensdauer"
  (residual uncertainty on battery life) (Table 10, p. 116), and uncertainty on the residual value of
  the battery (p. 125). Depot charging at lower power is preferred, among other reasons, because it is
  "positiv auf die Lebensdauer der Batterie" (good for battery life) (p. 121). Operators ask for long
  battery warranties; these exceed what the cell suppliers give to the bus makers (pp. 123-124, 126). No
  warranty length is given.
- **Battery replacement in cost practice (assumption, not field data).** The cost comparison assumes one
  battery replacement at half of a 12-year holding period, 55,000 km per year (p. 238). The replacement
  adds about 56,000 EUR (discounted) to a 12 m depot-charged bus in 2023 (CAPEX bus 625,881 against
  569,573 EUR, table p. 377; difference computed here).
- **Maintenance cost against diesel (assumption, not field data).** "Wartungskosten Bus" (bus
  maintenance cost) over 12 years, discounted at 2 %: diesel 302,455 EUR; BEB depot charger 226,841 EUR
  (75 % of diesel); BEB opportunity charger 285,005 EUR (table p. 377). Staff costs are excluded (p. 238).
  The source of these rates is not stated in the pages read; they stay constant to 2030 because "die
  Datenlage hier unsicher ist" (the data are uncertain) (p. 239).
- **Inverter, DC-link capacitors, tyres, brakes.** No finding in the pages read, except "Reifenventile"
  (tyre valves) above.
- **Workshop.** Operators build roof work stands for battery work (Ludwigslust-Parchim p. 286 in the
  keyword search; Berlin: a hall for up to 150 e-buses with roof stands and lifting jacks, p. 302).
- **Case studies (secondary, cited by PwC).** Santiago: BYD maintains the whole electric drivetrain at a
  fixed rate of about 0.15 EUR/km with an availability clause; the operator Metbus gives about
  0.24 EUR/km for diesel; availability 99.2 % in the first year from 2017 (p. 335; source: C40 Knowledge
  Hub). Oslo: lower wear of brakes and drivetrain reported, but early reliability below expectations,
  with larger problems on battery and drivetrain and many small non-drive problems; one operator had
  50 % availability in the first months; heating about 50 % of energy in winter; soft asphalt deformed by
  the heavier buses disturbed pantograph contact (p. 358; source: Thorne et al. 2021, Energy Policy,
  a lead). Oslo maintenance cost "noch etwas höher" (still somewhat higher) than diesel (p. 362, in the
  keyword search).
- **Degradation or wear data.** None in the pages read.
- **Scope note.** The authors warn that the operators assigned the causes themselves, so the split by
  cause must be read with care (p. 217).

### R5. Engadin Bus, St. Moritz Energie and EBP for the Federal Office of Transport, rural mountain regions (2020)

- **Reference.** Kollegger, R. (Engadin Bus); Casagrande, P. (St. Moritz Energie); Clausdeinken, H.;
  Rosser, S.; de Haan, P. (EBP Schweiz AG); with the Paul Scherrer Institut (PSI) (2020).
  *Elektrifizierung von Bussen in ländlichen Bergregionen. Schlussbericht* (electrification of buses in
  rural mountain regions, final report). Bundesamt für Verkehr (BAV, Federal Office of Transport),
  programme ESöV 2050 (Energiestrategie 2050 im öffentlichen Verkehr: energy strategy 2050 in public
  transport), project P-196, Bern, 29 Oct 2020, 52 PDF pages. URL:
  https://www.gr.ch/DE/institutionen/verwaltung/diem/aev/dokumenteoev/elektrifizierungvonbusseninlaendlichenbergregionenschlussbericht.pdf
  (downloaded 9 Nov 2026).
- **Evidence basis.** Read in full, except the English and French summaries (PDF pp. 13-21), which
  translate the German summary (PDF pp. 9-12) and were only scanned. Page numbers below are the printed
  ones ("Seite"); PDF page = printed page + 2.
- **Nature.** A planning study: fleet model, life-cycle assessment and cost comparison for the Upper
  Engadine. **No field data on maintenance, failures or wear.** It is entered because it is the Swiss
  federal guidance for BEBs on hilly, cold, rural lines, close to the countryside lines of the case study.

Findings.

- **Traction battery is the critical component.** "Bei batterieelektrischen Bussen ist vor allem die
  Fahrzeugbatterie eine kritische Komponente" (in BEBs the vehicle battery above all is a critical
  component) (p. 25).
- **Battery ageing drivers listed** (p. 25): number of charge and discharge events, their depth, their
  power, calendar age, battery temperature in operation. Any increase of one of them tends to increase
  ageing (p. 26). The report says ageing models must be calibrated per battery type and maker, so a
  general model is "meist nicht hilfreich" (mostly not helpful) (pp. 25-26).
- **Battery life and replacement.** End of life (EoL) at 80 % of the initial capacity, by market
  definition. "Die meisten Bushersteller gehen davon aus, dass ihre Batterien ihr EoL nach einem Betrieb
  von 6 bis 8 Jahren erreichen" (most bus makers expect EoL after 6 to 8 years of operation). With a bus
  life of 12 to 14 years in Switzerland, plan one battery replacement per bus (p. 26). The cost section
  puts the replacement battery after about 5 to 7 years (p. 42). These are manufacturer expectations and
  planning rules, not measured lives.
- **Tyres.** Depot chargers carry large batteries; "Das höhere Gewicht führt zu einem etwas höheren
  Traktionsenergiebedarf und etwas höheren Reifenabrieb" (the higher weight leads to somewhat higher
  traction energy and somewhat higher tyre abrasion) (pp. 23-24). No numbers.
- **Usage conditions named for mountain lines** (pp. 7, 20, 31-32): temperatures from about -25 °C to
  +30 °C, snow and heavy road salting in winter, steep gradients, long daily distances. Traction energy,
  heating or cooling, and topography have the largest effect on energy use (p. 31). Planning value:
  70,000 km per year and 1.7 kWh/km for a 12 m bus (p. 35).
- **Cell chemistry by charging concept.** NMC (lithium nickel manganese cobalt oxide) typical for
  opportunity chargers, LFP (lithium iron phosphate) typical for depot chargers (caption of Figure 11,
  p. 39).
- **Heating.** Auxiliary heaters burning (bio)diesel or bioethanol below about 5 °C; heat pumps
  guaranteed to about 5 °C (p. 28).
- **Inverter, DC-link capacitors, brakes, other components.** Not discussed.
- **Comparison with diesel.** Bus purchase cost about double; diesel and BEB have the same life of 12 to
  14 years (pp. 42, 47). No maintenance cost comparison.
- **Degradation or wear data.** None.

### R6. NREL, Long Beach Transit final report (2020)

- **Reference.** Eudy, L.; Jeffers, M. (2020). *Long Beach Transit Battery Electric Bus Evaluation: Final
  Report.* National Renewable Energy Laboratory, NREL/TP-5400-75582, September 2020, 53 PDF pages.
  Funded by CARB. URL: https://www.nrel.gov/docs/fy20osti/75582.pdf (downloaded from
  https://docs.nlr.gov/docs/fy20osti/75582.pdf, 9 Nov 2026).
- **Evidence basis.** Read in full (text layer; appendix tables pp. 37-42 scanned for the system rows).
  Page numbers below are the printed ones; PDF page = printed page + 11.
- **Fleet.** 10 BYD 40-foot BEBs, LFP ("ferro-type lithium iron phosphate") battery of 324 kWh, upgraded
  to 360 kWh; one 90 kW traction motor; overnight depot charging at 80 kW. Baseline: 8 Gillig CNG buses
  of similar age, randomly dispatched. Main data year 2019; the BEBs ran mostly on a slow downtown
  shuttle (8.1 mph average), 1,600 miles per bus per month against 3,586 for CNG (Table 1 p. 3; pp. 8,
  10-11).
- **Warranty.** BYD extended the full-vehicle ("bumper-to-bumper") warranty to the 12-year bus life. The
  battery warranty guarantees 70 % of a base capacity for 12 years; the base capacity is 80 % of the
  original 324 kWh (p. 3). Warranty repairs are removed from all costs (p. 26). So BEB parts costs are
  mostly absent from the cost shares (p. 28).

Findings.

| Item (2019) | BEB | CNG | Location |
|---|---|---|---|
| Availability (target 85 %) | 71.3 % | 83.4 % | Table 3, p. 12 |
| Unavailable days: general bus maintenance | 553.0 (17.4 % of planned days) | 163.0 (5.6 %) | Table 4, p. 14 |
| Unavailable days: preventive maintenance | 177.5 (5.6 %) | 167.5 (5.8 %) | same |
| Unavailable days: electric drive | 66.0 (2.1 %) | - | same |
| Unavailable days: ESS (traction battery) | 6.5 (0.2 %, computed here; the table prints 2.1 %) | - | same |
| Unavailable days: charging issues | 39.5 (1.2 %) | - | same |
| Unavailable days: transmission | 66.0 (2.1 %) | 21.0 | same |
| Miles between road calls (MBRC), all, 2018-2019 | 4,416 | 10,307 | Table 8, p. 25 |
| Propulsion-related MBRC | 11,776 | 17,828 | same |
| ESS-related MBRC (5 road calls) | 70,656 | - | same |
| Total maintenance cost, USD/mile | 0.47 | 0.56 | Table 9, p. 27 |
| Cab, body and accessories share of cost | 52 % (rank 1) | 28 % | Table 10, p. 29 |
| PMI (inspection labour) share | 23 % (rank 2) | 14 % | same |
| Propulsion-related share | 9 % (rank 3) | 38 % | same |
| Brakes share | 4 % | 2 % | same |
| HVAC share | 2 % | 5 % | same |
| Tires share | 0 % | 0 % | same; no reason given |

- **Where downtime comes from.** "Many of the issues affecting availability involved general bus systems
  not related to the advanced technology": doors, axles, suspension, fire suppression, air system,
  cameras, wheelchair ramps (pp. 14-15). Advanced-technology issues: inductive charging, electric drive,
  high-voltage batteries (p. 15).
- **Traction battery.** "The ESS is not the system causing in-service issues" (p. 25). Daily use: the
  buses discharge 50-60 % of the usable energy per day (p. 17). No capacity-fade data.
- **Propulsion costs inside.** Electric propulsion plus ESS: 17 % of propulsion cost in 2019, against
  35 % in 2018, mostly labour (p. 31). Non-lighting electrical (low-voltage batteries, switches): 81 % of
  BEB propulsion cost; ITS equipment (radios, cameras, fareboxes) drains the low-voltage batteries when
  parked, and NREL has seen premature failure of these batteries at other agencies (p. 31).
- **Inverter, DC-link capacitors.** Not named.
- **Axle.** Shaft bolts broke on the rear axle reduction gear sets of several buses; other agencies saw
  the same; BYD redesigned the part and replaced it on all buses under warranty (p. 35).
- **Tyres.** Cost zero in both fleets (Table 10, p. 29; Table A-3); the report gives no reason.
- **Comparison with CNG, not diesel.** BEB total cost 16 % lower; propulsion cost 81 % lower, but the
  BEBs were under warranty and the CNG buses were not (p. vi, p. 31).
- **Maintenance actions.** Scheduled cost = preventive maintenance by the makers' recommendations
  (p. 26); no intervals given.
- **Degradation or wear data.** None.
- **Scope note.** One small fleet of one maker, on a slow shuttle; the CNG buses ran different routes at
  higher speed (p. 1).

### R7. AC Transit (Oakland, California), Zero Emission Program annual progress report (2026)

- **Reference.** Alameda-Contra Costa Transit District (AC Transit) (2026). *Zero Emission Program.
  Annual Progress Report.* Publication date 24 June 2026, Board staff report SR 26-253, Attachment 1,
  33 PDF pages. It continues the "Zero Emission Transit Bus Technology Analysis" (ZETBTA, the "5x5
  study"). URL:
  https://www.actransit.org/sites/default/files/2026-10/2026%20Zero%20Emission%20Program%20Annual%20Progress%20Report.pdf
  (downloaded 9 Nov 2026).
- **Evidence basis.** Read in full (text layer; monthly charts not read, their numbers are in the text).
  Page numbers below are the printed ones; PDF page = printed page + 2.
- **Fleet.** Calendar year 2025, 40-foot buses built 2016-2025: 35 diesel (Gillig 2018), 25 diesel
  hybrid, 30 fuel-cell buses, 28 BEBs in three groups: 8000 (5 New Flyer 2019, 466 kWh, Siemens drive),
  8006 (2 Gillig 2021, 444 kWh, Cummins drive), 8008 (21 Gillig 2022, 444 kWh, Cummins drive, in service
  from August 2024). All lithium-ion. Depot charging. Lines "generally flat" (Figure 14, p. 14; p. 25).
- **Warranty caveat.** "The high voltage batteries (energy storage system) and fuel cell are still under
  warranty on the ZEBs therefore mitigating the high cost of those parts"; warranty claims take several
  months to a year (p. 12). BEB group 8008 filed 192 warranty claims worth 670,207 USD in 2025 (Figure 20,
  p. 20).

Findings (2025, BEB against the diesel baseline).

| Item | Diesel | BEB | Location |
|---|---|---|---|
| Fleet availability | 89.21 % | 65.19 % (groups: 19 %, 57 %, 72 %) | Figure 13 p. 13; p. 22 |
| Miles between chargeable road calls (MBCRC) | 14,761 | 5,791 (groups: 6,486; 10,062; 5,644) | Figure 13 p. 13; Figure 22 p. 23 |
| Maintenance cost, USD/mile | 1.01 | 3.56; 1.38; 0.76 (groups 8000, 8006, 8008) | Figure 18, p. 18 |
| Total cost with credits, USD/mile | 2.13 | 2.10 | Figure 13, p. 13 |

- **Road calls by system** (Figure 23, p. 23). Largest BEB group (8008, 72 road calls): high-voltage
  system 35 (49 %), "common system failures" 33 (46 %), transmission or electric drive 1, other 3
  (shares computed here). The most common BEB failures are "Electrical system problem" and "Battery
  related problem" (p. 24). For diesel: brakes, engine, air conditioning, air leaks (p. 23). Note: the
  diesel and hybrid totals in Figure 23 (142, 125) are swapped against Figure 22 (125, 142).
- **Downtime driver.** "The largest contributor to the low availability the ZEB fleet is experiencing is
  because of the length of time it is taking to repair the vehicle. Long lead time on batteries and
  mid-life repairs to the fuel cell being the main items" (p. 22).
- **Traction battery.** Here the high-voltage system leads the BEB road calls, unlike R1, R3 and R6.
  The battery is under warranty, so its cost is hidden (p. 12). Risk register: "Potential ZEB thermal
  events caused by high voltage battery design" (Figure 3, p. 4). No capacity-fade data.
- **Inverter, DC-link capacitors, tyres, brakes (BEB).** Not named for BEBs. Brakes are named only among
  diesel failures (p. 23).
- **Charging.** "BEB Charging Infrastructure Reliability" is a monitored programme risk (Figure 3, p. 4).
- **Comparison with diesel.** Diesel stays the most available and most reliable; the authority concludes
  that "Zero-emission technology systems are not less reliable than conventional technology", because
  zero-emission propulsion failures were fewer than diesel propulsion failures (p. 24).
- **Degradation or wear data.** None.
- **Scope note.** Small, young BEB groups; the report says BEB costs vary strongly and "further research
  will be essential as more service miles are logged" (p. 13).

### R8. World Bank, case study of the Shenzhen Bus Group (2021)

- **Reference.** The World Bank (2021). *Electrification of Public Transport: A Case Study of the Shenzhen
  Bus Group.* Mobility and Transport Connectivity Series, World Bank, Washington, DC; with Shenzhen Bus
  Group (SZBG), University of California Davis (Institute of Transportation Studies) and China
  Development Institute. 138 PDF pages. URL:
  https://documents1.worldbank.org/curated/en/708531625052490238/pdf/Electrification-of-Public-Transport-A-Case-Study-of-the-Shenzhen-Bus-Group.pdf
  (downloaded 9 Nov 2026).
- **Evidence basis.** Read in part: printed pp. 43, 56-62 (Section 4.4, maintenance and asset
  management, in full), 83-85 (Section 6.2.3, maintenance cost, in full), executive summary p. 6. Other
  pages (pp. 1-5, 11-15, 29, 32-40, 48-50, 70-79) seen only through a keyword search (maintenance,
  repair, failure, tire, brake, battery, warranty); claims from them are marked "keyword search".
  Printed page = PDF page - 10.
- **Fleet.** SZBG ran about 6,000 BEBs and was fully electric in June 2017; the main model is the BYD K8,
  10.5 m, about 250 km range, depot charging overnight, about 190 km per day in 2019 (p. 43; pp. 1, 29
  in the keyword search).
  Hot and humid climate. Baseline: 11 m diesel buses of the same operator, 8-year life, 66,000 km per
  year (pp. 58-59, 85).
- **Warranty.** The makers cover the "3-e system" (battery, electric motor, electronic controller): eight
  years for buses bought later; battery replaced when capacity falls below 80 % (printed p. 49 in the
  keyword search; p. 60). So 3-e costs are excluded from SZBG's maintenance costs (p. 84).

Findings.

| Item | BEB | Diesel | Location |
|---|---|---|---|
| Tyre replacement cost, CNY per 1,000 km | 125 | 90 (BEB +38.8 %) | p. 83 |
| Regular maintenance cost, CNY per 1,000 km (technicians' estimate) | about 200 | about 600 | p. 83 |
| Maintenance cost, CNY per 1,000 km, year 1 / 2 / 3 / 4 | 75 / 152 / 211 / 242 (measured, BYD K8) | 318 / 476 / 502 / 2,706 (year 4 = diesel overhaul) | Table 6-7, p. 84 |
| Maintenance cost, years 5-8 | assumed +20 % per year | measured | p. 84 |
| Maintenance levels | level I every 4,000-5,000 km; level II every 20,000 km | level II every 20,000 km; overhaul in year 4 | pp. 57, 83 |

- **Traction battery (field experience).** "Some buses experienced battery degradation as early as at
  their 50,000-kilometer mileage." Most BYD K8 batteries needed replacement after 2-2.5 years; other
  models after about 3-4.5 years. Makers replaced the battery only after several repair attempts, and
  often only some cells. The report assumes one replacement every four years on average, two packs per
  bus life (p. 60). Elsewhere: K8 batteries need a change after 2-4 years "depending on the driving
  behavior, the route characteristics, and the battery energy density" of the batch (p. 84). Later
  batches moved from air-cooled to liquid-cooled batteries "to prolong battery life" (p. 50, keyword
  search). Early models (2011-2013) lost range and needed "frequent maintenance" (p. 43).
- **Defects.** The BEBs had more defects per 1,000 km than diesel buses in year one, because of design
  changes. In year two about half of all BEB defects were in the 3-e system under warranty; other repair
  issues were "compressor defects and battery degradation" (Figure 4-9 and text, p. 58).
- **Tyres.** "Tire wear is more for electric buses because of heavier weight" (p. 57). Tyre cost is
  about 30 % (p. 58) or 38.8 % (p. 83) higher than diesel. Reasons given: higher vehicle weight, and
  in-wheel electric motors that act in propulsion and braking (p. 83).
- **Brakes.** Disc brakes need less work than drum brakes (p. 57). No wear data.
- **Electronics.** Mechanical work falls, "However, work on electronic parts increases" (p. 57). After
  electrification only 55 % of the mechanical technicians were needed; 619 electromechanical technicians
  were added (Table 4-8, p. 61). Inverter and DC-link capacitors are not named.
- **HVAC.** The inverter air conditioner is fully welded and needs less repair (p. 57); compressor
  defects appear among BEB repair issues (p. 58).
- **Comparison with diesel.** Lifetime maintenance cost of the BEB estimated at about 30-40 % of the
  diesel bus (p. 58), with 3-e work under warranty.
- **Degradation or wear data.** Battery replacement ages and mileage (yes, coarse); tyre cost per km
  (yes). No capacity curves.
- **Scope note.** One operator, one dominant model (BYD K8), subtropical climate, flat city; makers carry
  the 3-e risk, so the operator's costs omit the battery.

### R9. NREL and CTE, Duluth Transit Authority evaluation: cold, hilly, salted roads (2022)

- **Reference.** Jeffers, M.; Eudy, L. (NREL); Bigelow, E.; Olberding, G.; Posner, A. (Center for
  Transportation and the Environment, CTE) (2022). *Duluth Transit Authority Battery-Electric Bus
  Evaluation.* National Renewable Energy Laboratory, NREL/TP-5400-83038, September 2022, 55 PDF pages.
  Funded by the Federal Transit Administration (FTA) and the US Department of Energy. URL:
  https://www.nrel.gov/docs/fy22osti/83038.pdf (downloaded from https://docs.nlr.gov/docs/fy22osti/83038.pdf,
  9 Nov 2026).
- **Evidence basis.** Read in full (text layer; charts not read; Appendix B, the diesel duty cycle, read
  by keyword). Page numbers below are the printed ones; PDF page = printed page + 12.
- **Fleet.** 7 Proterra Catalyst 40-foot BEBs, 440 kWh (354 kWh usable), 220 kW permanent-magnet motor,
  depot charging at 50 kW; baseline 10 Gillig diesel buses of 2018. All buses carry diesel-fired
  auxiliary heaters. Duluth, Minnesota: cold winters (down to -30 °F, about -34 °C), snow, road salt,
  steep hills. BEB data late 2018 to spring 2021 (Table 1 p. 4; pp. 3, 7-8, 33).
- **Maintenance costs.** Not available: early work was under warranty, and later DTA work orders were
  not provided (p. 29).

Findings.

| Item | BEB | Diesel | Location |
|---|---|---|---|
| Availability (target 85 %) | 51.8 % | 88.4 % | Table 4, p. 12 |
| Unavailable: general bus maintenance (auxiliary heater, brakes, suspension, doors) | 29.7 % of planned days | 5.0 % | Table 5, p. 15 |
| Unavailable: electric drive (HV batteries, traction motor, inverters) | 9.1 % | - | same |
| Unavailable: ESS (traction battery) | 5.3 % | - | same |
| Unavailable: charging issues | 2.6 % | - | same |
| Unavailable: preventive maintenance | 0.6 % | 2.0 % | same |
| Unavailable: engine | - | 4.5 % | same |
| MBRC, all / propulsion / ESS (Apr 2019-Feb 2020) | 7,717 / 46,303 / 92,607 | 32,178 / 75,083 / - | Table 7, p. 28 |

- **Inverter.** Named inside the "electric drive" category: "problems with high-voltage batteries,
  traction motor, and inverters" (p. 15). No count of its own.
- **Temperature and failure mix.** "As temperatures decreased, the relative fraction of issues related to
  the electric drive system decreased, while the fraction of ESS (battery) issues and general bus issues
  both increased" (pp. 25-26; Figure 24).
- **Traction battery use.** The lowest 20 % of state of charge "should be avoided whenever possible to
  minimize battery degradation"; traction power is derated at 5-7 % SOC (p. 20). Range falls about 33 %
  for a 30 °F (about 17 °C) drop below 55-60 °F, and about 20 % for a 30 °F rise (p. 24). No capacity-fade
  data.
- **Brakes and connectors: corrosion.** "Multiple issues with corrosion on the buses (brakes, electrical
  connectors) due to frequent cold and snowy conditions and exposure to corrosive agents such as road
  salt"; the maker added splash guards and undercarriage covers (pp. 32-33).
- **Regenerative braking.** Hilly routes give high energy recovery; on average 21.6 % of the energy is
  recovered (p. 18). On snow, regenerative braking switches off when wheel slip is detected (pp. 26, 32),
  so friction brakes work more there (a reading of the mechanism; the report gives no brake-wear data).
- **Low-voltage batteries.** They went flat within weeks; a low 24 V state of charge caused many fault
  codes; they are "replaced more often than on the diesel buses" (pp. 30-31).
- **Capacitors (in the charger, not in the bus).** One depot charger failed catastrophically: "The root
  cause was traced to a capacitor quality issue—the capacitor overheated"; the maker replaced the
  capacitors in all its chargers with parts from another supplier and added a software shutdown on the
  warning code seen 12-18 hours before the failure (pp. 31-32). Charger availability was 95.3 % but fell
  later, and DTA had to park the BEB fleet at times (pp. 15, 32).
- **Other cold-climate items.** Axle undersized for a full load, retrofitted (p. 30); power-steering
  pump noise at -30 °F (p. 33); cabin heating insufficient until software and hardware changes (p. 32).
- **Downtime drivers.** Parts availability and long supply chains, staff shortages, and lack of maker
  technicians (pp. 31-32).
- **Degradation or wear data.** None.
- **Scope note.** Small, early fleet; BEBs had only 10 % of the fleet and were held out of some service by
  policy, which lowers their measured availability (p. 31).

### R10. NHTSA, safety recall 23V-697 for Proterra traction inverters (2023)

- **Reference.** US National Highway Traffic Safety Administration (NHTSA). Recall campaign 23V697000,
  Proterra Operating Company, Inc., report received 17 Oct 2023. Official record retrieved from
  https://api.nhtsa.gov/recalls/campaignNumber?campaignNumber=23V697000 (9 Nov 2026).
- **Evidence basis.** Read in full (the recall record: component, summary, consequence, remedy). The
  maker's Part 573 defect report itself was not read.
- **Content.** 2019-2022 Proterra Catalyst and ZX5 800 V transit buses with the ProDrive 1.0 drive,
  238 units. "A transistor inside the inverter may overheat and fail, which can result in a loss of power
  steering and drive power." Remedy: "Dealers will replace the inverter".
- **What it shows.** A power semiconductor of the traction inverter failing by overheating, in one maker's
  fleet, serious enough for a safety recall. It gives no failure rate and no usage driver. Evidence that
  one maker's product failed this way, not evidence of how common the failure is.

### R11. ThEGA (Thuringian energy agency), real data of the zero-emission bus fleet of Thuringia (2025)

- **Reference.** Schlegel, M. (ThEGA), with Ehrich, M.; Hofmann, R. (ThEGA); Winkelhöfer, A.; Zimmermann, E.
  (TMUENF, Thuringian Ministry for Environment, Energy and Nature Conservation) (2025). *Realdaten der
  emissionsfreien Thüringer Bus-Flotte. Analyse und Einordnung auf der Datenbasis von Ende 2024* (real
  data of the zero-emission bus fleet of Thuringia; analysis on the data of end 2024). Thüringer Energie-
  und GreenTech-Agentur GmbH, document version December 2025, 22 PDF pages. URL:
  https://www.thega.de/fileadmin/user_upload/Elektromobilit%C3%A4t/Studien/Betriebsdaten_Emissionsfreie_Busse_Thueringen_2024.pdf
  (404 on thega.de on 9 Nov 2026; read from the Wayback Machine copy of 15 Jun 2026,
  http://web.archive.org/web/20260615082630/https://www.thega.de/fileadmin/user_upload/Elektromobilit%C3%A4t/Studien/Betriebsdaten_Emissionsfreie_Busse_Thueringen_2024.pdf).
- **Evidence basis.** Read in full. Page numbers below are the printed ones; PDF page = printed page + 1.
- **Fleet and data.** 32 zero-emission buses funded by the state programme to the end of 2023 (3 of them
  fuel-cell buses in Weimar), with 2024 operating data; mostly 12 m city buses and midibuses, mostly
  depot charging; models named: Mercedes eCitaro, Sileo S12, MAN Lion's City 12E, most with
  diesel-fired auxiliary heaters (pp. 4, 7, 16). The authors add values from other German states and
  studies where needed (pp. 4, 6). The underlying monitoring report (ThEGA 2025) is internal, not public
  (p. 15).
- **Nature.** An agency summary of operators' data; many values are given as "Richtwerte" (guide
  values). The report does not give the number of buses behind each value.

Findings.

| Item | BEB | Diesel | Location |
|---|---|---|---|
| Maintenance cost, EUR/km | average < 0.20 (good cases < 0.15); Annex 2: "meist < 0,15" | > 0.30 | pp. 11-12; Annex 2, p. 17 |
| Energy use, kWh/100 km | about 120 (range 100-140), heating included | 40-45 l/100 km | p. 8 |
| Winter extra energy (electric heating) | +20 % to +35 % | small | p. 10 |
| Summer extra energy (cooling) | +5 % to +10 % | +5 % to +20 % | p. 10 |
| Life of the main propulsion part | battery: 8-12 years, 600,000-1,000,000 km ("600 -1.000 tkm") | engine: 12 years, 1-1.2 million km | table p. 4 |

- **Traction battery.** Charge cycles so far below 2,000. "Geplante Lebensdauer: 6-8 Jahre, 4.000
  Zyklen, Wahrscheinliche Lebensdauer: 8–12 Jahre (bei einer minimalen Restkapazität von 70%)" (planned
  life 6-8 years and 4,000 cycles; probable life 8-12 years, at a minimum remaining capacity of 70 %).
  The batteries show "nur moderate Degradationen" (only moderate degradation) after several years
  (pp. 9-10). Makers target up to 4,000 full cycles or about 12 years (p. 10). Annex 2 adds: battery
  ageing "kann langfristig zu höherem Instandhaltungsaufwand führen" (may raise maintenance effort in the
  long run), but most operators report stable range and no notable cell-degradation problems so far
  (p. 17). The annex attributes "über 3.000 Ladezyklen ohne signifikanten Kapazitätsverlust" (over 3,000
  cycles without significant capacity loss) to BMWK 2024 (R4) and to the PwC E-Bus-Radar 2025; the claim
  is not in the parts of R4 read, and the E-Bus-Radar was not read (UNVERIFIED secondary claim).
- **Brakes.** "Thüringer Betreiber berichten von verdoppelten Wartungsintervallen" (Thuringian operators
  report doubled maintenance intervals) for brake components, thanks to regenerative braking (p. 12).
- **Sources of the saving** (Annex 2, p. 17): no engine, gearbox or exhaust work; less brake wear; less
  work on auxiliaries.
- **Problems in the introduction phase.** Occasional lack of spare parts and software problems (p. 12).
  No safety incidents, vehicle fires or serious charging failures recorded in the monitoring (Annex 3,
  p. 18).
- **Inverter, DC-link capacitors, tyres.** Not discussed.
- **Degradation or wear data.** Battery cycle count and a qualitative statement of moderate fade; no
  capacity numbers. Brake interval doubled (qualitative-quantitative). No tyre data.
- **Scope note.** Small fleet; summary values without per-bus data; the source monitoring report is not
  public.

## 4. Could not read

| Item | Reason (9 Nov 2026) | Why it looks worthwhile |
|---|---|---|
| Transportation Research Board, TCRP Synthesis 130, *Battery Electric Buses: State of the Practice* (2018) | The National Academies Press (NAP) download needs a login; the scratchpad file `tcrp130.pdf` is the landing page (HTML), not the report | A survey of US agencies on BEB operation and maintenance |
| AC Transit and Stanford, *Zero Emission Transit Bus Technology Analysis* (ZETBTA, "5x5 study"), volume 3 (June 2022), and volumes 1, 2, 4 | The copy at h2fcp.org answered with an anti-bot HTML page; the address d7.h2fcp.org does not resolve; the AC Transit page links only the 2026 progress report (R7) | Five buses of each technology side by side, with cost per mile by system against diesel |
| NOW GmbH, final report of the federal bus programme accompanying research (cited by R4 as "NOW GmbH 2021a"; published about 2022) | Not located; searches found only references to it | Source of the 93 % diesel availability used in R4; may hold failure causes by component |
| ThEGA, *Monitoringbericht E-Busse Thüringen 2024* (2025) | Internal, not public (R11 p. 15) | The per-operator data behind R11, including maintenance costs |
| PwC, *E-Bus-Radar 2025* | Not attempted; pwc.de refuses scripts (R4) | Cited by R11 for battery lives beyond eight years and over 3,000 cycles |

## 5. Leads not followed

Grey literature with fleet data:

- **NREL and FTA, King County Metro BEB evaluation** (FTA Report No. 0118, 2018): hilly Seattle,
  fast-charged Proterra buses.
  https://www.transit.dot.gov/sites/fta.dot.gov/files/docs/research-innovation/115086/zero-emission-bus-evaluation-results-king-county-metro-battery-electric-buses-fta-report-no-0118.pdf
- **NREL progress reports superseded by final reports**: Foothill Transit NREL/PR-5400-75581 (2020,
  downloaded, superseded by R1); Long Beach first year, FTA Report No. 0163 (superseded by R6).
- **CTE for FTA, "Four-Season" analyses of temperature effects on BEB fuel economy and range** (2019,
  2020), cited by R9: https://cte.tv/wp-content/uploads/2019/12/Four-Season-Analysis.pdf
- **ICCT, urban fleets in Indore** (October 2026): telematics of 49 e-buses, maintenance against capital
  cost. https://theicct.org/wp-content/uploads/2026/10/ID-710-–-Urban-fleets-Indore_report_final-v3.pdf
- **Altoona Bus Research and Testing Center (FTA) test reports**: durability-test failure lists by
  component for each bus model; could give inverter or capacitor failures under an accelerated test.
- **NHTSA recalls of other BEB makers** with the component "inverter" or "electrical system" (method of R10).
- **Operator reports named in the press, not read**: VBZ, December 2024 (door, air conditioning, turn
  assistant and charging faults on about 50 MAN and 7 Hess e-buses); New York MTA, June 2026 (about
  2,500 miles between major failures against 8,500 for diesel); RNV Mannheim, January 2026 (more than
  half of 48 e-buses not available); Edmonton (Proterra fleet, parts shortages). An official report of
  each would be worth reading; the press items are not evidence.
- **European projects and associations**: ZeEUS eBus Report #2 (UITP, 2017); ASSURED; ELIPTIC; UITP
  knowledge brief on large-scale bus electrification (2021,
  https://cms.uitp.org/wp/wp-content/uploads/2021/07/Large-scale-Bus-Electrification-KB-Final.pdf);
  E-MetroBus and E-Bus 2030+ (BVG, Reiner Lemoine Institut, TU Berlin); Hamburg BEEDeL final report.
- **TNO and Arriva Limburg**: on-site battery state-of-health measurement through the charger, 43 buses,
  one-year trial (TNO news, 2025); no report found.
- **Swiss federal and cantonal studies**: INFRAS, *Fossil-free bus fleets* (for BAV); INFRAS, H2-bus
  environmental balance (BAV, December 2025); INFRAS for the canton of St. Gallen (2020); ESöV P-144
  *Leitfaden Flottenelektrifizierung für Busbetriebe* (VCDB, 2019); ESöV P-113; ESöV P-196 *Fallstudie
  Oberengadin* (2020). All planning or cost studies, probably without maintenance field data.
- **World Bank, Lessons from Chile's experience with e-mobility** (Santiago), and the C40 Knowledge Hub
  page on Metbus, both cited by R4 for the 0.15 against 0.24 EUR/km maintenance rates.

Checked and out of scope:

- BFE publication 7366 (pubdb.bfe.admin.ch): a 5-page trade article on the Geneva TOSA flash-charging
  demonstrator; no maintenance data (first page read).
- *Abschlussbericht Plattform Innovative Antriebe Bus* (PE International for BMVBS, October 2011,
  edocs.tib.eu): mostly diesel-hybrid buses (title pages read).

Peer-reviewed (leads only, covered by another agent): Thorne et al. (2021), Energy Policy, Oslo e-bus
experience (cited in R4); a 2022 International Journal of Prognostics and Health Management paper on
fleet-level battery state-of-health estimation for 105 vehicles.
