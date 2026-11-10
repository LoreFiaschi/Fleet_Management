# Battery-electric city bus types for the case study

Status: done (10 Nov 2026). Sections 1-5 filled: 13 sources entered (V1-V13; two of them are official
Swiss datasets, downloaded and counted) and 6 carried entries (C1-C6). Next, if more work is wanted: the
items of section 4 (the TÜV SÜD document of the MAN run first) and the Hess and Irizar data sheets
(section 5). Downloads in scratchpad `downloads\bus\01`-`27`; text extracts in `text_bus\`; scripts in
`scripts_bus\`.

This file lists the battery-electric city bus types that the scheduling case study could use. For each type it records the public data on the three tracked components: the traction battery, the tires and the traction-inverter power modules.

## 1. Comparison table

Abbreviations: NMC = lithium nickel manganese cobalt oxide cathode; LFP = lithium iron phosphate;
LTO = lithium titanate; SiC = silicon carbide; IGBT = insulated-gate bipolar transistor; CCS = Combined
Charging System plug; GVW = gross vehicle weight; GVWR = gross vehicle weight rating (US term);
PSM = permanent-magnet synchronous motor; CH = Switzerland; n/a = no source read gives it.
Source codes (V1-V13, C1-C6) are the entries of section 3. "Registered empty mass" = the mass in the
Swiss vehicle register (V7), driver of 75 kg included (V8).

The table is split in three parts (vehicle and tyres; battery; drive and energy), one row per type in
each part. 12 m types first, then 18 m types.

### 1a. Vehicle, masses and tyres

| Type | Length; years | Where it runs | Empty mass | Gross mass; axle loads | Passengers | Tyres |
|---|---|---|---|---|---|---|
| **Mercedes-Benz eCitaro (12 m)** | 12,135 mm (V1); series from 2018 (C6) | > 3,000 delivered by 2025 (C6); CH 211 registered, 4 approvals (V7); VBG line 759, Zurich region (V7 inference) | 13,400 kg (NMC1 test bus, V10); registered 13,450-15,250 kg, medians 14,280-14,800 kg (V7) | 20,000 kg; front 7,500 kg, drive 13,000 kg (V1); CH registered mostly 19,500 kg (V7) | up to 88 (V1) | 275/70 R22.5; 315/60 R22.5 front option (V1, V4, V10) |
| **MAN Lion's City 12 E** | 12,185 mm (V2); CH approval 2020 (V4) | Germany, Spain trial (V11); CH 42 registered, of them 22 in Zurich city = VBZ (V7) | registered 14,570-15,100 kg, median 14,900 kg (V7) | 20,000 kg (V2); front 6,300-8,000 kg, rear 11,500-13,000 kg (V4) | seats up to 44 (V2); registered 76 + 44 (V7) | 275/70R22.5 (V4) |
| **Solaris Urbino 12 electric** | 12,000 mm (V3); since 2013 (V13) | > 800 in Europe in 2022 (V12); CH 120 registered (V7) | 13,900 kg, maker value, 300 kWh bus (V5); registered 14,200-15,370 kg (V7) | 18,745 kg; front 7,245 kg, rear 11,500 kg (V5); 20,000 kg (V3) | up to 77 (V3) | 275/70 R22.5 all round (V5, V4) |
| Irizar ieBus / ieTram 12 m | 12,160 mm (V4) | CH 63 registered under 12 m approvals (V7) | registered 13,450-15,250 kg (V7) | 19,000-21,165 kg; front 7,500-8,165 kg, rear 13,000 kg (V4) | registered 30-86 seats (V7) | 275/70R22.5 (V4) |
| Volvo 7900 Electric | 10-13.5 m (V4) | CH 4 registered (V7) | registered 13,400 kg (V7) | 19,100-19,500 kg; front 7,100-7,500 kg, rear 12,000 kg (V4) | registered 77 + 39 (V7) | 275/70R22.5 (V4) |
| VDL Citea SLF-E 120 | 12,000 mm (V4) | CH approval 2018, none registered (V4, V7) | n/a | 19,000 kg; front 7,245 kg, rear 12,000 kg (V4) | n/a | 275/70R22.5 (V4) |
| New Flyer Xcelsior CHARGE NG 40 ft | 12.24 m (V9) | USA, Canada (V9) | n/a | n/a | 40 seats + 44 standees (V9) | 305/70R22.5 (V9) |
| BYD 40 ft (K9 family, 6120LGEV) | 40.2 ft (12.3 m) (C1) | USA (C1); not approved in CH (V4) | n/a | GVWR 19,700 kg (C1) | n/a | n/a |
| Proterra Catalyst E2 40/42 ft | 42 ft (12.8 m) (C3) | USA (C2, C3) | curb 14,225 kg (C3) | GVWR 19,050 kg (C2, C3) | 40 seats (C3) | n/a |
| **Mercedes-Benz eCitaro G (18 m)** | 18,125 mm (V1) | CH 163 registered (V7); VBG line 768 (V7 inference) | registered 19,650-22,100 kg, medians by approval 19,650-20,900 kg (V7) | 29,000-30,000 kg; front 7,500, centre 10,000-11,500, drive 13,000 kg (V1) | up to 146 (V1) | 275/70 R22.5 (V1, V4) |
| **MAN Lion's City 18 E** | 18,060 mm (V2); CH approval 2021 (V4) | CH 32 registered, 28 in Zurich city = VBZ (V7) | registered 21,950-22,600 kg, median 22,600 kg (V7) | 30,000 kg (V2); 30,000-31,000 kg (V4) | seats up to 60 (V2); registered 113 + 70 (V7) | 275/70R22.5 (V4) |
| **Solaris Urbino 18 electric** | 18,000 mm (V4) | CH 64 registered (V7); Berlin BVG 50 ordered (V12) | registered 20,500-22,340 kg, median 21,380 kg (2025 approval) (V7) | 28,000-30,000 kg (V4) | registered 121 + 79 (V7) | 275/70R22.5; options 315/60 front, 455/45 axle 2 (V4) |
| Irizar ieBus / ieTram 18 m | 18,730 mm (V4) | CH 27 registered under 18 m approvals (V7) | registered 20,300-21,450 kg (V7) | 29,000-31,165 kg (V4) | registered 38-133 seats (V7) | 275/70R22.5, twin on axles 2-3 (V4) |
| New Flyer Xcelsior CHARGE NG 60 ft | 18.29 m (V9) | USA, Canada (V9) | n/a | n/a | n/a | 305/70R22.5 (V9) |

### 1b. Traction battery

| Type | Capacity (gross; usable) | Chemistry; cells | Packs and layout | Life, warranty (maker statements) | Charging |
|---|---|---|---|---|---|
| **eCitaro 12 m** | NMC1 up to 243 kWh (V10); NMC3 98 kWh per pack (C6); NMC4 444/555/666 kWh; usable n/a (V1) | NMC; Akasol cells (NMC1, NMC3: 21700 cylindrical); BMZ Poland prismatic (NMC4) (V10, C6) | NMC1: 25 kWh modules; NMC4: 4-6 packs of 111 kWh, 2 rear + up to 4 roof (V1, V10) | Akasol "assumes 4,000 charging cycles"; NMC4 guarantee 10 years, up to 15 optional (C6) | CCS up to 150 kW; pantograph or rails up to 300 kW (C6, V1) |
| **MAN 12 E** | 4/5/6 packs: 356/445/534 kWh; usable 65 % (231-347 kWh) or 90 % (320-480 kWh) by strategy (V2); older packs 80 kWh (V6) | NMC; Volkswagen Group cells (V2, V6, V11) | 12 exchangeable modules per pack, packs on the roof (V2); 550 kg per pack (V6) | "range up to 270 km over the entire service life" (V11); "up to 12 years" (V6) | depot CCS up to 150 kW (V2) |
| **Solaris U12 electric** | 300 kWh nominal, about 80 % used = 240 kWh (V5, 2020); 420 kWh (V3); > 520 kWh (V12) | NMC (V5); cell maker n/a | 6 modules of 50 kWh (V5); roof and rear (V13) | n/a | plug up to 260 kW; pantograph up to 450 kW (V3) |
| Irizar ieBus 12 m | n/a | n/a | n/a | n/a | n/a |
| Volvo 7900 Electric | n/a | n/a | n/a | n/a | n/a |
| VDL Citea SLF-E | n/a | n/a | n/a | n/a | n/a |
| New Flyer CHARGE NG 40 ft | 345/435/520 kWh (V9) | NMC (V9) | standard enclosures on roof and in the rear (V9) | 12-year warranty available (V9) | plug; pantograph up to 450 kW (V9) |
| BYD 40 ft | 324 kWh, upgraded 360 kWh (C1) | LFP, by BYD (C1) | n/a | 70 % of a base (80 % of 324 kWh) for 12 years (C1) | depot plug; 50 kW inductive (C1) |
| Proterra Catalyst E2 | 440 kWh; 354 kWh usable (C2) | n/a | n/a | operator-bought warranty to 80 % of usable capacity (C2) | depot, 50 kW (C2) |
| **eCitaro G** | NMC4 up to 777 kWh (V1); VBG buses 550 kWh usable (`zurich_depot_and_lines.md`) | as eCitaro | 4-7 packs of 111 kWh, 2 rear + up to 5 roof (V1) | as eCitaro | as eCitaro |
| **MAN 18 E** | 6/7/8 packs: 534/623/712 kWh; usable 90 %: 480/560/640 kWh (V2) | NMC; VW cells (V2, V11) | as 12 E (V2) | n/a | depot CCS up to 150 kW (V2) |
| **Solaris U18 electric** | about 700 kWh (V12) | n/a | n/a | n/a | n/a |
| New Flyer CHARGE NG 60 ft | 520/605 kWh (V9) | NMC (V9) | as 40 ft | as 40 ft | as 40 ft |

### 1c. Drive, inverter and measured energy use

| Type | Drive and motor | Inverter maker; semiconductor | Measured energy use (conditions) |
|---|---|---|---|
| **eCitaro 12 m** | to 2025 in CH: ZF AVE 130 axle, two wheel-hub asynchronous motors ZF AI300V2, 2 × 125/126 kW or 2 × 141 kW (V4, V10); 2026: ZF AxTrax 2 LF, 360/440 kW (V1) | AxTrax 2 LF: two ZF 800 V SiC inverters (`component_inverter.md`, S7); AVE 130 versions: n/a | none for this type; mixed German fleet with eCitaro 1.0-1.4 kWh/km, heating included (C5) |
| **MAN 12 E** | one central PSM motor (Scania/MAN MG4115-1), 160 kW continuous, 240 kW peak (V2, V4, V6) | n/a | none read (TÜV value of the 550.8 km run not read, V11) |
| **Solaris U12 electric** | ZF AVE 130 2 × 125 kW (2017-2020); ZF CeTrax 220 kW central (2022); TSA motors 270-320 kW (2024-2025) (V4, V5) | Medcom (V3, V5); SiC for the electric-axle version (V12) | 1.13 kWh/km over 235 km at 5-12 °C; 1.02 at 70 % load, 12 °C; 1.53 at full load, 5 °C, rain; incl. part of the charging losses (V5) |
| Irizar ieBus 12 m | Siemens 1DB2016 240 kW or Siemens/Accelera 1CS2016 290 kW (V4) | n/a | n/a |
| Volvo 7900 Electric | Volvo MDS2043A-200, 200 kW (V4) | n/a | n/a |
| VDL Citea SLF-E | Siemens 1DB2016, 160 kW (V4) | n/a | n/a |
| New Flyer CHARGE NG 40 ft | Siemens ELFA 3, central motor 160 or 209 kW (V9) | Siemens ("compact inverters"); semiconductor n/a (V9) | none read (range by Altoona protocol only, V9) |
| BYD 40 ft | BYD-TYC90A traction motor, 90 kW (C1) | n/a | 1.21 kWh/km, slow downtown shuttle, 2019 (C1) |
| Proterra Catalyst E2 | permanent-magnet motor, 220 kW peak (C2, C3) | n/a | 1.18 kWh/km, Los Angeles area, 2020 (C3); 1.44 kWh/km (1.64 with diesel heater), Duluth, cold and hilly (C2) |
| **eCitaro G** | AVE 130 on one or two axles, 252-564 kW (V4); AxTrax 2 LF (V1) | as eCitaro | n/a |
| **MAN 18 E** | two central motors on axles 2 and 3, 267 kW continuous, 400 kW peak (V2, V4) | n/a | n/a |
| **Solaris U18 electric** | TSA motors 270-320 kW, or ZF AVE 130 on two axles 500 kW (V4); 240 kW central motor (U18.75, V12) | n/a | n/a |
| New Flyer CHARGE NG 60 ft | Siemens drive with ZF AVE 130 centre axle, 280 kW (V9) | Siemens; n/a | n/a |
| *Reference only:* Caetano e.City Gold 12 m (VBZ test bus) | n/a | n/a | 0.74-0.91 kWh/km on five Zurich lines, May-Nov 2018, diesel heater (C4) |

## 2. Ranking and recommendation

### 2.1 How the ranking counts

Each component model needs a fixed list of inputs. A type scores one point for each input that a source
read here gives (½ point where the value is only partial, marked "~").

| Component | Inputs counted (points) |
|---|---|
| Battery (6) | chemistry; gross capacity; usable capacity or depth-of-discharge rule; pack or module layout (needed for a partial repair); cell maker; stated life or warranty |
| Tyres (4) | tyre size; empty mass; gross mass; axle loads |
| Inverter (3) | drive type and motor power; inverter maker; semiconductor (IGBT or SiC) |
| Energy use (1) | a measured kWh/km with its conditions |

### 2.2 Ranking by data completeness (14 points)

| Rank | Type | Battery /6 | Tyres /4 | Inverter /3 | Energy /1 | Total /14 | Main gaps |
|---|---|---|---|---|---|---|---|
| 1 | **eCitaro G 18 m** | 6 (usable 550 kWh from the VBG buses) | 4 | 3 for the AxTrax 2 LF build (1 for AVE 130 builds) | 0 | 13 | measured energy; inverter of the AVE 130 buses in service |
| 2 | **eCitaro 12 m** | 5 (usable n/a) | 4 | 3 for the AxTrax 2 LF build (1 for AVE 130 builds) | 0 | 12 | measured energy; usable energy; inverter of the AVE 130 buses |
| 3 | **Solaris Urbino 12 electric** | 4 (cell maker, life n/a) | 4 | 2.5 (Medcom; SiC only for the electric-axle version, ~) | 1 | 11.5 | cell maker; warranty; which inverter goes with which motor |
| 4 | **MAN Lion's City 12 E** | 5.5 (life only as a range claim, ~) | 4 | 1 | 0 | 10.5 | inverter maker and semiconductor; measured energy |
| 5 | **MAN Lion's City 18 E** | 5 (life n/a) | 3 (axle loads n/a) | 1 | 0 | 9 | as MAN 12 E; axle loads |
| 6 | BYD 40 ft (USA) | 4 | 1 | 1 | 1 | 7 | masses, tyres, inverter; no European data read |
| 6 | Proterra Catalyst E2 (USA) | 3 | 2 | 1 | 1 | 7 | chemistry, tyres, inverter; later history of the maker not checked (section 5) |
| 8 | New Flyer CHARGE NG 40 ft / 60 ft | 3 | 1 | 2 | 0 | 6 | masses, axle loads, measured energy |
| 9 | Irizar ieBus 12 m; Volvo 7900 Electric | 0 | 4 | 1 | 0 | 5 | battery (no maker document read) |
| 9 | Solaris Urbino 18 electric | 1 | 3 | 1 | 0 | 5 | battery details |
| 12 | Irizar 18 m; VDL Citea SLF-E | 0 | 3 | 1 | 0 | 4 | battery; empty mass (VDL) |

Not ranked: Iveco E-Way, Heuliez GX Elec, Ebusco 2.2 and 3.0, Yutong E12, BYD European models,
Gillig, Proterra ZX5, Hess lighTram battery buses (no maker document read; section 5). The rank of the
"n/a" types reflects the sources read here, not the makers' real documentation.

### 2.3 Recommendation

- **12 m: Mercedes-Benz eCitaro.** Reasons:
  - The battery data are the most complete of all types: chemistry, pack sizes and layout, the cell maker
    of each generation, a cycle-life claim and a guarantee (C6, V1, V10). The battery dossier already
    builds on them, including the repair offers (`component_battery.md`, B1).
  - The tyre inputs are complete: size (V1), axle loads (V1), and the registered empty mass of 211
    Swiss buses (V7), for example 13,450 kg for the seven 2023 buses that are probably those of line 759.
    This fills the empty-mass gap of `component_tires.md` (G6) and of `component_inverter.md`.
  - The drive of the 2026 build (ZF AxTrax 2 LF, two 800 V SiC inverters) is documented
    (`component_inverter.md`, S7). The buses in Swiss service up to 2025 have the ZF AVE 130 axle (V4),
    whose inverter is not documented in any source read. The inverter model stays a composite.
  - It runs on the case-study lines 759 and 768 (VBG), and 211 + 163 buses run in Switzerland (V7).
  - **Gap:** no measured energy use of this type was found. Use the Solaris Urbino 12 electric test (V5:
    13.9 t empty, ZF AVE 130 axle like the early eCitaro, 1.02-1.53 kWh/km at 5-12 °C) and the Zurich
    values of C4 to check the energy formula of `component_battery.md`.
- **Alternative 12 m: Solaris Urbino 12 electric**, if a measured energy use and a named inverter maker
  (Medcom) matter more than the cell maker and the battery life statements.
- **18 m: Mercedes-Benz eCitaro G.** It shares the battery packs, the tyre size and the drive family with
  the 12 m eCitaro, so one parameter set covers both lengths. Its usable energy (550 kWh) and empty mass
  (about 22,000 kg for the six 2025 buses that are probably those of line 768, V7) are known.
- **If the case study uses a VBZ garage** (Hardau or Hagenholz): the MAN Lion's City 12 E and 18 E are the
  local fleet (22 and 28 buses registered in Zurich city, V7). Their battery data are good (V2: packs,
  modules, usable-energy rule), but no source read names their inverter or measures their energy use.

## 3. Source entries

Sources are numbered V1, V2, ... ("V" for vehicle). The evidence basis is stated in each entry.
Downloads are in the scratchpad folder `downloads\bus\NN`; text extracts in `text_bus\`.

### V1. Daimler Buses, *The eCitaro. Technical data and equipment at a glance*, data sheet 04/2026

- **Reference.** Daimler Buses GmbH, international English edition, dated 04/2026 (p. 13), 13 pages.
  https://www.mercedes-benz-bus.com/int/en/models/ecitaro/facts-ecitaro.pdf. The copy read is the one
  downloaded by the battery agent on 9 Nov 2026 (scratchpad `downloads\bat\04\facts-ecitaro.pdf`).
- **Evidence basis.** Read in full (text layer, all 13 pages), 10 Nov 2026. The same document is B1e in
  `component_battery.md`, S6 in `component_tires.md` and S1 in `component_inverter.md`.
- **Nature.** Maker data sheet of the bus sold in 2026.

| Item | eCitaro 12 m (2 or 3 doors) | eCitaro G 18 m (3 or 4 doors) | Location |
|---|---|---|---|
| Length | 12,135 mm | 18,125 mm | p. 4 |
| Wheelbase | 5,900 mm | 5,900 mm + 5,990 mm | p. 4 |
| Tyre size | 275/70 R 22.5; option 315/60 R 22.5 | same | p. 4 |
| Drive | ZF AxTrax 2 LF electric axle; 440 kW peak, 360 kW continuous; 2 × 525 Nm; ratio 2 × 11.896, final 22.66; nominal voltage 650/750 V | same axle as the rear (third) axle; centre axle ZF AV 133 not driven | p. 6 |
| Gross vehicle weight | 20,000 kg | 30,000 kg (3 doors), 29,000 kg (4 doors) | p. 9 |
| Maximum axle loads | front 7,500 kg; drive 13,000 kg | front 7,500 kg; centre 11,500 kg (3 doors) or 10,000 kg (4 doors); drive 13,000 kg | p. 9 |
| Passenger capacity, maximum | 88 (seats 29 with 2 doors, 26 with 3 doors) | 146 (seats 45 or 41) | p. 8 |
| Battery | NMC4 modules of 111 kWh; 4 to 6 modules: 444, 555 or 666 kWh; 2 at the rear, up to 4 on the roof | NMC4, 4 to 7 modules: up to 777 kWh; 2 at the rear, up to 5 on the roof | p. 10 |
| Charging | Combo 2 (CCS) plug; options: pantograph on the bus, charging rails | same | p. 10 |
| Range | up to 600 km | up to 480 km | p. 8; footnote: internal test, maximum battery, new battery, typical city operation, moderate weather (p. 10) |

- **Not in the data sheet.** Empty (kerb) mass; usable energy; cell maker; warranty; charging power;
  inverter maker or semiconductor; energy use per km. The word "module" here means a pack of 111 kWh
  (B1c of `component_battery.md`: 15 modules of 12 prismatic cells per pack).
- **Limit.** The data sheet describes the 2026 build. The buses of line 759 (264 kWh) carry an older
  battery generation (NMC2 by the pack size, per `component_battery.md`, B1). The seven eCitaro of 2023
  that are probably those of line 759 are registered under approval 2MB425, which has the ZF AVE 130
  axle, not the AxTrax 2 LF (V4, V7).

### V2. MAN Truck & Bus, brochure *MAN Lion's City E. Elektromobilität mit Mehrwert*, 03/2026

- **Reference.** MAN Truck & Bus SE, München, document D112.596/d, 03/2026, 22 pages, German.
  https://media.man.eu/is/content/MAN/Brochure-MAN-Lions-City-E-DEpdf (downloaded 10 Nov 2026,
  13,451,100 bytes, `downloads\bus\06`).
- **Evidence basis.** Read in full (text layer, 22 pages).
- **Nature.** Maker brochure. Range values follow SORT (Standardised On-Road Test of the UITP, the
  International Association of Public Transport) "with air conditioning and heating and a
  pre-conditioned vehicle", for the 12 E "under optimal conditions" (p. 2 footnote). The brochure gives
  no SORT consumption value.

| Item | Lion's City 12 E | Lion's City 18 E | Location |
|---|---|---|---|
| Length | 12,185 mm | 18,060 mm | p. 20 (printed p. 21) |
| Gross vehicle weight (technical) | 20,000 kg | 30,000 kg | p. 20 |
| Seats, maximum | 44 | 60 | p. 20 |
| Battery chemistry; place | NMC; roof | NMC; roof | p. 20 |
| Packs, installed capacity | 4 packs 356 kWh; 5 packs 445 kWh; 6 packs 534 kWh (89 kWh per pack) | 6 packs 534 kWh; 7 packs 623 kWh; 8 packs 712 kWh | p. 9 (table) |
| Usable energy | "Reliable range": 65 % DoD (depth of discharge), so 231/289/347 kWh. "Maximum range": 90 % DoD, so 320/400/480 kWh | 65 %: 347/405/463 kWh; 90 %: 480/560/640 kWh | p. 9 |
| Pack layout | each pack holds 12 exchangeable modules; 4 to 7 packs by length (the table on p. 9 shows 8 packs for the 18 E) | same | p. 10 |
| Battery production | at MAN in Nürnberg since 2025 | same | p. 9 |
| Charging | depot (overnight), CCS plug, up to 150 kW DC; charge time < 4 h | same; < 5 h | p. 20 |
| Drive | one electric central motor, single-stage gear; 160 kW continuous, 240 kW peak | two driven axles, two central motors, electronic synchronisation; 267 kW continuous, 400 kW peak | p. 20 |
| Range | up to 500 km | up to 460 km | p. 20 |
| HVAC (heating, ventilation and air conditioning) | heat pump (CO2 refrigerant available) and electric heaters; auxiliary heater with biodiesel or HVO possible | same | p. 20 |
| Repair | battery-repair workplaces in selected MAN workshops | | pp. 18, 21 |

- **Not in the brochure.** Empty mass, axle loads, tyre size, cell maker, warranty or cycle life,
  inverter maker or semiconductor, energy use per km.
- **Limit.** The VBZ buses (22 Lion's City 12 E and 22 Lion's City 18 E, built 2022, see
  `zurich_depot_and_lines.md`) are an older battery generation: VBZ states 480 kWh (12 E) and 640 kWh
  (18 E). These numbers equal the 90 % usable values of the 6-pack and 8-pack builds above; whether VBZ
  quotes installed or usable energy is not stated in the sources read.

### V3. Solaris Bus & Coach, *Solaris Urbino 12 electric – technical specification*, press sheet, UITP 2023

- **Reference.** Solaris Bus & Coach S.A., press release sheet for the UITP Summit 2023, 1 page.
  https://www.solarisbus.com/public/assets/Biuro_prasowe/2023_05_11_UITP_zapro/Technical_details_Solaris_Urbino_12_electric.pdf
  (downloaded 10 Nov 2026, 213,040 bytes, `downloads\bus\05`).
- **Evidence basis.** Read in full (1 page).
- **Values.** Length 12,000 mm; wheelbase 5,900 mm. Front axle ZF RL 82 EC; drive axle ZF AV 133.
  Motor ZF CeTrax, 220 kW (a central motor). "Electric drive: Medcom" (Medcom is a Polish maker of
  traction inverters). Charging by plug up to 260 kW, by pantograph up to 450 kW. Battery "Solaris High
  Energy: 420 kWh" (chemistry not stated). GVW 20,000 kg. Up to 77 passengers, 21 seats.
- **Not in the sheet.** Empty mass, axle loads, tyre size, chemistry, pack layout, cell maker, warranty,
  energy use per km, inverter semiconductor.

### V4. ASTRA (Swiss Federal Roads Office), type-approval base data `TG-Automobil.txt`, state of 30 Jul 2026

- **Reference.** Bundesamt für Strassen ASTRA, open data "Basisdaten Typengenehmigungen ab 1995",
  file `TG-Automobil.txt` (tab-separated, Windows-1252), last modified 2026-07-30 (server listing).
  https://opendata.astra.admin.ch/ivzod/2000-Typengenehmigungen_TG_TARGA/2200-Basisdaten_TG_ab_1995/TG-Automobil.txt
  Field list: *Basisdatensätze Typengenehmigungen*, 12 pages,
  https://opendata.astra.admin.ch/ivzod/2000-Typengenehmigungen_TG_TARGA/2200-Basisdaten_TG_ab_1995/2220-Datenbeschreibung/Basisdaten_TG_ab_1995.pdf
  (both downloaded 10 Nov 2026; `downloads\bus\09`, 322,893,637 bytes; `downloads\bus\11`).
- **Evidence basis.** Dataset downloaded and counted: 210,673 type approvals; 2,717 of kind
  *Gesellschaftswagen* (bus) or *Gelenkbus* (articulated bus); 90 of them with an electric motor: 83
  with fuel code "E", and 7 with another fuel code whose motor field names an electric motor (for
  example the Mercedes O 530 BH hybrid of 2011, fuel code "X"). All 90 records extracted and read (script `scripts_bus\tg_buses.py`; output
  `text_bus\tg_ebus_full.txt`). Field list read in part (pp. 1-7, the fields used).
- **Nature.** Official Swiss type approval (*Typengenehmigung*). It states the approved ranges, not one
  vehicle. Its fields: motor maker and type, power, 30-minute power, approved gross weight
  (*Garantiegewicht*), axle guarantees, tyres. Brands with an electric bus approval: Mercedes-Benz 22,
  Irizar 15, Solaris 12, Hess 11, Iveco 9, MAN 5, Karsan 3, Renault 3, VDL 3, Scania 2, Volvo 2, one each
  Altas Auto, Van Hool, Zhongtong. **No approval for BYD, Ebusco or Yutong buses.**
- **Empty mass.** The field *Leergewicht* (empty weight) is empty in every Mercedes-Benz, MAN, Solaris,
  Irizar, VDL and Volvo record read. The eCitaro record says why: fields 07/36/37/51/**52**/56 are fixed
  "bei der Einzelabnahme" (at the inspection of each vehicle) (2MB442, remark Z18). Hess gives only wide
  ranges (below).

Values by type (TG = type-approval number, with the date of approval).

| Type | TG, date | Motor maker, type; power (peak / 30 min) | Gross weight approved | Axle loads approved | Tyres |
|---|---|---|---|---|---|
| Mercedes-Benz eCitaro 12 m | 2MB339 2019-07-03; 2MB425/426 2022-05-17; 2MB442/443 2023-10-02 | ZF AVE 130 axle; wheel-hub motors ZF AI300V2B1 (2 × 126 kW = 252 kW; 30 min 2 × 87 kW = 174 kW) or AI300V2A2 (2 × 141 kW = 282 kW) | 17,500-20,000 kg | front 6,400-8,000 kg; rear 11,100-13,000 kg | 275/70R22.5 all axles; 315/60R22.5 option front |
| Mercedes-Benz eCitaro G 18 m | 2MB385/386 2020-05-26; 2MB427-430 2022; 2MB444-447 2023; 2MB491-494 2024-11-08 | ZF AVE 130; one driven axle 252 or 282 kW, or two driven axles 504 or 564 kW | 25,000-30,000 kg | not filled | 275/70R22.5 all axles |
| MAN Lion's City 12 E ("12C, Elektro") | 2MA386 2020-04-30; 2MA453 2023-10-31; 12 LE: 2MA461 2025-07-01 | SCANIA/MAN MG4115-1, central motor; 245 kW (2020) or 240 kW; 30 min 160 kW | 17,900-20,000 kg | front 6,300-8,000 kg; rear 11,500-13,000 kg | 275/70R22.5 |
| MAN Lion's City 18 E ("18C") | 2MA398 2021-02-09 | SCANIA/MAN MG4115-1, two central motors on axles 2 and 3; 400 kW; 30 min 267 kW (107 + 160 kW) | 30,000-31,000 kg | not filled | 275/70R22.5 |
| Solaris Urbino 12 electric | 2SC149 2017-10-25 (ZF AVE 130, 220 kW, 30 min 160 kW); 2SC169 2022-09-30 (ZF CeTrax CX 220, 220 kW, 30 min 148 kW); 2SC176 2024-05-08 (TSA TMF 35-44-4, 270 kW, 30 min 240 kW); 2SC178 2025-06-19 (TSA TMPW 40A-26-8, 320 kW, 30 min 240 kW) | as left | 18,000-20,000 kg (2017, 2022); 19,845-20,000 kg (2024, 2025) | front 7,100-7,700 kg (2017) or 7,100/7,245-8,165 kg; rear 11,500-12,600 kg | 275/70R22.5 all; 315/60R22.5 option front |
| Solaris Urbino 18 electric | 2SC150 2018-01-29 (TSA TMF 35-44-4, 270 kW); 2SC172 2023-06-05 (ZF AVE 130 on two axles, 500 kW, 30 min 4 × 95 kW); 2SC179 2025-06-19 (TSA TMPW 40A-26-8, 320 kW) | as left | 28,000-30,000 kg; 30,000 kg | not filled | 275/70R22.5; options 315/60R22.5 front, 455/45R22.5 axle 2 |
| Hess lighTram battery bus (BT-N1D, 9.5-13.5 m) | 2HB207 2022-04-22 | Siemens 1DB20* / 1CS2*; 230 kW; 30 min 176 kW | 21,200 kg | front 8,200 kg; rear 13,000 kg | 275/70R22.5 or 305/70R22.5 |
| Hess lighTram articulated (BGT, 17.5-18.75 m) | 2HB208/209 2022-08-04; 2HB212-215 2026-04-27 | ABB AMXM250 (190 or 2 × 190 kW; 30 min 176 or 352 kW) or Siemens/Accelera 1CS2 (230 or 460 kW) | 26,000-34,200 kg | not filled | 275/70R22.5; options 305/70 or 315/70R22.5 front; twin tyres on axles 2 and 3 |
| Irizar ieBus / ieTram 12 m | 2IA309 2021-08-05; 2IA312 2022; 2IA319 2025-07-24; 2IA321 2026-02-12 | Siemens 1DB2016 (240 kW) or Siemens/Accelera 1CS2016 (290 kW) | 19,000-21,165 kg | front 7,500-8,165 kg; rear 13,000 kg | 275/70R22.5 |
| Irizar ieBus / ieTram 18 m | 2IA311 2021-07-06 (Alconza PMSM, 235 kW); 2IA313 2023-03-01 (Irizar e-mobility, 265 kW); 2IA317 2024-12-04 (Siemens 1CS2022, 375 kW) | as left | 29,000-31,165 kg | not filled | 275/70R22.5; twin tyres on axles 2 and 3 (2024) |
| VDL Citea SLF-E 120 (12 m) | 2VD537, 2VD539 2018 | Siemens 1DB2016; 160 kW; 30 min 116 or 153 kW | 19,000 kg | front 7,245 kg; rear 12,000 kg | 275/70R22.5 |
| VDL Citea SLFA-E 181 (18 m) | 2VD548 2018-11-27 | Siemens 1DB2022; 240 kW; 30 min 210 kW | 29,000 kg | not filled | 275/70R22.5 |
| Volvo 7900 Electric (10-13.5 m) | 2VA984 2021-09-22 | Volvo MDS2043A-200; 200 kW; 30 min 166 kW | 19,100-19,500 kg | front 7,100-7,500 kg; rear 12,000 kg | 275/70R22.5 |
| Volvo 7900 Electric Articulated | 2VA985 2021-09-22 | Volvo MDS2043A-200 × 2; 400 kW; 30 min 332 kW | 30,000 kg | not filled | 275/70R22.5 |

- **Hess empty-mass ranges.** BT-N1D (solo, 9.5-13.5 m): 9,500-16,000 kg; BGT (articulated): 14,500-25,000 kg
  (2HB207-209). These ranges cover all builds of the approval; they do not give the mass of one bus.
  The Hess articulated approvals may cover trolleybuses as well as battery buses (see V7).
- **What the data show for the inverter question.** The eCitaro buses approved in Switzerland up to
  2025 carry the **ZF AVE 130** axle with ZF AI300V2 wheel-hub motors (remark of 2MB442: "bei beiden
  Radnabenmotor der Antriebsachse"), not the AxTrax 2 LF of V1. No approval names an inverter maker.
- **Twin tyres.** The field "Achsen/Räder" counts axles and wheel positions ("2/4", "3/6"). Some records
  add a note on twin tyres: the Mercedes O 530 BH hybrid of 2011 (2MB178, "h = Zwillingsbereifung", twin
  tyres at the rear), the Hess articulated buses (axles 2 and 3) and the Irizar 18 m of 2024-2026 (axles 2
  and 3). The eCitaro, MAN and Solaris 12 m records carry no such note. The records read do not settle
  whether their drive axles carry twin tyres.

### V5. bus blickpunkt 05/2020, *Im Bus Blickpunkt-Test: Solaris Urbino 12 Electric*

- **Reference.** W. Tschakert, bus blickpunkt (German trade magazine for bus operators), issue 05/2020,
  p. 11 (one page). https://www.busnetz.de/files/11_BUS_Mai_2020_Bustest.pdf (downloaded 10 Nov 2026,
  503,289 bytes, `downloads\bus\12`).
- **Evidence basis.** Read in full (one page, text layer).
- **Nature.** A magazine road test of one bus of the operator Ettenhuber (Munich region), with a
  Solaris energy meter at the battery output. Not peer-reviewed; the measuring method is described only
  briefly.
- **Vehicle.** ZF AVE 130 portal axle with two wheel-near asynchronous motors, 2 × 125 kW nominal,
  2 × 60 kW continuous, 400 V nominal; "Traktionselektrik und Elektronik von Medcom" (traction
  electrics and electronics by Medcom). NMC battery of 6 modules × 50 kWh = 300 kWh nominal; "about
  80 %" used, 240 kWh. CCS depot charging, up to 150 kW (operator: 100 kW). Heating: 25 kW electric
  boiler, convectors, diesel auxiliary heater; roof air conditioning.
- **Masses.** "Leergewicht lt. Hersteller" (empty mass according to the maker) **13,900 kg**; gross
  18,745 kg; front axle 7.245 t, rear axle 11.5 t; tyres 275/70 R 22.5 all round. Seats 33 (+2 folding),
  standing 37.
- **Measured energy use** (battery output, "traction plus air conditioning", includes a share of
  charging losses):

| Run | Conditions | kWh/km |
|---|---|---|
| Whole test, 235.4 km | 5-12 °C, partly heavy rain | 1.126 |
| City route, full load (test mass 18,740 kg) | about 5 °C, rain; mean speed 20.50 km/h | 1.525 |
| City route, about 70 % load (17.4 t) | about 12 °C, dry; mean speed 21.09 km/h; heating on | 1.023 |
| Constant 50 km/h | air conditioning and heating on | 0.848 |

### V6. bus blickpunkt 12/2021, bus test *MAN Lion's City 10 E*

- **Reference.** bus blickpunkt, issue dated 12/2021 in the page header (the file name says February,
  the print stamp 14 Feb 2023), p. 22 (one page). https://www.busnetz.de/files/21_BUS_Feb_BustestTWA.pdf
  (downloaded 10 Nov 2026, 7,569,953 bytes, `downloads\bus\13`).
- **Evidence basis.** Read in full (one page, text layer).
- **Use here.** The 10.5 m bus, not a 12 m or 18 m type, but it shares the battery and drive of the
  Lion's City 12 E (same platform, "Baukasten des Lion's City 12 E").
- **Values.** Central motor MAN GE21F1, permanent-magnet synchronous (PSM), oil-spray cooled,
  160/240 kW, 2,100/3,500 Nm; the motor is supplied by Scania. Batteries: water-cooled modular
  Volkswagen lithium-ion NMC packs, type 36.26615-6007, on the roof, **550 kg per pack**, 4 or 5 packs,
  320 or 400 kWh; "Mindest-Reichweite [...] über die gesamte Lebensdauer (bis zu 12 Jahre)" (minimum range
  over the whole life, up to 12 years) – a maker claim. CCS depot charging 22-150 kW. Empty mass "k.A."
  (not stated); gross 19.5 t.

### V7. ASTRA, Swiss vehicle stock `BEST_ANDERE.txt`, state of 1 Nov 2026

- **Reference.** Bundesamt für Strassen ASTRA, IVZ (*Informationssystem Verkehrszulassung*, the Swiss
  vehicle register) open data, monthly stock data set, file `BEST_ANDERE.txt` (vehicles other than
  passenger cars, motorcycles and trailers), column "Datenstand" 01.11.2026.
  https://opendata.astra.admin.ch/ivzod/1000-Fahrzeuge_IVZ/1300-Fahrzeugbestaende/1320-Datensaetze_monatlich/BEST_ANDERE.txt
  (downloaded 10 Nov 2026, 287,325,788 bytes, `downloads\bus\16`).
- **Evidence basis.** Dataset downloaded and counted: 933,440 vehicle records; 10,259 buses
  (*Gesellschaftswagen* 7,945, *Gelenkbus* 2,314); **1,185 buses with fuel "Elektrisch"** (hybrids
  "Diesel / Elektrisch" excluded; the register has no field that separates trolleybuses). Script `scripts_bus\best_buses.py`;
  outputs `text_bus\best_ebus_rows.tsv` (one row per bus) and `text_bus\best_ebus_stats.txt`.
- **Nature.** One record per registered vehicle, with its type approval, seats, standing places,
  **registered empty mass** (*Leergewicht*), registered gross mass, year of first registration, canton
  and a masked postcode ("80.." = 80xx). No operator name.
- **Definition of the empty mass** (V8): ready-to-use unladen vehicle with coolant, lubricant, at least
  90 % fuel, the optional equipment, and **the driver counted as 75 kg** (VTS Art. 7 Abs. 1).

Registered empty mass of battery-electric city buses (12 m and 18 m), by type approval (TG of V4).
"n" = buses in the stock; mass as minimum / median / maximum over these buses.

| Type | TG | n | Empty mass min / median / max (kg) | Seats + standing (most common) | First registration | Cantons (most common) |
|---|---|---|---|---|---|---|
| Mercedes eCitaro 12 m | 2MB339 | 21 | 13,700 / 14,500 / 14,600 | 73 + 42 | 2019-2022 | BL 15 |
| Mercedes eCitaro 12 m | 2MB425 | 44 | 13,450 / 14,280 / 15,250 | 77 + 47 | 2022-2023 | BS 19, AG 10, ZH 7, GE 7 |
| Mercedes eCitaro 12 m | 2MB442 | 71 | 13,800 / 14,400 / 14,960 | 71 + 44 | 2023-2026 | GE 20, BE 10, VS 9 |
| Mercedes eCitaro 12 m | 2MB443 | 75 | 13,800 / 14,800 / 15,100 | 70 + 41 | 2025-2026 | BL 19, VS 16, ZH 11 |
| Mercedes eCitaro G 18 m | 2MB427 | 46 | 20,150 / 20,150 / 20,500 | 133 + 94 | 2022-2023 | BS 38, ZG 8 |
| Mercedes eCitaro G 18 m | 2MB447 (two driven axles) | 73 | 19,800 / 20,900 / 22,100 | 134 + 92 | 2025-2026 | AG 23, ZG 17, FR 14, ZH 12 |
| MAN Lion's City 12 E | 2MA386 | 33 | 14,750 / 14,900 / 15,100 | 76 + 44 | 2022-2023 | ZH 22, SO 5, VD 5 |
| MAN Lion's City 12 E | 2MA453 | 9 | 14,570 / 14,570 / 15,000 | 80 + 44 | 2024-2025 | GE 8 |
| MAN Lion's City 18 E | 2MA398 | 32 | 21,950 / 22,600 / 22,600 | 113 + 70 | 2021-2026 | ZH 28, SO 4 |
| Solaris Urbino 12 electric | 2SC169 (2022) | 9 | 14,200 / 14,200 / 14,210 | 74 + 38 | 2022 | LU 7 |
| Solaris Urbino 12 electric | 2SC178 (2025) | 100 | 14,470 / 14,740 / 15,370 | 63 + 38 | 2025-2026 | BL 25, SG 24, TG 13 |
| Solaris Urbino 18 electric | 2SC179 | 62 | 20,670 / 21,380 / 22,340 | 121 + 79 | 2025-2026 | AG 28, LU 9, ZH 8 |
| Hess lighTram articulated (BGT-N2D), **battery bus or trolleybus** (see below) | 2HB209 | 43 | 19,260 / 19,470 / 19,750 | 141 + 100 | 2022-2024 | NE 18, VD 14, SG 11 |
| Irizar ieBus 12 m | 2IA319 | 26 | 14,500 / 15,050 / 15,050 | 30 + 36 (mostly) | 2025-2026 | BE 19 |
| Irizar ieTram 18 m | 2IA320 | 7 | 20,700 / 20,770 / 20,770 | 131 + 90 | 2025-2026 | SH 6 |
| Volvo 7900 Electric | 2VA984 | 4 | 13,400 (all) | 77 + 39 | 2022 | VD 4 |
| VDL Citea SLFA-E 181 (18 m) | 2VD548 | 1 | 20,800 | 117 + 77 | 2019 | BS 1 |

- **Hess articulated buses: battery bus or trolleybus.** The register has no field that separates
  trolleybuses, which also run on electricity, and the Hess approvals do not say whether a vehicle has
  a trolley pole. Hess sells
  the name lighTram for both kinds (search results name "lighTram 19 DC" trolleybus flyers for Zurich
  and Salzburg; not read). The Hess articulated rows of V4 and V7 are therefore **not used** as
  battery-bus data. The Mercedes, MAN, Solaris, Irizar, VDL and Volvo approvals of V4 name battery-bus
  models (no trolley equipment in their records).
- **Seats + standing** are the registered places; they differ by operator layout. The registered gross mass of two-axle buses is mostly
  19,500 kg, the Swiss legal limit (VTS Art. 95; see `component_tires.md`, S4), even where the type
  approval allows 20,000 kg.
- **Case-study buses (inference from canton, postcode prefix, year and count; the register names no
  operator).**
  - Zurich city (postcode 80..): 22 MAN Lion's City 12 E (15 first registered 2022 at 14,900 kg; 7 in
    2023 at 14,750 kg) and 28 MAN Lion's City 18 E (2021-2024; 22,400-22,600 kg). VBZ states 22 and 22
    vehicles but lists 28 vehicle numbers for the 18 E (`zurich_depot_and_lines.md`, section 1). Also
    15 Hess lighTram BT-N1D (10.8 m class) at 12,350-12,450 kg, matching the 15 VBZ lighTram 10 Plug.
  - Postcode 83.. (this includes Bassersdorf, 8303): 1 eCitaro (2MB339, 2021, 13,700 kg) and 7 eCitaro
    (2MB425, 2023, 13,450 kg), matching the VBG line 759 pilot bus of 2021 and the 7 buses of Feb 2023;
    6 eCitaro G (2MB447, 2025, 22,000 kg), matching the 6 buses of line 768 inferred in
    `zurich_depot_and_lines.md`; 11 eCitaro (2MB443, 2026, 14,900 kg), not yet in that file.
- **BYD, Ebusco, Yutong.** No BYD bus is registered as an electric bus; 2 "EBUSCO 01/11A2L" (2018, 13,160
  kg, canton VS) and 1 Yutong "T14E" (2026) are registered without a Swiss type approval.

### V8. Swiss ordinance VTS (SR 741.41), Art. 7, version of 1 Oct 2026

- **Reference.** Verordnung über die technischen Anforderungen an Strassenfahrzeuge (VTS), SR 741.41,
  consolidated German version of 1 Oct 2026, Fedlex XML (URL in `component_tires.md`, S4); downloaded
  10 Nov 2026 (`downloads\bus\17`).
- **Evidence basis.** Read in part: Art. 7 in full (script `scripts_bus\vts_article.py`).
- **Values.** Art. 7 Abs. 1: "Leergewicht" is the weight of the ready-to-use unladen vehicle with coolant,
  lubricant and fuel (at least 90 %), including optional equipment, special accessories and the driver,
  "dessen [...] Gewicht mit 75 kg angenommen wird" (whose weight is taken as 75 kg). Abs. 3:
  "Garantiegewicht" is the technically permissible maximum mass, equal to the EU "Gesamtmasse". Abs. 4:
  "Gesamtgewicht" is the mass that counts for the registration, the highest mass with which the vehicle
  may run.
- **Use here.** The registered empty masses of V7 include a 75 kg driver. The empty mass without driver
  is 75 kg lower.

### V9. New Flyer, *Xcelsior CHARGE NG* spec sheet, model year 2024 (file dated 29 Jan 2024)

- **Reference.** New Flyer of America, brochure and specification sheet "Xcelsior CHARGE NG", 7 pages,
  file name "AL-MY24-Xcelsior-CHARGE-NG-Spec-Sheet-240129", hosted by the California HVIP programme
  (Hybrid and Zero-Emission Truck and Bus Voucher Incentive Project).
  https://californiahvip.org/wp-content/uploads/2020/09/AL-MY24-Xcelsior-CHARGE-NG-Spec-Sheet-240129.pdf
  (downloaded 10 Nov 2026, 20,126,247 bytes, `downloads\bus\19`). The model-year 2026 sheet named by a
  search (same site) answered HTTP 404.
- **Evidence basis.** Read in full (text layer, 7 pages; the table of p. 6 also read from a rendering,
  because the text layer jumbles it).
- **Values (40 ft bus, about 12.2 m; and 60 ft, about 18.3 m).**
  - Length 40 ft: 12.24 m over body, 12.50 m over bumpers; 60 ft: 18.29 m and 18.54 m (p. 6).
  - Tyres 305/70R22.5, all lengths (p. 6).
  - Drive: "Siemens electric drive system", ELFA 3 traction system "with compact inverters and embedded
    drive controllers" (p. 3). 40 ft: central motor, 160 kW standard (1,400 lb-ft) or 209 kW
    high-gradeability option (2,000 lb-ft). 60 ft: ZF AVE 130 in-wheel motor centre drive axle,
    280 kW (p. 6). The sheet does not name the semiconductor type.
  - Battery: NMC (p. 3). Energy storage 40 ft: 345, 435 or 520 kWh; 60 ft: 520 or 605 kWh (p. 6). Range
    40 ft up to 258 miles (about 415 km) with 520 kWh, "per FTA Altoona test protocol - HVAC off" (p. 4).
    Depot charge of 520 kWh in 3.8 h; on-route pantograph charging up to 450 kW (p. 4).
  - "12-Year comprehensive warranty available on batteries, inverters and electric motors" (p. 4) – a
    maker offer; no end-of-life criterion given.
  - Capacity 40 ft: up to 40 seats and 44 standees (p. 6).
- **Not in the sheet.** Curb (empty) mass, gross mass, axle loads, cell maker, pack layout, measured
  energy use.

### V10. bus blickpunkt 03/2019, *Im Bus Blickpunkt-Test: Mercedes-Benz eCitaro*

- **Reference.** W. Tschakert, bus blickpunkt, issue 03/2019, p. 16 (one page).
  https://www.busnetz.de/files/Maerz_2019_Bustest.pdf (downloaded 10 Nov 2026, 1,769,142 bytes,
  `downloads\bus\22`).
- **Evidence basis.** Read in full (one page, text layer).
- **Values (first eCitaro generation, NMC1).** ZF AVE 130 portal axle, two wheel-near asynchronous
  motors, 2 × 125 kW, 2 × 485 Nm. Lithium-ion NMC modules by **Akasol**, up to 10 modules of 25 kWh,
  at most 243 kWh; depot plug charging; range about 150 km (> 200 km with ventilation only). Heat pump
  and CO2 air conditioning (Konvekta), 20 kW heating. Front axle ZF RL 82 EC, 8.2 t, tyres 315/60 R 22.5;
  drive axle ZF AVE 130, 13 t, tyres 275/70 R 22.5. **Empty mass ("Leergewicht") 13,400 kg**; gross
  19,500 kg; 80-85 passengers. The inverter ("Wechselrichter") sits in the rear tower with battery
  packs, braking resistor and cooling unit.
- **Not measured.** The test was a short drive; no energy use is given.

### V11. MAN Truck & Bus, press release *MAN sets the standard for range: Fully electric bus breaks the 550 kilometre barrier*, 2021

- **Reference.** MAN Truck & Bus press portal, English, about the run of 18 May 2021.
  https://press.mantruckandbus.com/corporate/man-sets-the-standard-for-range-fully-electric-bus-breaks-the-550-kilometre-barrier/
  (downloaded 10 Nov 2026, `downloads\bus\23`).
- **Evidence basis.** Read in full (web text).
- **Values.** A Lion's City 12 E ran 550.8 km in 24 h without charging on Munich lines 176 and X80,
  stopping at every stop, "partially loaded in spring-like temperatures on a relatively flat
  urban/inter-urban route"; TÜV SÜD sealed the charging socket and confirmed the distance. The battery
  uses "battery cell technology developed by the Volkswagen Group"; the range "is up to 270 km over the
  entire service life of the batteries" (maker claim).
- **Not in the release.** The battery energy used and the installed capacity. A search snippet says
  that the TÜV SÜD confirmation gives 363.3 kWh drawn from a 480 kWh battery; that document could not
  be read (section 4), so no consumption is taken from it.

### V12. Solaris Bus & Coach, press releases of 12 Oct 2022 (Transexpo) and 17 Dec 2024 (battery passport)

- **References.** *Solaris presents its latest battery solutions at Transexpo 2022*, 3 pages,
  https://solarisbus.com/en/press/pdf/1847; *Solaris delivers the world's first electric bus with a
  battery passport*, 2 pages, https://www.solarisbus.com/en/press/pdf/2257 (both downloaded 10 Nov 2026,
  `downloads\bus\26`, `\27`).
- **Evidence basis.** Both read in full.
- **Values.** Urbino 12 electric with "over 520 kWh" Solaris High Energy batteries, range "around 300
  km"; "an electric axle with two integrated electric motors of 125 kW each. The propulsion system was
  manufactured using energy-saving **SiC** technology" (2022, p. 1). More than 800 Urbino 12 electric in
  European service (2022, p. 1). Urbino 18.75 electric: 240 kW central motor, over 520 kWh, 138
  passengers; 183 ordered by Unibuss, Oslo (2022, p. 2). Urbino 18 electric for BVG Berlin with about
  700 kWh, 50 buses ordered (2024, p. 1). Solaris has delivered over 5,000 zero-emission vehicles
  (2024).
- **Not in the releases.** Chemistry, cell maker, warranty, empty mass, energy use.

### V13. Truck & Bus Builder, *T&BB report from Omnibus E-Bus Test 2025 in Bonn*, 13 May 2025

- **Reference.** B. Osborne, Truck & Bus Builder (trade web magazine), 13 May 2025.
  https://truckandbusbuilder.com/article/2025/05/13/tbb-report-from-omnibus-e-bus-test-2025-in-bonn
  (downloaded 10 Nov 2026, `downloads\bus\15`).
- **Evidence basis.** Read in full (web text).
- **Values.** The new Urbino 12 electric has batteries on the roof and at the rear (up to four at the
  rear) and a motor by Traktionssysteme Austria (TSA), "shedding up to 100 kilograms"; range over 600 km
  (Solaris claim). "Mercedes said it was shifting its entire eCitaro family to ZF's AxTrax 2". The
  article gives **no measured energy use** of any bus; the Omnibusspiegel test results are not public.

### Carried entries (read by other agents of this case study; I re-read only the parts named)

- **C1. NREL, Long Beach Transit BEB evaluation, final report, 2020** (R6 of
  `maintenance_focus_reports.md`; read in full there). I re-read Table 1 (printed p. 3) and Table ES-1
  (scratchpad `txt_75582.txt`). **BYD 40-ft (12.3 m) bus, model 6120LGEV (2017)**: GVWR (gross vehicle
  weight rating) 43,431 lb (19,700 kg); LFP battery by BYD, 324 kWh, upgraded to 360 kWh; one BYD-TYC90A
  traction motor, 90 kW; 50 kW on-board inductive charging. Energy use **1.94 kWh/mile = 1.21 kWh/km**
  in 2019 (Table ES-1), on a slow downtown shuttle (8.1 mph, 13 km/h, per R6). Battery warranty: 70 % of a
  base capacity for 12 years; base = 80 % of 324 kWh (p. 3).
- **C2. NREL and CTE, Duluth Transit Authority BEB evaluation, 2022** (R9; read in full there). I
  re-read Table 1 (printed p. 4) and Table ES-1 (`txt_83038.txt`). **Proterra Catalyst 40 ft (2018)**:
  GVWR 42,000 lb (19,050 kg); 440 kWh nominal, 354 kWh usable; 220 kW peak permanent-magnet motor;
  50 kW depot chargers. Energy use **2.32 kWh/mile = 1.44 kWh/km** from the chargers, **2.64 kWh/mile =
  1.64 kWh/km** with the diesel of the auxiliary heaters; cold and hilly city. Battery warranty bought
  by the operator: down to 80 % of the initial usable capacity (printed p. 35).
- **C3. NREL, Foothill Transit BEB evaluation, final report, 2021** (R1; read in full there). I re-read
  Tables 2 and 3 (`text_rep\01_nrel_foothill_final.txt`). **Proterra Catalyst E2, 42 ft (2017)**: GVWR
  42,000 lb / **curb weight 31,360 lb (14,225 kg)**; 440 kWh; permanent-magnet motor 220 kW peak.
  Proterra BE35 (35 ft, 2014): 37,320 / 27,680 lb; lithium-titanate (LTO) by Altairnano, 88 kWh; UQM PP220
  motor. Energy use of the E2 fleet **1.90 kWh/mile = 1.18 kWh/km** (results table, about 21 mph mean
  speed, 2020).
- **C4. VBZ 2019, *Elektrisch in die Zukunft*** (R2 of `maintenance_focus_reports.md`; C3 of
  `component_battery.md`). Caetano e.City Gold 12 m, 150 kWh: 0.74-0.91 kWh/km on five Zurich lines
  (May-November 2018, diesel auxiliary heater); the steepest line (37, 3.8 % mean grade) the highest.
  Not one of the candidate types, but the only measured energy use on Zurich lines.
- **C5. ThEGA 2025** (R11; C8 of `component_battery.md`). 32 buses in Thuringia, mostly 12 m, among them
  eCitaro and MAN Lion's City 12 E: about 1.2 kWh/km (range 1.0-1.4) with heating (p. 8); no split by
  type.
- **C6. Daimler Buses battery documents B1a-B1d** (`component_battery.md`, read in full there): NMC1
  generation in series from 2018; eCitaro NMC3 packs of 98 kWh with Akasol 21700 cylindrical cells (600 cells per module, 9 modules per pack);
  NMC4 packs of 111 kWh with prismatic BMZ Poland cells (15 modules of 12 cells); "Akasol assumes 4,000
  charging cycles" (maker claim); NMC4 guarantee of ten years, up to 15 years optional; more than 3,000
  eCitaro delivered by Oct 2025.

## 4. Could not read

| Item | Reason (10 Nov 2026) | Why it looks worthwhile |
|---|---|---|
| TÜV SÜD, confirmation of the "MAN eBus Efficiency Run" (18 May 2021), `TUeV_Bestaetigung_MAN_eBus_Efficiency_Run_DE.pdf` on man.eu | HTTP 404 under the `/hr/` path named by a search and under six other language paths (empty folders `downloads\bus\21`, `25_*`) | A search snippet says it gives the battery energy drawn (363.3 kWh) and the installed capacity (480 kWh): a measured kWh/km of the Lion's City 12 E under stated conditions. Can you retrieve it? |
| MAN data sheet `man_datenblatt_lionscity_12e.pdf` (older Lion's City 12 E generation, as delivered to VBZ in 2022) | man.eu answered HTTP 404 (`downloads\bus\04` empty); the copy on bahnbus.lima-city.de is an HTML page, not the PDF (`downloads\bus\07`) | Battery of the 2022 build (480 kWh), empty mass, tyres. Can you retrieve it? |
| Altoona bus test reports (Penn State Bus Research and Testing Center), for example New Flyer XE40, bus 2022-04 | The report is served only through a form button (ASP.NET postback); not submitted | Measured curb weight, axle loads, tyres and energy use on standard cycles for the US types |
| New Flyer spec sheet model year 2026 (`AL-MY26-Xcelsior-CHARGE-NG-35-40-60-Spec-Sheet-260304.pdf`) and the XE60 sheet on havis.com | HTTP 404 and HTTP 403 | Newer values for the US types; V9 (MY24) was read instead |
| swisscarinfo.ch full data sheets | WebFetch got HTTP 403; the page served to a script shows only a summary; the full sheet needs an account (not created) | Not needed: V4 and V7 give the same official data |
| Omnibusspiegel E-Bus-Test results (Bonn, 2017-2025) | Not public; the press reports read (V13) give no measured values | The only multi-type test under one protocol in Europe |

## 5. Leads not followed

- **Daimler Truck press releases** "Technical data for Mercedes-Benz eCitaro with NMC 3 batteries" and
  "Mercedes-Benz eCitaro with NMC 3 batteries: The test vehicle in detail" (daimlertruck.com newsroom,
  2023): may give the NMC3 empty mass and the axle of the 2023 buses.
- **Eurotransport, "Mercedes-Benz eCitaro: Auf Stuttgarts Straßen wurde hart getestet"**: a search
  snippet gives 116.5 kWh/100 km on a test route; the battery generation is not stated. Not read.
- **Finnish operator report on eCitaro NMC3** (boy.fi, "Akkuaskel pidemmälle tehokkaammin", PDF):
  ranges above 300 km. Not read.
- **New Flyer competitive comparison sheet** (nfiportal.nfigroup.com, 2017): Altoona energy values of
  the XE40 and competitors (maker document). Not read.
- **City of Winnipeg tender 763-2023, Appendix A**: maker planning values for the New Flyer XE40 and
  XE60 (curb weight, battery). Not read.
- **Hess flyers** (hess-ag.ch): "lighTram 12 Plug BGU", "lighTram 19 DC Zürich", "lighTram 25 DC
  BERNMOBIL"; BVB Basel lighTram sheet. Would separate the Hess battery buses from the trolleybuses in
  V4 and V7 and give the battery of the VBZ lighTram 10 Plug. Not read.
- **Irizar ie bus, Volvo 7900 Electric (and BZL chassis), VDL Citea, Iveco E-Way, Heuliez GX Elec,
  Ebusco 2.2 and 3.0, BYD European models (for example the 12 m model sold in Europe), Yutong E12,
  Gillig battery-electric, Proterra ZX5**: maker data sheets not searched or not read, within the limit
  of about 15 sources. The Swiss register (V7) shows no BYD bus and two Ebusco buses of 2018.
- **ASTRA type approvals in the new TAS application** (`opendata.astra.admin.ch/ivzod/4000-Typengenehmigungen_TAS/`):
  approvals issued after the TARGA file (V4); may hold the eCitaro with AxTrax 2 LF.
- **Solaris battery supplier**: search snippets name Impact Clean Power Technology (2025) and BMZ
  Poland (2019) as Solaris battery suppliers; not read.
- **MAN Lion's City E with central motor from Scania**: the motor type MG4115-1 (V4) and GE21F1 (V6)
  could lead to a Scania or MAN technical document that names the inverter.
