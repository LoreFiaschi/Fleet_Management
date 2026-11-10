# Simulation tools and data for validating a schedule of the case study

Status: done (10 Nov 2026). Sections 1-6 filled: 10 tool sources (T1-T10, read in full, page limits stated) and 9 data sets (D1-D9, confirmed by download except D3). Next, if wanted: the leads of section 6 (Gallet et al. 2018 first, for the drive-cycle step), the Ist-Daten documentation (blocked here), and the vehicle parameters of `bus_types.md`.

Terms used in this file:

- **GTFS** (General Transit Feed Specification): the common file format of timetables. `shapes.txt` is its optional file of road geometry per trip.
- **OSM** (OpenStreetMap): a map database built by volunteers.
- **BEB** (battery-electric bus); **SOC** (state of charge); **HVAC** (heating, ventilation and air conditioning).
- **Block** (German *Umlauf*): the sequence of trips that one bus serves in one day.
- **Drive cycle**: speed (and grade) over time, second by second.
- **Monte Carlo**: repeating a random simulation many times and counting the outcomes.
- **Soll-Ist** (German): scheduled against actual. **Ist-Daten** (German): actual data.
- Source codes: T1-T10 are tool sources, D1-D9 data sets (section 5).

## 1. Tools

The table says what each tool simulates and what it offers for our question. "Not read" means that no source on the tool was read; the row then gives no claim.

| Tool | Level | E-bus support | GTFS | Gradients | Licence | Uses for e-bus fleets (sources read) |
|---|---|---|---|---|---|---|
| **SUMO** (T1) | microscopic traffic flow and single-vehicle motion, every time step; background traffic from origin-destination data (T10) | battery device: energy content, charging stations on lanes, charge curve, one constant auxiliary power; energy from the Electric Vehicle Model, MMPEVEM or HBEFA 4.2 (PHEMlight needs non-free data); no HVAC model over temperature; no ageing | yes: `gtfs2pt.py`; with `shapes.txt` and OSM lines, routes follow OSM; else fastest path between stops | yes: heights from OSM or a GeoTIFF height map; the electric model uses them | Eclipse Public License 2.0; free | one study read: Transit-Gym (T10), GTFS + OSM -> SUMO -> neural-network energy per trip, Chattanooga (USA). The SUMO energy paper (Kurczveil et al. 2014) was not read |
| **MATSim** with `ev`, `pt2matsim`, eqasim (T2) | agent-based travel demand and the traffic it makes | `ev`: charging behaviour of car drivers (charger choice, queues, tariffs); no bus energy model, no ageing in the README | yes, through `pt2matsim` (GTFS or HAFAS, mapped onto an OSM network) | not mentioned in the sources read | GNU GPL 2 or later; free | none found in the sources read; eqasim lists no Swiss scenario |
| **PTV Visum** (T8) | macroscopic transport planning; line blocking (vehicle scheduling) | e-bus line blocking with linear charging and a range in km; elevation import "might be of use" for energy | not stated in the source read | elevation import (vendor blog) | commercial (PTV Group); price not public in the source | vendor examples only |
| **PTV Vissim**, **Aimsun** | not read | - | - | - | commercial | - |
| **FASTSim** (T3) | single-vehicle powertrain energy over a given drive cycle (backward model) | battery-electric vehicles; "medium- and heavy-duty vehicles"; no bus file seen | no (pages read) | yes: `grade` per time step; also air temperature for thermal models | Apache 2.0; free (Python) | feeds the EV state-of-charge profiles of BLAST-Lite (T4); no bus use in the sources read |
| **Autonomie** (T9) | single-vehicle powertrain simulation (MATLAB) | not stated in the source read | no | not stated | licence agreement (in 2014 through LMS International) | none in the source read |
| **ADVISOR**, **Modelica** bus models | not read | - | - | - | - | - |
| **MATLAB Simulink / Simscape bus model** of Konzept et al. (T7) | single-vehicle longitudinal dynamics plus an electrical battery model | MAN Lion's City 12 E; heat pump over air temperature; constant auxiliaries; SOC; no ageing | no (speed profile input) | yes: grade over distance | MATLAB licence (commercial); model code not published | checked on VBZ telemetry: SOC within 1.45 % on a 44 km round trip |
| **BLAST-Lite** (T4) | battery cell ageing (capacity fade) | semi-empirical fade models of many Li-ion cells (LFP, NMC, NCA, LTO); input SOC and temperature time series | no | no | BSD-style; free (Python, MATLAB) | cars and stationary storage in the README; its coefficients are used in `component_battery.md` (B8) |
| **SimBA** (T5) | e-bus fleet operation: trip energy, charging, SOC per bus | yes: energy per trip from a table (load, incline, mean speed, temperature) that the user must supply; depot and opportunity charging (SpiceEV); battery life only as a fixed number of years in costs | no: CSV schedule of trips with rotations | mean grade per trip from the heights of first and last stop | GNU AGPL 3; free (Python) | RLI planning tool; no published case read |
| **eflips-X** (T6) | e-bus fleet planning: block building, depot assignment, terminus chargers, depot simulation, smart charging | yes: energy lookup tables over speed, gradient, temperature and load; SOC per block; no ageing | not stated in the paper | yes (in the lookup table) | GNU AGPL; free (Python, Rust) | Berlin (BVG), one week, three scenarios (T6) |
| **Our own Python code** | any step of section 2 | as built | GTFS read by our own scripts (`extract_line_features.py` exists) | swisstopo profiles (already used) | ours | - |

**Randomness in the tools read.** In SUMO, the departure times of background trips can be drawn at random (T10, step 6); the example vehicle types of T1 carry an attribute `sigma="0.0"` whose meaning the pages read do not explain. SimBA has only a fixed buffer time at chargers (T5). eflips-X ran without randomness (T6); a study it cites drew delays and consumption from distributions (T6, section 1.4.5). BLAST-Lite can scale the fade rate and simulate a spread of fade (T4). No tool read draws component failures.

**Standard tool or single study.**

- No source read shows a tool that the field treats as the standard for simulating an e-bus fleet with component ageing. The eflips-X authors write that before their work "No unified framework exists that combines all necessary steps for bus electrification" (T6, section 1). Transit-Gym writes that "Very limited literature studied electric bus" at the microscopic level (T10, section II.D).
- Established **methods** (not tools), stated as methods in peer-reviewed sources read: the traction energy "can be calculated using a physics-based model, if the driving profile and vehicle parameters are known", else empirical models are used (T6, section 1.4.1); HVAC dominates the auxiliary energy (same place). For battery ageing, semi-empirical life models are what "the industry mainly uses" (carried from `component_battery.md`, B7).
- Each tool use in the table is **one study** (T6, T7, T10).

**Recommendation.** Write the chain in our own Python code. Take the physics from published practice (T6, T7), the battery fade from BLAST-Lite, and the measured running times and blocks from the open data (D5, D6). Use SUMO only as an optional check of speed profiles under traffic (T10 shows that traffic changes bus energy strongly). Reasons: (1) no tool read ages tires or inverters or draws failures, so own code is needed anyway; (2) our inputs per bus and day are few (trips, km, stops, grade, load, temperature); (3) measured stop times exist for almost all trips of our lines (D6: 3,512 of 3,599 arrivals of line 759 had status `REAL` on 1 Oct 2026; D5 for VBZ until July 2023), so a traffic simulation is not needed to get realistic running times.

## 2. Proposed simulation chain

Goal: take one schedule of the library (which bus serves which line in each period; when each component is repaired or replaced) and simulate what the real buses would experience. The validators of the library (`validate`, the Gamma validators) replay the model's own increment process; this chain instead derives each increment from physics and data, then draws failures.

Example to keep in mind: bus 3 serves line 768 for one week of January, then line 759; its battery pack is partly renewed in week 30.

| Step | What it does | Inputs (section 3) | Tool | Published practice |
|---|---|---|---|---|
| 1. Bus-days | For each period and each bus, pick the line from the schedule; for each day, pick one block of that line (day type: weekday, Saturday, Sunday). | schedule; blocks from Ist-Daten `UMLAUF_ID` (D6) or from our own block estimate (`zurich_depot_and_lines.md`, section 5) | own Python | block building is a step of eflips-X (T6, section 2.2) and SimBA rotations (T5) |
| 2. Segments | Split each trip of the block into stop-to-stop segments with length and grade profile. | ZVV GTFS `stop_times`, `shapes` with `shape_dist_traveled` (D2); heights from the swisstopo profile service (D4) | own Python (extends `extract_line_features.py`) | grade over distance as model input: T7, section 1 |
| 3. Running and dwell times | Draw the time of each segment and each stop from the measured times of the same segment, line, hour and day type. | VBZ Soll-Ist 2015-07/2023, in seconds (D5); Ist-Daten from 08/2023, all operators, VBG included (D6) | own Python | my choice; it replaces a traffic simulation |
| 4. Speed profile, 1 Hz | Build a drive cycle per segment: accelerate, cruise, brake, so that the segment length is covered in the drawn time; optional extra stops at signals. | step 3; acceleration limits (assumption, or bus data in `bus_types.md`); signals from OSM (D9) | own Python; SUMO as an optional check (T1, T10) | my construction; a published method (Gallet et al. 2018) is a lead, not read (section 6) |
| 5. Energy and power, 1 Hz | Longitudinal model: rolling, air, grade and inertia forces; drive and recuperation efficiency; HVAC power from air temperature; constant auxiliaries. Mass = empty mass + passengers × 68 kg. | vehicle data (`bus_types.md`); occupancy per segment (D7, VBZ lines); hourly air temperature (D8) | own Python (equations of T7), or FASTSim with a bus file (T3) | physics-based traction energy: T6, section 1.4.1; T7, Eqs. 1-2; 68 kg per passenger: T6, section 3.3.2 |
| 6. SOC and charging | Integrate battery power over the day; add depot charging at night and opportunity charging at the terminus where it exists (line 759). Flag a block with SOC below a minimum. | step 5; charger power and places (`zurich_depot_and_lines.md`, section 1) | own Python; SimBA or eflips-depot as an alternative (T5, T6) | SOC per block: T5, T6 |
| 7a. Battery damage | Capacity fade from the SOC and temperature series of each day (calendar and cycle ageing, cycle counting inside). | step 6; pack temperature (assumption: kept near 25 °C, `component_battery.md`) | BLAST-Lite (T4), cell model scaled to the pack | NREL chain FASTSim -> SOC profile -> BLAST-Lite for cars (T4) |
| 7b. Tire damage | Tread wear per second from the slip-energy structure (force squared over normal load), summed per day and axle, scaled so that the fleet mean equals the cited wear rate. | step 5 forces; `component_tires.md`, sections 2.3 (F1, F2) | own Python | structure from HDM-4 as cited in `component_tires.md` (S10) |
| 7c. Inverter damage | Phase current -> junction temperature (thermal model) -> rainflow cycle count -> Miner damage with the power-cycling model. | step 5 power; `component_inverter.md`, sections 2.2-2.3 | own Python | mission-profile lifetime estimation, carried from `component_inverter.md` (S3, S5) |
| 8. Scatter and failures | At the start of each run, draw per bus a fade-rate factor (battery), a wear-rate factor (tires) and a strength (inverter, Weibull). A component fails when its damage passes its limit. Apply the planned repairs and replacements with their laws (for example ARD∞); a failure before the planned action is a violation. | coefficients of variation and Weibull shape from the three component files | own Python | spread of fade: BLAST-Lite `degradation_scalar` (T4); the failure logic is my construction |
| 9. Monte Carlo | Repeat steps 3-8 many times (for example 1,000 runs). Draw a weather year from the hourly records since 1980 (D8), loads around the means of D7, running times from D5 or D6. Count, per component and period, how often the limit is passed; compare with the risk level ε of the schedule. | all above | own Python | my construction |

Notes on the chain:

- **Same coefficients, different question.** If steps 7a-7c use the coefficients of the component files, the simulation tests the approximations of the model (one mean increment per line and period, independent increments, the bounds, the linear forms), not the coefficients. Testing the coefficients needs operator data on real buses (section 4).
- **Cost.** Steps 4-7 run at 1 Hz. One bus-day of about 250 km at 20 km/h is about 45,000 time steps; a fleet of 48 buses over one year is about 8 × 10^8 steps per Monte Carlo run. A run per trip pattern can be cached: a segment with the same drawn time, load and temperature gives the same energy. This estimate is mine.

## 3. Data for our scenarios

"Confirmed" = downloaded (whole file or a byte range) and counted, on the date given. Details and numbers are in section 5 (D1-D9).

| Input | Exists? | Source | Licence | Format | Confirmed by download? |
|---|---|---|---|---|---|
| Timetable and stop positions | yes | national GTFS of SKI+, copy on the Mobility Database (D1); ZVV GTFS on data.stadt-zuerich.ch (D2) | D1: terms of opentransportdata.swiss, not read (site blocked); D2: CC0 (licence field) | GTFS zip | yes: D1 by the depot study (9 Nov 2026) and its member list here; D2 by byte ranges, 10 Nov 2026 |
| Road geometry of the lines | yes | ZVV GTFS `shapes.txt`, every trip has a shape (D2); OSM route relations (depot study); cantonal "Linien des öffentlichen Verkehrs" (D3) | D2: CC0; OSM: ODbL 1.0; D3: open use, but vector data behind a sign-in | GTFS shapes; OSM JSON; D3: WMS images only | D2: yes (line 759: 14.07 / 13.56 km; OSM: 14.08 / 13.56 km); OSM: yes (depot study); D3: no (WFS HTTP 401) |
| Elevation | yes | swisstopo profile service; swissALTI3D 0.5 m and 2 m tiles (D4) | licence field "proprietary"; swisstopo terms not read | JSON profile; cloud-optimised GeoTIFF | yes: profile test and one 2 m tile (1 km², 500 × 500 cells), 10 Nov 2026; all lines profiled by the depot study |
| Measured running and dwell times | yes for VBZ until 29 Jul 2023; yes for all operators since then | VBZ Fahrzeiten Soll-Ist (D5); Ist-Daten of opentransportdata.swiss, archive host (D6) | D5: CC0; D6: open use, source must be named (terms "terms_by") | D5: weekly CSV, about 250-274 MB, times in seconds; D6: daily CSV of 405-683 MB in monthly zips (1.4 GB for October 2026) | D5: yes (4 MB sample of one week); D6: yes (the whole day 1 Oct 2026) |
| Vehicle blocks (*Umläufe*) | yes | D5 (`umlauf_von`, and the vehicle number `fahrzeug`, VBZ, until 2023); D6 (`UMLAUF_ID`, all operators); not in either GTFS (`block_id` joins only 2 trips) | as above | CSV | yes: line 759, 8 `UMLAUF_ID` on 1 Oct 2026 (D6) |
| Passenger loads per line and stop | VBZ: yes, yearly means per planned trip and segment; VBG: no stop-level open data found | VBZ Fahrgastzahlen (D7); VBG annual report 2025, p. 8, passengers per line (read by the depot study) | D7: CC0 | D7: CSV or Parquet; 95 MB main file | D7: yes (4 MB sample; 111 lines in `LINIE.csv`, VBG lines absent) |
| Air temperature | yes | MeteoSwiss SwissMetNet, Zürich / Fluntern (SMA) and Kloten (KLO) (D8) | CC BY (licence field) | CSV, 10 min to yearly; hourly back to 1980 | yes: hourly 1 Jan - 9 Nov 2026, 7,512 values per station, no gap |
| Vehicle parameters | in progress | `case_study/bus_types.md` (another agent) | - | - | not checked here |
| Charger power and places | partly | operator pages (`zurich_depot_and_lines.md`, section 1): Hardau 105 points of 100 or 150 kW; line 759: 300 kW pantograph at the airport (2021 plan) and 40 kW in the depot | operator pages | text | carried from the depot study |
| Traffic signals | yes (positions only) | OSM `highway=traffic_signals` (D9) | ODbL 1.0 | Overpass JSON | yes: 104 signals around Oerlikon; count per route failed |
| Congestion | only inside measured running times | D5, D6 | as above | - | as above; no separate congestion data checked |
| Real condition of the components (SOH, tread depth, inverter failures per bus) | no public data found | gaps of `component_battery.md`, `component_tires.md`, `component_inverter.md` | - | - | - |

**VBG lines (Bassersdorf depot), what exists:**

| Input | VBG |
|---|---|
| Timetable and stops | yes: both GTFS feeds (D1, D2; VBG = agency 773 in D1, route prefix 31 in D2) |
| Road geometry | yes: ZVV GTFS shapes for 759, 768, 787, 739, 769, 797 checked (D2) |
| Elevation | yes: depot study profiles; D4 |
| Measured times | **only D6** (Ist-Daten): line 759 on 1 Oct 2026, 3,512 of 3,599 arrivals with status `REAL`. D5 is VBZ only. |
| Blocks | D6 `UMLAUF_ID` (759: 8; 768: 12; 787: 5; 769: 2; 797: 2 on 1 Oct 2026) |
| Passenger loads | no stop-level data found; line totals in the VBG annual report (depot study). The 4 MB sample of D7 and its line table hold no VBG line. |
| Temperature | Kloten station (KLO); lines 759 and 768 end at Zürich Flughafen, in the municipality of Kloten. The distance from the station to each line was not measured. |

## 4. Verdict and gaps

**Verdict: yes, the operation of the buses can be simulated with public data, for the VBZ garages and for the VBG lines.** Timetable, road geometry, elevation, measured running times, blocks and temperature are all open and confirmed by download (section 3). Passenger loads are open for VBZ lines only. What is not public is the bus itself (parameters) and the real condition of its components.

**Gaps and how to estimate them:**

| Gap | Effect on the simulation | How to estimate it |
|---|---|---|
| Vehicle parameters: empty mass, drag area, rolling resistance, motor and inverter efficiency maps, HVAC power over temperature | sets the energy per km (step 5) | maker data sheets in `bus_types.md`; the parameter structure and the checked VBZ energy split of T7 (drivetrain 57 kWh and auxiliaries 12 kWh on 44 km in winter); the energy-per-km fit of `component_battery.md` as a check |
| Speed inside a segment (only stop times are public) | sets the acceleration events (inverter, tires) more than the energy | synthetic drive cycles (step 4); check the energy per km against T7 and the field range 1.0-1.4 kWh/km carried in `component_battery.md`; GPS traces at 1 Hz are not public (T7 used operator telemetry) |
| Passenger loads of VBG lines | mass on 759, 768 and the other VBG lines | scale a VBZ load profile of a similar line (for example a line into Oerlikon) to the VBG line totals of the annual report |
| Spread of the load between single trips (D7 gives means only) | width of the Monte Carlo spread | an assumed distribution around the mean (for example Poisson boardings); no source read gives it |
| Ist-Daten documentation; the meaning of `UMLAUF_ID` | blocks for step 1 | blocked here by the antivirus; the user can open `opentransportdata.swiss` from another network or allow the site. For 759, 787 and 31 the count of `UMLAUF_ID` equals our vehicle estimate; for 768 and 89 it is about twice it |
| Real condition of the components (SOH from the battery management system, tread depths, inverter failures) | the simulation cannot test the degradation laws, only the approximations of the model | operator data from VBZ or VBG; until then, use the component files and their stated ranges |
| Charging plan in the depot (when and at what power each bus charges) | SOC and battery temperature at night | assume charging at the nominal power from arrival until full, or the "balanced" strategy of SimBA (T5) |
| Pack temperature in operation | calendar and cycle fade | assume a thermal management that keeps the pack near 25 °C (`component_battery.md`); add a seasonal case |

**Main risk.** The simulator and the schedule would share the same component coefficients. The simulation then validates the scheduling model's simplifications under realistic operation, not the truth of the degradation laws. This should be said in any result.

## 5. Source entries

Downloads are in the session scratchpad, `downloads\sim\NN\` (log: `downloads\sim\log_sim.tsv`). All accessed 2026-11-10 unless stated.

### T1. SUMO documentation: electric vehicles, emissions, elevation, GTFS import, licence

- **What it is.** The user documentation of Eclipse SUMO (Simulation of Urban MObility), an open-source microscopic traffic simulator mainly developed by the Institute of Transportation Systems of the German Aerospace Center (DLR). Five pages, taken as raw Markdown from the SUMO GitHub repository (branch `main`): `docs/web/docs/Models/Electric.md` (download 01), `Tools/Import/GTFS.md` (02), `Models/Emissions.md` (08), `Networks/Elevation.md` (11), and the repository `README.md` (12).
- **Evidence basis.** Read in full (all five pages).
- **Findings.**
  - Battery device (`Electric.md`, sections "Overview", "Defining Electric Vehicles"): since version 0.24.0, a vehicle device `device.battery` tracks the energy content of the battery; charging stations sit on lanes; the option `--battery-output` writes, per vehicle and time step, energy consumed, energy regenerated, battery content, speed, acceleration and position. Parameters: capacity, maximum charge rate (constant or a curve over state of charge), mass and `loading` (extra mass), front area, air-drag, rolling and radial drag coefficients, rotating mass, `constantPowerIntake` (auxiliary power, default 100 W), propulsion and recuperation efficiencies.
  - The page gives an example vehicle type "ElectricBus" (12 m, mass 10,000 kg) with "the values for a city bus from the original publication" (Kurczveil, López, Schnieder 2014); its battery capacity is 2,000 Wh, so it is a demonstration, not a calibrated bus.
  - Auxiliaries: one constant power (`constantPowerIntake`). No heating or air-conditioning model that depends on air temperature is described.
  - Behaviour: "By default, vehicle behavior will not be affected by battery level" (section "Vehicle behavior"); an external script through TraCI (the Traffic Control Interface) must react to an empty battery.
  - No battery ageing: the page models energy content only; the capacity is a fixed parameter.
  - Calibration note (section "Example Configurations", Kia Soul EV): simulated 6.3-6.7 km/kWh against measured 6.4-6.8 km/kWh "in short and medium range simulations with realistic traffic"; "abstract scenarios without junctions and other cars overestimate the efficiency by a large factor (~ twice as many km/kWh)".
  - Energy models (`Emissions.md`, table "Pollutants / Measurements covered by models"): electricity consumption comes from the Electric Vehicle Model (`emissionClass="Energy"`), from MMPEVEM (an electric-vehicle model of RWTH Aachen), from HBEFA 4.2 (Handbook of Emission Factors for Road Transport), or from PHEMlight / PHEMlight5 "only if the data files for battery powered or hybrid vehicles are available (which are not part of the free data set)".
  - Gradients (`Elevation.md`): elevation enters the network from OpenStreetMap "with some tricks", from shapefiles, or from a grey-scale GeoTIFF height map (`netconvert --heightmap.geotiff`). The Electric Vehicle Model and PHEMlight use elevation data (section "Models that use Elevation Data").
  - GTFS (`GTFS.md`): `gtfs2pt.py` imports a GTFS feed (needs `routes.txt`, `stops.txt`, `stop_times.txt`, `trips.txt`, `calendar.txt`, `calendar_dates.txt`) for one date. Without `shapes.txt` it builds each route as the fastest path between stops; with `shapes.txt` and OSM public-transport lines it uses the OSM routes.
  - Licence (`README.md`, section "License"): Eclipse Public License 2.0.
- **Use for us.** SUMO can turn the timetable and the road network into a second-by-second speed and battery-energy profile of each bus, with gradients from a height model. It does not age the battery, the tires or the inverter.

### T2. MATSim: electric-vehicle contribution, pt2matsim, eqasim, licence

- **What it is.** MATSim (Multi-Agent Transport SIMulation) is an open-source, agent-based transport simulation in Java. Four files from GitHub: the README of the `ev` contribution (`matsim-libs/contribs/ev/README.md`, download 13); the README of `pt2matsim` (06), a converter of timetables to MATSim; the README of `eqasim-java` (07), the scenario framework of the ETH Institute for Transport Planning and Systems (IVT); the licence file `matsim-libs/matsim/LICENSE` (read on GitHub, not saved).
- **Evidence basis.** Read in full (all four).
- **Findings.**
  - `ev` README: two packages, WEVC (within-day electric vehicle charging) and SEVC (strategic electric vehicle charging). Both model when and where *agents* (simulated persons) charge their cars: charger choice, queues, tariffs, a "charging score" that penalises zero state of charge. The README says nothing about buses, about a drive-energy model, about gradients or about battery ageing.
  - `pt2matsim` README: converts GTFS or HAFAS (the Swiss timetable format) into a MATSim transit schedule and maps each route onto a network built from OSM (`Osm2MultimodalNetwork`, `PublicTransitMapper`). Footnote 1: the Swiss timetable is on opentransportdata.swiss, and "only HAFAS currently supports chained trips (Durchbindungen)" (*Durchbindung*: a vehicle that runs on as another trip).
  - `eqasim-java` README: "ready-to-run" MATSim scenarios; the pipelines listed are Île-de-France and Paris, California and Sao Paulo. No Swiss or Zurich pipeline is listed. Main reference: Hörl and Balac 2021, *Procedia Computer Science* 184, 712-719 (not read). The repository `eqasim-org/switzerland` did not answer (HTTP 404 for its README on three branches, 2026-11-10).
  - Licence: GNU General Public License, version 2 or later.
- **Use for us.** MATSim simulates travel demand and the traffic that results; its electric part targets private-car charging. For a fixed bus schedule with no demand question, it adds little that SUMO or our own code does not give. pt2matsim is a useful GTFS-to-road mapper in Java.

### T3. FASTSim documentation (NLR, formerly NREL)

- **What it is.** FASTSim (Future Automotive Systems Technology Simulator), a vehicle powertrain simulator of the National Laboratory of the Rockies (NLR; the pages use this name for the former NREL, National Renewable Energy Laboratory). Read: the README of branch `fastsim-3` (download 14), the documentation pages "FASTSim Documentation" (15), "User Guide" (16) and "Drive Cycles" (17); the licence file of branch `fastsim-3`.
- **Evidence basis.** Read in full (the four pages and the licence header). The pages "Vehicle Models", "Running Simulations" and "Background" were not read.
- **Findings.**
  - It models conventional, hybrid, plug-in hybrid, battery-electric and fuel-cell vehicles; it "has been used to simulate ... a wide variety of medium- and heavy-duty vehicles" (README). It is a "backward-looking model" (Drive Cycles page): the speed trace is given, the power demand is computed.
  - A drive cycle holds `time_seconds`, `speed_meters_per_second`, `grade` and `temp_amb_air_kelvin` (the last "only affects thermal vehicle models"); custom cycles load from CSV, JSON, YAML and other formats (Drive Cycles page).
  - Python package (`pip install fastsim`, Python 3.10-3.15); Apache License 2.0.
  - The pages read do not mention buses, GTFS or battery ageing.
- **Use for us.** FASTSim fits the step "speed profile with grade -> battery energy and power per second" if a bus vehicle file exists or is built. Its own pages read here give no bus model; the vehicle database was not opened.

### T4. BLAST-Lite README (NLR, formerly NREL)

- **What it is.** The README of BLAST-Lite (Battery Lifetime Analysis and Simulation Toolsuite), a Python and MATLAB library of semi-empirical battery ageing models (download 03; licence file read on GitHub).
- **Evidence basis.** Read in full.
- **Findings.**
  - Models of capacity fade for many commercial lithium-ion cells (LFP, NMC, NCA, LTO), as functions of temperature, state of charge, depth of discharge and charge or discharge rate; input is a state-of-charge time series (sections 1, "Simulation details").
  - EV use: the state-of-charge profiles come from FASTSim (section "Electric vehicles").
  - `degradation_scalar` scales the fade rate; `simulate_battery_life_distribution` simulates a spread of fade (section "Simulation details").
  - Caveats (section "Caveats"): models are fitted on 1-3 years of accelerated tests and "it is good to treat all models as qualitative"; there is "not enough information to estimate cell-to-cell variability"; a pack lasts 20-30 % less than a cell (rule of thumb).
  - Licence: BSD-style, "Copyright (c) 2023, Alliance for Energy Innovation, LLC".
  - The README does not mention buses.
- **Use for us.** BLAST-Lite fits the step "state-of-charge profile per day -> capacity fade" of the battery. `component_battery.md` (entry B8) already took the LFP 250 Ah and NMC coefficients from it.

### T5. SimBA documentation (Reiner Lemoine Institut, Berlin)

- **What it is.** SimBA (Simulation toolbox for Bus Applications), a Python tool of the Reiner Lemoine Institut (RLI), a non-profit research institute in Berlin. Read: GitHub README (download 05), documentation pages "Getting Started" (19), "Features of SimBA" (20) and "Simulation Parameters" (21) on rli-simba.readthedocs.io (revision b4fed685); licence file of branch `master` on GitHub (header only).
- **Evidence basis.** Read in full (the four pages). The page "Modes" was not read.
- **Findings.**
  - Purpose ("Getting Started"): "analyze and optimize electrified bus fleets": locate and size chargers, size buses, check whether trips and rotations (*Umläufe*) can be electrified, choose charging strategies, compute costs.
  - Input ("Simulation Parameters", table 3): a CSV schedule with one row per trip: `rotation_id`, departure and arrival stop and time, distance in m, vehicle type; optional line, charging type, temperature and `level_of_loading` (0 empty, 1 full). The pages read describe no GTFS import.
  - Energy ("Features", "Consumption analysis"): either a constant kWh/km, or a table interpolated in level of loading, incline, mean speed and ambient temperature. "Its creation is at this point of development not part of SimBA": the user must supply the table. The incline is the mean grade between the heights of the first and the last stop of a trip, from a file `all_stations.csv`.
  - Charging: done by the open-source tool SpiceEV, built in; depot buses charge with the strategy "balanced", opportunity chargers with "greedy".
  - Outputs: state of charge per vehicle and per rotation, power per charging station, costs.
  - Battery: a fixed `lifetime_battery` in years (example value 7) enters the cost calculation only; there is no ageing model. No randomness is described, except a fixed "buffer time" for delays at opportunity chargers.
  - Licence: GNU Affero General Public License, version 3 (licence file, branch `master`). A search-engine summary said "MIT"; the file says otherwise.
- **Use for us.** SimBA checks whether each bus block of our schedule stays above a minimum state of charge with the planned chargers. It needs an energy table per bus type, which we would build ourselves (with FASTSim or SUMO, or from field data). It does not age components.

### T6. Heide, Guo, Göhlich 2025: the eflips-X framework (TU Berlin)

- **Reference.** Heide, L., Guo, S., Göhlich, D. (2025). From simulation to implementation: a systems model for electric bus fleet deployment in metropolitan areas. *World Electric Vehicle Journal* 16(7), 378. DOI 10.3390/wevj16070378. Read as HTML in the browser pane (www.mdpi.com/2032-6653/16/7/378); the PDF download was refused (HTTP 403 to curl and to PowerShell).
- **Evidence basis.** Read in full: text of sections 1-5, the data statement and the 16 tables. Figures were not inspected; the reference list was not read.
- **Findings.**
  - What it is: "a modular, open-source simulation framework" (abstract) with steps energy consumption, battery-aware block building, depot-block assignment, terminus charger placement, depot simulation and smart charging (section 2, Figures 1-2). Code under the GNU Affero General Public License (section 2); packages `eflips-model`, `eflips-depot`, `eflips-opt`, `eflips-schedule-rust`.
  - Energy (section 2.1): `eflips-model` uses lookup tables "parameterized by speed, gradient, ambient temperature, and vehicle loading", with a generator after Ji et al. (reference 20, not read). In the Berlin case (section 3.3), the articulated-bus table comes from 798,502 km of measured driving, binned by mean speed and outside temperature, median per bin; payload 1.2 t (17.6 passengers × 68 kg). The coldest day (-12 °C) gives about 50 % more consumption than the warmest day (section 4.1.1).
  - Input: the case study used complete schedules and blocks given by the operator BVG (Berliner Verkehrsbetriebe) for one week (section 3); the paper does not mention GTFS.
  - Randomness: none in the case study; a cited depot study (Giuntoli et al., reference 42) drew delays and consumption from distributions (section 1.4.5).
  - Battery ageing: not modelled; "battery degradation" appears only as a motive for smart charging (section 2.6).
  - The literature review (sections 1.4.1, 1.4.7, Tables 1-2) lists integrated works by Rogge et al. 2018, Berthold 2019, Topić et al. 2020, He et al. 2022, Foda et al. 2023, Zeng et al. 2023, Guanetti et al. 2023 and Jahic 2024; it names no common simulation tool among them.
  - Claim of the authors (section 1.2): "the first fully integrated, open-source simulation framework for battery-electric bus fleets"; "fine-grained, integrated solutions exist in the commercial domain [9]" (reference 9 not read).
- **Use for us.** eflips-X is the closest open-source tool to a whole-fleet e-bus simulation: blocks, depot, charging, state of charge per bus. Its central question (fleet size, chargers, grid power) is a planning question; it does not age components or draw failures. Its energy lookup-table idea (speed, gradient, temperature, load) is the published practice that our energy step can copy.

### T7. Konzept, Hitz, Reick 2025: energy model of a battery-electric bus, checked on VBZ data

- **Reference.** Konzept, A., Hitz, A., Reick, B. (2025). Development of a simulation model for predicting energy consumption of battery-electric buses. *SNE Simulation Notes Europe* 35(1), 49-53. DOI 10.11128/sne.35.sn.10728. Open-access PDF (download 25).
- **Evidence basis.** Read in full (5 pages).
- **What it is.** A short peer-reviewed note (post-conference paper of ASIM SST 2024) from the Institute for e-mobility of the Ravensburg-Weingarten University of Applied Sciences, Germany.
- **Findings.**
  - Model (section 1, Figure 1, Eqs. 1-4): MATLAB Simulink, longitudinal dynamics only. Inputs: a speed profile over time, the elevation profile over distance, the mean outside temperature. Resistances: rolling, air, gradient, inertia; a motor efficiency map; constant efficiency for gearbox and power electronics; heat-pump power from a lookup table over outside temperature; other auxiliaries as a constant power. Battery: the table-based cell model of Simscape (NMC cell, 4.15 V, 14.44 Ah), with open-circuit voltage and internal resistance over state of charge and temperature; "Self-discharge and aging are initially neglected" (section 1.1).
  - Bus: MAN Lion's City 12 E; "800 V system with 65 % of the 480 kWh battery capacity usable" (section 1.1).
  - Data and check (section 1.2): data from "Verkehrsbetriebe Zürich" (VBZ): GPS position, speed, elevation, outside temperature, heat-pump power, drivetrain power, auxiliary energy, state of charge. Winter data; the diesel heater ran below 7 °C. On a 44 km round trip, maximum state-of-charge deviation 1.4469 %, mean squared error 0.31737 (Figure 2). Table 1: drivetrain 57.07 kWh measured against 57.56 kWh simulated; auxiliaries 11.98 against 11.34 kWh; HVAC 0.84 against 0 kWh.
  - The VBZ GPS data have 5.3 samples per minute and no altitude; the authors rebuild 1 Hz speed and slope with a route-matching method (reference 6, not read).
  - Outlook (section 2): battery ageing is to be added later.
- **Use for us.** This is a published case of our energy step for a VBZ battery bus: speed profile plus elevation plus temperature gives energy and state of charge within 1.5 % of state of charge. It used operator telemetry that is not public. The model code is not published in the paper.

### T8. PTV blog: "Planning electric bus operations with PTV Visum", 6 May 2024

- **What it is.** A vendor blog post of PTV Group (Karlsruhe), maker of the commercial programs PTV Visum (macroscopic transport planning) and PTV Vissim (microscopic traffic simulation). Download 24 (HTML).
- **Evidence basis.** Read in full (text; the videos were not watched).
- **Findings.**
  - Visum does "line blocking" (vehicle scheduling) for e-buses: charging is a repeated activity; "The functions are linear": a full charge in 2 h gives half a charge in 1 h; consumption is a range in km (example: 150 km).
  - "Users of PTV Visum can easily import elevation data, which might be of use to calculate the energy consumption"; one example uses "an advanced consumption model" (not described).
  - The post names battery fade, depth of discharge and temperature as concerns, but describes no ageing model in Visum.
  - No price or licence terms are given.
- **Use for us.** Visum can check the range feasibility of blocks, as our own code can. Nothing in the post shows a second-by-second energy model or component ageing.

### T9. Argonne National Laboratory: Autonomie fact sheet (2014)

- **What it is.** A one-page fact sheet of the U.S. Department of Energy on Autonomie, a vehicle simulation environment of Argonne National Laboratory (download 26, energy.gov, dated 2014 by its path).
- **Evidence basis.** Read in full (1 page).
- **Findings.** MATLAB-based environment for powertrain simulation, "licensed to more than 150 organizations"; "Available through licensing agreements"; distributed by LMS International (in 2014). The page does not mention buses, GTFS, gradients or battery ageing.
- **Use for us.** A licensed, MATLAB-based alternative to FASTSim for the energy step; no advantage for our purpose is shown in the source read.

### T10. Sun et al. 2021: Transit-Gym, a SUMO-based bus transit simulator

- **Reference.** Sun, R., Gui, R., Neema, H., Chen, Y., Ugirumurera, J., Severino, J., Pugliese, P., Laszka, A., Dubey, A. (2021). Transit-Gym: a simulation and evaluation engine for analysis of bus transit systems. arXiv:2107.00105v1 (download 58). Not peer-reviewed in the version read.
- **Evidence basis.** Read in full (8 pages, text and Table I; figures not inspected).
- **Findings.**
  - SUMO as the traffic and transit simulator; inputs: OSM map (converted with `netconvert`, then corrected by hand in `netedit`), static GTFS for stops and trip schedules, vehicle types, an origin-destination matrix for background traffic (section III.A, steps 1-8).
  - Energy: an artificial neural network trained on the buses of the Chattanooga (Tennessee) transit authority takes the 1 Hz speed and acceleration of each simulated bus and returns 1 Hz energy (section III.B, "Energy Estimation"; the model is in their reference 29, not read).
  - Randomness: departure times of background trips drawn uniformly or at random within a period (step 6); a "stochastic check" of the OSM network against traffic counts (section I).
  - Result (Table I): energy rates per trip for diesel, hybrid and electric buses under the original and a 20 % lower background demand; for electric buses the rate changes up to a factor of about 2.3 between the two demands on one trip (trip 1: 11.6 against 5.0 miles per diesel-equivalent gallon).
  - No battery state of charge over a day, no charging, no ageing.
  - Code: "freely available on GitHub" (section V); licence not stated in the paper.
- **Use for us.** One published case of the chain GTFS + OSM -> SUMO -> 1 Hz bus trajectory -> energy per trip. It shows that background traffic changes bus energy strongly, so a speed profile without traffic (or without measured running times) is a weak input.

### Data sets checked by download

"Counted" means: downloaded (whole or a byte range) and counted with my scripts in the scratchpad folder `scripts_sim\`. A catalogue record (CKAN or STAC) is metadata; it is cited only for the licence field, the file list and the file sizes.

### D1. National GTFS feed (SKI+), copy on the Mobility Database

- **Data.** The national timetable 2026 in GTFS, feed version 20260527, taken by the depot study on 9 Nov 2026 (`zurich_depot_and_lines.md`, section 4; file `downloads\lines\34\gtfs_ch_2026_mdb-2898-202605290027.zip`, 162,613,673 B).
- **Counted here (2026-11-10).** The zip has 10 members: `agency`, `routes`, `trips`, `stops`, `calendar`, `calendar_dates`, `transfers`, `feed_info`, `frequencies`, `stop_times`. It has **no `shapes.txt`** (no road geometry). Trip and vehicle counts for VBG (agency 773) and VBZ (agency 849) are in `zurich_depot_and_lines.md`, section 4.
- **Licence.** Terms of opentransportdata.swiss, not read (site blocked, see D6).

### D2. ZVV timetable in GTFS on the open-data portal of the city of Zurich

- **Catalogue record** (download 29, CKAN `vbz_fahrplandaten_gtfs`): "ZVV Fahrplan Tram und Bus (Static GTFS)", author Zürcher Verkehrsverbund (ZVV), licence field "Creative Commons CCZero" (CC0, public domain), updated weekly, files per timetable year; `2026_google_transit.zip` 90,321,184 B.
- **Counted.** End of the zip (download 30, last 400,000 B) for the member list; first 19,496,090 B (download 31) for the members `agency`, `routes`, `trips`, `calendar`, `shapes`, extracted with `zip_partial_extract.py` and counted with `zvv_gtfs_check.py` and `block_check.py`.
  - Members: `agency`, `routes`, `trips`, `stops`, `calendar`, `calendar_dates`, **`shapes.txt`** (65,277,336 B unzipped), `transfers`, `stop_times` (599,559,330 B unzipped).
  - 397 routes (369 bus, 22 tram), 353,681 trips, **every trip has a `shape_id`**; 3,317 shapes with `shape_dist_traveled`.
  - VBG lines are included (route prefix 31). Line 759: 978 trips, main shapes 14.07 km and 13.56 km (572 points each); line 768: 1,080 trips, 5.61 km and 5.70 km. The OSM lengths of the depot study were 14.08 / 13.56 km and 5.66 / 5.62 km: the two sources agree within 1.6 %.
  - VBZ lines 31, 33, 39, 70, 89 and 165 are included with shapes (for example line 31: 12.92 km and 13.09 km).
  - `block_id` is filled for few trips and always joins exactly 2 trips (prefix 1: 11,814 of 58,093 trips; VBG: 110 of 37,230; VBZ prefixes 2, 3, 4: none). It does not give vehicle blocks.
- **Use.** Official road geometry per trip for VBZ and VBG lines, CC0; it replaces the OSM geometry of the depot study.

### D3. Cantonal data set "Linien des öffentlichen Verkehrs" (canton of Zurich)

- **Catalogue record** (download 43, opendata.swiss `linien-des-offentlichen-verkehrs`): lines updated "zusammen mit ZVV und VBZ" (with ZVV and VBZ), based on the line geometry in DIVA (the timetable system); one resource only, a map preview (`geo.zh.ch/maps?initialMapIds=ZvvLinienZH`), terms "open use".
- **Tried.** WMS (Web Map Service, images) GetCapabilities answered (HTTP 200); the WFS (Web Feature Service, vector data) at `maps.zh.ch/wfs/ZvvLinienZHWFS` answered HTTP 401 (sign-in needed). Not downloaded; not counted.
- **Use.** Not needed: D2 gives the same kind of geometry as open vector data.

### D4. swisstopo elevation: profile service and swissALTI3D

- **Profile service** (`api3.geo.admin.ch/rest/services/profile.json`): used by the depot study for all lines (downloads 38-136 of `downloads\lines`). Test today (download 55): a 1.1 km line in Oerlikon gave 50 points with heights COMB, DTM2 and DTM25 (455.9 m to 431.1 m).
- **swissALTI3D** (download 56, STAC search; download 57, one tile): tile `swissalti3d_2020_2683-1251`, 2 m grid, cloud-optimised GeoTIFF, 1,050,562 B, 500 × 500 cells of 32-bit float (1 km²). Also offered at 0.5 m. The STAC licence field says "proprietary"; the swisstopo terms page was not read.
- **Use.** Heights along the GTFS shapes (profile service), or a height map for SUMO (`netconvert --heightmap.geotiff`, T1).

### D5. VBZ "Fahrzeiten Soll-Ist" (scheduled and actual times), data.stadt-zuerich.ch

- **Catalogue record** (download 27, `vbz_fahrzeiten_ogd_2023`): title ends "(Nachführung eingestellt)" (updating stopped); time range 1 Jan 2023 - 29 Jul 2023; one CSV per week, about 250-274 MB each; licence field CC0. The record says that data after 29 Jul 2023 are in the "Ist-Daten" of opentransportdata.swiss (D6). Years 2015-2023 exist as separate data sets (search result: 9 yearly records, 2015-2023).
- **Counted.** `Haltestelle.csv` (download 32, 35,887 B; stops with `halt_diva`, name), `Haltepunkt.csv` (download 34, 1,007,780 B; platform positions with GPS latitude, longitude, bearing), first 4,000,000 B of `Fahrzeiten_SOLL_IST_20230723_20230729.csv` (download 33), counted with `soll_ist_sample.py`:
  - 34 columns, one row per stop-to-stop run: line, direction, operating date, **`fahrzeug` (vehicle number)**, `kurs`, **`umlauf_von` (vehicle block)**, planned and actual arrival and departure at both stops (seconds after midnight), route variant.
  - Sample: 21,806 rows, 34 lines (trams, buses, regional lines 7xx run by VBZ), 355 vehicles, 500 block numbers. Delay at the next stop: median 7 s, 5 % quantile -52 s, 95 % quantile 126 s. Running time between two stops, actual minus planned: median 0 s, 5 % -15 s, 95 % 26 s.
- **VBG.** VBG lines 759 and 768 are not VBZ lines; this data set is for VBZ vehicles (record text). Not checked in the full weekly file.
- **Use.** Measured stop-to-stop running times and dwell times per vehicle until July 2023: a basis for random travel times. Also the real assignment of vehicles to trips, which shows how VBZ rotates buses between lines.

### D6. Ist-Daten (actual data of all Swiss public transport), opentransportdata.swiss

- **Catalogue record** (download 44, opendata.swiss `ist-daten-v2`): actual times of every trip of the previous day, "entweder effektive Ist-Daten oder die letzte Prognose" (either measured times or the last forecast); one CSV per day, 418-674 MB; terms "terms_by" (open use, must name the source). Older months: `archive.opentransportdata.swiss`.
- **Blocked / reachable.** `data.opentransportdata.swiss`: HTTP 403 from the local antivirus ("Avast WebShield [malicious URL blocked]", 2026-11-10, download 45 empty). `archive.opentransportdata.swiss` is **not** blocked: the listing (downloads 46-48) gives monthly zips `ist-daten-v2-2026-MM.zip`.
- **Counted.** October 2026 zip: 1,396,919,794 B; 30 daily files (2026-10-19 is missing) of 405-683 MB each (download 49, member list). First daily file, 2026-10-01 (download 50, the first 49,949,385 B), counted with `istdaten_scan.py` and `istdaten_umlauf.py`:
  - 22 columns, among them `BETREIBER_ABK` (operator), `LINIEN_TEXT` (line), **`UMLAUF_ID`**, `HALTESTELLEN_NAME`, `ANKUNFTSZEIT` / `AN_PROGNOSE` (planned and actual arrival), `AN_PROGNOSE_STATUS` (`REAL`, `PROGNOSE`, `GESCHAETZT`), `FAELLT_AUS_TF` (trip cancelled), `ZUSATZFAHRT_TF` (extra trip).
  - 2,580,472 rows, 145 operators; VBG 49,832 rows; VBZ 142,504 rows (plus 105,235 rows under the code "VBZ    Fpl").
  - VBG line 759: 3,744 stop events, 145 trips, **8 distinct `UMLAUF_ID`**, 3,512 of 3,599 arrivals with status `REAL`. VBG 768: 244 trips, 12 `UMLAUF_ID`; VBG 787: 133 trips, 5; VBG 769: 18 trips, 2; VBG 797: 64 trips, 2. VBZ 31: 320 trips, 15 `UMLAUF_ID`; VBZ 70: 236 trips, 8; VBZ 89: 282 trips, 25.
  - The meaning of `UMLAUF_ID` is taken from its name (*Umlauf*, vehicle block); the documentation on opentransportdata.swiss was blocked and not read. For lines 759, 787 and 31 the count equals the vehicle estimate n_j of the depot study (8, 5, 15); for 768 and 89 it is about twice n_j (12 against 6; 25 against 13), so an `UMLAUF_ID` may cover part of a day.
- **Use.** The only open source found with actual running times **for VBG lines** and with a block identifier per trip for VBG and VBZ, every day up to the previous day.

### D7. VBZ "Fahrgastzahlen" (passenger counts), data.stadt-zuerich.ch

- **Catalogue record** (download 28, `vbz_fahrgastzahlen_ogd`): computed yearly values of boardings, alightings and occupancy per planned trip and stop, years 2014-2025, CC0. Text: "berechnete Mittelwerte einzelner Fahrplanfahrten, nicht permanent gemessene Werte" (computed means of single timetable trips, not permanently measured values); counting devices in "etwa 20% der Fahrzeuge" (about 20 % of the vehicles); values stratified by day type and extrapolated; unmeasured trips interpolated.
- **Counted.** `LINIE.csv` (download 35: 111 lines, systems tram, trolleybus, bus, night bus and others), `TAGTYP.csv` (download 36: day types of 2025), first 4,000,000 B of `REISENDE.csv` (download 37; full file 95,250,100 B), counted with `reisende_sample.py`:
  - 22 columns: day type, line, planned trip, direction, stop sequence, stop and next stop, departure time, `Anzahl_Messungen` (number of measurements), `Einsteiger`, `Aussteiger` (boardings, alightings), **`Besetzung` (occupancy, persons on board)**, `Distanz` (m to the next stop), days per day type.
  - Sample: 44,105 rows, 60 lines; occupancy per segment median 17.8, maximum 153.9 persons; measurements per trip and segment median 11 (minimum 0).
  - Lines 31, 33, 39, 70, 89, 165 and 743 are in `LINIE.csv`; **759, 768 and 787 (VBG) are not**.
- **Use.** Mean load per trip and segment for VBZ lines: the passenger mass in the energy, tire and inverter steps. No spread per single trip (means only).

### D8. MeteoSwiss automatic stations (SwissMetNet), open data

- **Catalogue record** (download 38, STAC collection `ch.meteoschweiz.ogd-smn`): 158 stations; resolutions 10 min, hour, day, month, year; files "now", "recent" (current year to yesterday) and "historical" (by decade); licence field "CC-BY". Parameter list (download 41): `tre200h0` = "Air temperature 2 m above ground; hourly mean", °C.
- **Counted** (`meteo_count.py`): Zürich / Fluntern (SMA, download 40, 1,017,075 B) and Kloten (KLO, download 42, 1,103,509 B), hourly, 1 Jan 2026 00:00 - 9 Nov 2026 23:00: 7,512 rows each, no gap in `tre200h0`. SMA: minimum -9.6 °C, mean 13.2 °C, maximum 36.2 °C. KLO: -13.1, 13.1, 37.3 °C. Hourly historical files exist from 1980 (SMA item, download 39).
- **Use.** Air temperature per hour for the heating and cooling power and the battery temperature; Kloten lies in the area of the VBG lines.

### D9. Traffic signals and congestion

- **Traffic signals.** OpenStreetMap tags signals as `highway=traffic_signals` (ODbL licence). Counted (download 59, Overpass, database state 2026-11-10T14:13Z): **104** signal nodes in a box of about 1.4 km × 1.9 km around Zurich-Oerlikon (47.405-47.418 N, 8.535-8.560 E). A count per bus route failed: the query of download 52 returned 0 for all eight route relations of lines 31, 89, 759 and 768, which I take as a fault of my query, not as a fact; two retries failed (download 53: HTTP 502; download 54: time-out). OSM does not give signal timings.
- **Congestion.** No open data set of road congestion was checked. D5 and D6 hold actual running times, which already include the effect of traffic and signals on the buses.

## 6. Could not read, and leads not followed

**Could not read or download:**

| Item | Reason (2026-11-10) | Why it matters |
|---|---|---|
| Ist-Daten documentation and current daily files on `data.opentransportdata.swiss`; its terms of use | HTTP 403 from the local antivirus ("Avast WebShield [malicious URL blocked]"); the archive host `archive.opentransportdata.swiss` works | field definitions (`UMLAUF_ID`, status values) |
| PDF of Heide et al. 2025 (T6) on mdpi.com | HTTP 403 to curl and PowerShell; the HTML version was read in the browser pane | none (read as HTML) |
| WFS of the cantonal data set "Linien des öffentlichen Verkehrs" | HTTP 401 (sign-in needed) | none: the ZVV GTFS has shapes |
| Count of traffic signals per bus route (Overpass) | query returned 0 for all routes (fault of my query); retries: HTTP 502 and time-out | signals per km for step 4; a box count worked (104 near Oerlikon) |
| `eqasim-org/switzerland` on GitHub | HTTP 404 for the README on three branches | whether a Zurich MATSim scenario is public |
| FASTSim pages "Vehicle Models", "Running Simulations", "Background" | not fetched (page budget) | whether a bus vehicle file exists |

**Leads not followed (not read; no claim made about their content):**

- **Kurczveil, López, Schnieder 2014**, the SUMO energy model (link in T1): the physics behind SUMO's electric model.
- **Gallet, Massier, Hamacher 2018**, *Applied Energy* 230, 344-356 (cited by T10): energy demand of electric buses from real-world data for large networks; probably the closest published method for step 4.
- **Fiori et al. 2021**, *Transportation Research Part D*, microscopic e-bus energy model calibrated on 435 trips (search result only).
- **Hitz, Konzept, Reick, Rheinberger 2024**, SAE paper on GPS route matching for e-bus fleets (reference 6 of T7).
- **Ji et al.**, the energy regression used by eflips-X (reference 20 of T6); **Jahic 2024** dissertation on depot simulation (reference 45 of T6).
- **GTFS4EV** (GitHub; GTFS-based e-bus fleet and charging simulation), **WeBus** (free tool announced in 2026), **EBSM** (Interreg Central Europe e-bus simulation model): search results only.
- **Gao et al.** (TU Delft; SUMO coupled with a grid model, Rotterdam line 36), **Palazzotto 2023** and **Liu 2024** (Politecnico di Torino; SUMO with Simulink or a backward energy model): search results only.
- A 2026 Politecnico di Milano thesis on a simulation framework for battery ageing of e-buses (search result only).
- **PTV Vissim** and **Aimsun** documentation; PTV blog "E-bus planning: Using simulation to avoid costly surprises" (4 Mar 2026).
- **ADVISOR**, **Modelica** bus libraries, **VECTO** (the European Commission's energy tool for heavy-duty vehicles).
- **Hörl and Balac 2021**, the eqasim pipeline paper (*Procedia Computer Science* 184).
- Traffic counts of the city of Zurich (motor traffic), for congestion apart from bus running times.
