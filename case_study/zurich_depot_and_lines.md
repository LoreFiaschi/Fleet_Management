# A bus depot in the canton of Zurich with battery-electric buses on city and countryside lines

Status: done (9 Nov 2026). Tasks 1-5 complete: operators checked (section 1), three candidate depots compared and depot A, EUROBUS Bassersdorf (VBG), recommended (section 2), its 15 lines measured (section 3, `zurich_line_features.csv`), the script `extract_line_features.py` kept. Open points: section 6 (empty masses, depot fleet, official road geometry; the antivirus block of opentransportdata.swiss).

Terms used in this file:

- **ZVV** (Zürcher Verkehrsverbund): the Zurich transport network, the public-transport authority of the canton of Zurich.
- **BEB** (battery-electric bus): a bus that runs on its traction battery only, with no overhead wire.
- **Depot** (German *Depot*, *Garage* or *Busgarage*): the site where the buses stay overnight and are maintained.
- **Umlauf** (German): a vehicle block, the sequence of trips that one vehicle serves in one day.
- **n_j**: the number of vehicles that line j needs in service at the same time.
- **GTFS** (General Transit Feed Specification): the common file format of public-transport timetables (stops, trips, stop times, service days).
- **OSM** (OpenStreetMap): a map database built by volunteers; its bus route relations give the road that a line follows.
- **Opportunity charging** (German *Streckenladung*): charging at a stop during the day. **Depot charging**: charging in the depot, mostly at night.
- **LV95**: the Swiss map coordinates (EPSG:2056) that the swisstopo services use.

## 1. Operators and depots with battery-electric buses (task 1)

Findings per operator. Each row gives its source (see section 4 for the full reference).

| Operator | Depot(s) | BEB in service | Charging | Source |
|---|---|---|---|---|
| **VBG** (Verkehrsbetriebe Glattal AG). VBG is a management company: bus contractors (*Transportbeauftragte*) run the buses from their own depots. The buses belong to the contractors. | Contractors and the bus lines each one runs: EUROBUS welti-furrer AG, **Bassersdorf**: 739, 742, 749, **759**, 766, **768**, 769, 771, 772, 773, 774, 781, 787, 795, 797. EUROBUS welti-furrer AG, Regensdorf: 449-456, 485, 491, 492, 593, N45. Maag Busbetrieb AG, Kloten: 731-736, 748, 752, 754, 756, 760-762, 796, N75-N78. ATE Bus AG, Effretikon: 640, 650, 652, 655-659, 662, 765, N62, N79. Ryffel AG, Volketswil: 720, 721, 725-727, N72. | 14 electric buses of 121 buses on 31 Dec 2025 (2024: 8 of 120). Both e-bus lines run from the Bassersdorf contractor. **Line 759** (Wangen, Dorfplatz - Dübendorf - Zürich Flughafen, Bahnhof; "over 27 km"): one Mercedes eCitaro standard bus since Sep 2021, all-electric since end of Feb 2023 with 7 more e-buses; VBG calls it the longest battery-bus line in Switzerland. **Line 768** (Zürich Flughafen, Bahnhof - Zürich Oerlikon, Bahnhof): Mercedes eCitaro G articulated buses since Apr 2025 (annual report: since autumn 2025). Plans: about 40 more e-buses by end of 2027 (story of 5 Sep 2025); about 20 more e-buses (annual report 2025, p. 12, outlook); whole VBG bus fleet electric by 2040 at the latest. | Line 759, eCitaro standard bus: opportunity charging (*Streckenladung*) at the terminus Flughafen, Bahnhof by a pantograph that comes down from a mast; battery 264 kWh, range about 100-150 km (story 2023). The fact sheet of 12 Jul 2021 gives 200 kWh, a 300 kW mast charger and a 40 kW depot charger. Line 768, eCitaro G (over 18 m): depot charging with a CCS Combo-2 plug at night or off-peak, full charge about 4 h; usable battery 550 kWh, range about 350 km. | VBG annual report 2025, p. 4 (fleet table), p. 7 (line 768), p. 12 (outlook), p. 25 (contractors); VBG fact sheet line 759, pp. 1-2; VBG web pages *Partner*, *Glattalbahn und Fahrzeugflotte*, *E-Mobilität*; VBG media release 8 Mar 2023; VBG stories of 14 Mar 2023 (line 759) and 5 Sep 2025 (line 768) |
| **PostAuto** (Swiss post bus company), contractor Rolf Stutz AG | New garage in Affoltern am Albis (Knonauer Amt), opened end of March 2025: room for 26 articulated buses, 12 charging points at first, 26 at full build-out. | 2 BEB from mid-April 2025, 13 by end of 2025: all 18 m articulated buses, range at least 250 km. In July 2026 PostAuto names Affoltern am Albis (13 vehicles) as its third-largest e-bus site; it also names Winkel (ZH) as an e-bus site. | Depot charging, mostly at night, also during long breaks in the day. | PostAuto media release 28 Mar 2025; PostAuto news 7 Jul 2026 |
| **VBZ** (Verkehrsbetriebe Zürich), city operator; also the lead operator (*marktverantwortliches Unternehmen*) for four regions around the city | Bus garage **Hardau** (Zurich): about 180 vehicles; 105 charging points of 100 or 150 kW; lines 31, 32, 33, 46, 72, 83 (trolleybus) and 70, 89, 161, 165, 184, 185, 304, 66, 78, 67, 76, 77, 99, 35, 38, 39, 40, 42, 64, 73, plus night buses. Bus garage **Hagenholz** (Zurich): lines 61, 62, 69, 75, 80, 743, 745, 37, 744; being electrified in two stages. Regional garages of contractors: Dübendorf, Effretikon, Dietikon/Bergdietikon, Zollikon, Wädenswil. | 59 battery buses (built 2022): 15 HESS lighTram 10 Plug (10.80 m), 22 MAN Lion's City 12CE (12.20 m, 480 kWh), 22 MAN Lion's City 18CE (18.10 m, 640 kWh). VBZ uses battery buses on "Quartier- und Standardbuslinien" (neighbourhood and standard city bus lines). The VBZ sources read do not say which lines run with battery buses. | Depot charging only, CCS type 2 plug, 100 kW (lighTram) or 100-150 kW (MAN), chargers by Siemens. | VBZ web pages *Depots und Garagen* (modified 6 Aug 2026), *Batteriebusse* (modified 1 Sep 2025), *Elektromobilität* (modified 27 Jan 2026) |
| **Stadtbus Winterthur** (city of Winterthur) | Depot Grüzefeld (Winterthur): 96 vehicles at night (2024), at its capacity limit; a second depot is being studied. | No battery buses found in service. The 14 Hess vehicles ordered in 2021 (11 double-articulated, 3 articulated) are trolleybuses with "Dynamic Charging" (in-motion charging): they run up to 50 % of a route without overhead wire. Battery buses are named only as a long-term need at Grüzefeld (11 Jun 2024). | In-motion charging under the overhead wire (not the target). | Stadt Winterthur media releases 27 May 2021 and 11 Jun 2024 |
| **SZU** (Sihltal Zürich Uetliberg Bahn), brand *Zimmerbergbus* | Bus garage Rütihof, Wädenswil: largest Zimmerbergbus depot, over 30 vehicles. | None in service at the time of the page: all diesel. Plan: the first 8-10 e-buses around the end of 2026; all diesel buses replaced by 2040. 25 day lines and 4 night lines. | Depot charging only; no opportunity chargers on the lines (SZU decision). | SZU web page *E-Bus-Strategie* (undated) |
| **PostAuto Winkel (ZH)** (Zürcher Unterland, near the airport) | PostAuto site Winkel. | PostAuto names Winkel among the sites from which "larger networks" are served mainly by electric buses since 2025. The sources read give no vehicle count and no line list for Winkel. | Not stated. | PostAuto media release 15 Apr 2026 (*2025 annual performance*); PostAuto news 7 Jul 2026 |
| **VZO** (Verkehrsbetriebe Zürichsee und Oberland) | Depots in Grüningen (head office, 40 places), Wetzikon (opened 2020), Rüti and Meilen. | 3 electric buses of about 100 buses (Aug 2024), used on short lines and only in the peak hours (about 3 h morning and 3 h evening). | Depot charging; 4-5 h per charge; range "knapp 200 km" (just under 200 km). Wetzikon depot to get chargers in stages until 2026. | NZZ, 22 Aug 2024, p. 13 (article by S. Hotz) |

Notes:

- VBG, line 759: the battery size differs between sources. The fact sheet of 12 Jul 2021 (a plan) gives 200 kWh; the story of 14 Mar 2023 (vehicles in service) gives 264 kWh. This file uses 264 kWh for the vehicles in service.
- VBG, number of e-buses per line (inference, not stated by VBG): 14 e-buses on 31 Dec 2025 (annual report p. 4) minus 8 on line 759 (1 pilot bus + 7 from Feb 2023, media release 8 Mar 2023) gives 6 eCitaro G for line 768.
- VBG, line 768 start: the story of 5 Sep 2025 says "since April 2025"; the annual report 2025, p. 7, says "since autumn 2025". The two may mean the first bus and the full line.
- The PostAuto e-buses in Affoltern am Albis were at first not bound to fixed lines. PostAuto used them "flexibly" on several lines with different elevation profiles and lengths (media release 28 Mar 2025).
- ZVV plans battery buses charged in the depot, and more trolleybus lines in the large cities (NZZ, 22 Aug 2024). ZVV estimates CHF 440 million for e-buses and chargers in 2025-2029 (same source).

- **Trolleybuses, recorded apart (not the target).** VBZ: about 70 trolleybuses, all with traction batteries since 2017, on lines 31, 32, 33, 46, 72 and 83 (line 83 converted from diesel in 2020); lines 69 and 80 are to become trolleybus lines with sections without wire (VBZ *Elektromobilität*, *Depots und Garagen*). Stadtbus Winterthur: in-motion-charging trolleybuses on lines 1, 2, 2E and 3, later also 5 and 7 (media release 27 May 2021).
- VBZ, MAN Lion's City 18CE: the page gives 22 vehicles but the vehicle numbers 500-503 and 474-497 (28 numbers). This file keeps the count 22 as stated.
- VBZ also lists lines with regional numbers at Hardau (161, 165, 184, 185, 304) and Hagenholz (743, 744, 745). Their routes were not checked, and no source read says whether battery buses serve them.
- Operators not checked (no time in this round): Limmat Bus AG, AHW Busbetriebe, H. Baumgartner AG, PostAuto in other parts of the canton, Stadtbus Winterthur after June 2024.

## 2. Candidate depots (task 2)

Empty mass: no source read gives the empty mass of any of these battery-electric vehicle types (see section 6).

| | **A. EUROBUS welti-furrer AG, Bassersdorf** (for VBG) | **B. VBZ bus garage Hardau, Zurich** | **C. Rolf Stutz AG, Affoltern am Albis** (for PostAuto) |
|---|---|---|---|
| Operator | EUROBUS welti-furrer AG, bus contractor of VBG (VBG *Partner* page) | VBZ, city of Zurich (*Depots und Garagen*) | Rolf Stutz AG, PostAuto contractor (PostAuto release 28 Mar 2025) |
| Depot location | Bassersdorf, Glattal (the "Busdepot in Bassersdorf": VBG fact sheet 2021, p. 1) | Hardau, Zurich city (*Depots und Garagen*) | Affoltern am Albis, Knonauer Amt; new garage opened end of March 2025, 2,033 m², room for 26 articulated buses (PostAuto release 28 Mar 2025) |
| Lines served | 15 lines: 739, 742, 749, 759, 766, 768, 769, 771, 772, 773, 774, 781, 787, 795, 797 (VBG *Partner* page). Character from the timetable (section 3): urban 768, 781, 787; suburban 739, 742, 749, 759, 766, and the town or village lines 771-774 and 795; rural 769, 797. | 26 lines and night buses, of which 6 trolleybus lines (*Depots und Garagen*). City lines; the regional line numbers 161, 165, 184, 185, 304 were not checked. | Not named: the e-buses ran "flexibly" on several lines with different elevation profiles and lengths at the start (PostAuto release 28 Mar 2025). Character not checked. |
| BEB types | Mercedes eCitaro standard bus (length not stated), 264 kWh, range about 100-150 km, line 759 (VBG story 14 Mar 2023). Mercedes eCitaro G (articulated, over 18 m), 550 kWh usable, range about 350 km, line 768 (VBG story 5 Sep 2025). | HESS lighTram 10 Plug, 10.80 m; MAN Lion's City 12CE, 12.20 m, 480 kWh; MAN Lion's City 18CE, 18.10 m, 640 kWh (VBZ *Batteriebusse*). | 18 m articulated battery buses, range at least 250 km; model not named (PostAuto release 28 Mar 2025). |
| Charging | Line 759: opportunity charging at Zürich Flughafen, Bahnhof (pantograph from a mast; 300 kW in the 2021 plan) and 40 kW depot charger (fact sheet 2021). Line 768: depot charging, CCS Combo-2 plug, about 4 h for a full charge (story 5 Sep 2025). | Depot only: 105 charging points of 100 or 150 kW at Hardau and Herdern (*Depots und Garagen*); CCS type 2 (*Batteriebusse*). | Depot only, mostly at night, also in long breaks; 12 charging points at first, 26 at full build-out (release 28 Mar 2025). |
| Number of e-buses | 14 VBG e-buses on 31 Dec 2025 (annual report 2025, p. 4), all on lines 759 and 768 of this contractor (*Glattalbahn und Fahrzeugflotte*): 8 on line 759 (media release 8 Mar 2023) and, by difference, 6 on line 768 (inference). | 59 battery buses of VBZ (15 + 22 + 22, *Batteriebusse*); how many are at Hardau is not stated. | 13 (release 28 Mar 2025; PostAuto news 7 Jul 2026). |
| Total fleet of the depot | Not found in the sources read. Estimate from the timetable: 48 vehicles in service on a weekday (sum of n_j, section 3), without reserve vehicles. | About 180 vehicles of many types, trolleybuses included (*Depots und Garagen*). | Not found (garage room: 26 articulated buses). |

### Recommendation (task 3)

**Recommended: depot A, EUROBUS welti-furrer AG in Bassersdorf (VBG).**

| Criterion | A. Bassersdorf | B. Hardau | C. Affoltern am Albis |
|---|---|---|---|
| E-buses in regular service | Yes: lines 759 (since Feb 2023) and 768 (since 2025) | Yes: 59 battery buses (lines not stated) | Yes: 13 since 2025 |
| Mix of city and countryside lines | Yes, from one depot: urban lines into Zürich Oerlikon (768, 781, 787), suburban lines, and two rural lines (769 up to 598 m, 797) | City lines; regional lines not checked | Regional lines only; no city lines in the sources read |
| Public data | Full: timetable of all 15 lines in the national GTFS feed; e-bus counts per line from VBG | Timetable public; e-bus use per line not public in the sources read | Timetable public; e-bus use per line not fixed |

Reasons:

1. Both kinds of e-bus line run from this one depot today: an urban line (768) and a long suburban line (759, 14 km per direction).
2. The depot also runs two rural lines (769, 797) and short town lines. The wear of a bus then depends much on the line: from 0.97 km and 4 stops (774) to 14.08 km and 28 stops (759); ascent per trip from 0 m to 196 m.
3. The timetable reproduces the e-bus counts that VBG reports: the vehicle estimate gives n_j = 8 for line 759 and 6 for line 768 on a weekday, and VBG has 8 + 6 = 14 e-buses (section 3).
4. VBG plans more e-buses: about 40 by end of 2027 (story 5 Sep 2025) or about 20 (annual report 2025, p. 12). A case study with an all-electric depot fleet is therefore realistic.

Limit: today, only the urban line and one suburban line run with batteries. The rural lines still run with diesel or hybrid buses. No depot found has battery buses on city lines and rural lines at the same time; depot A is the closest option.

## 3. Lines of the recommended depot (task 4)

Depot A, EUROBUS welti-furrer AG, Bassersdorf: the 15 lines that the VBG *Partner* page assigns to it. The full table, with more columns, is `zurich_line_features.csv`; `extract_line_features.py` rebuilds it (section 5 gives the commands).

**Timetable used:** national GTFS feed, version 20260527, timetable year 2026 (14 Dec 2025 - 12 Dec 2026). Days: weekday Tuesday 16 Jun 2026; Saturday 20 Jun 2026; Sunday 21 Jun 2026. Only VBG trips (GTFS agency_id 773) are counted.

**Sources of the columns:**

| Column | Source |
|---|---|
| Termini, stops, running time, trips, first and last departure | GTFS `stop_times.txt`, `trips.txt`, `calendar*.txt`, main stop pattern per direction (the most frequent one on the weekday) |
| Length | OpenStreetMap road of the matching bus route relation, cut at the first and the last stop (crowd-sourced, not confirmed by VBG); where no relation fits, straight lines between stops (marked) |
| Ascent, descent, steepest grade, height range | swisstopo profile service (api3.geo.admin.ch, height model COMB) along the same geometry, smoothed (section 5) |
| n_j, vehicle-km, km per vehicle | computed from the GTFS trips (section 5: formulas and assumptions) |
| Character | own classification from the indicators of the table (section 5) |

d0 and d1 are the two directions; d0 runs from the first terminus named to the second.

**Route features (one trip of the main pattern):**

| Line | Termini | Character | Today | Length d0 / d1 (km) | Stops d0 / d1 | Stops per km | Run time d0 / d1 (min) | Mean speed (km/h) | Ascent d0 / d1 (m) | Descent d0 / d1 (m) | Steepest grade (%) | Height range (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 739 | Zürich Flughafen, REGA - Zürich Flughafen, Bahnhof | suburban (airport) | diesel or hybrid | 2.08 / 2.13 | 4 / 4 | 1.9 | 5.0 / 5.0 | 25.3 | 8 / 10 | 11 / 9 | 2.5 | 430-435 |
| 742 | Rümlang, Bahnhof - Zürich, Seebach | suburban | diesel or hybrid | 4.65 / 4.44 | 9 / 9 | 1.98 | 11.0 / 9.0 | 27.3 | 46 / 43 | 44 / 46 | 4.9 | 430-466 |
| 749 | Bassersdorf, Bahnhof Süd - Dietlikon, Hinentalstrasse | suburban | diesel or hybrid | 4.84 / 4.88 | 9 / 9 | 1.85 | 10.0 / 11.0 | 27.8 | 54 / 37 | 37 / 59 | 9.3 | 438-476 |
| **759** | Wangen b. Dübendorf, Dorfplatz - Zürich Flughafen, Bahnhof | suburban | **BEB** (8 eCitaro, opportunity charging) | 14.08 / 13.56 | 28 / 27 | 1.99 | 38.0 / 38.0 | 21.8 | 70 / 77 | 84 / 65 | 5.3 | 424-451 |
| 766 | Bassersdorf, Bahnhof - Kloten, Bahnhof | suburban | diesel or hybrid | 4.65 / 4.55 | 11 / 10 | 2.28 | 11.5 / 10.5 | 25.1 | 19 / 27 | 28 / 18 | 3.1 | 438-461 |
| **768** | Zürich Flughafen, Bahnhof - Zürich Oerlikon, Bahnhof | urban | **BEB** (6 eCitaro G, depot charging) | 5.66 / 5.62 | 13 / 13 | 2.31 | 16.0 / 16.0 | 21.1 | 54 / 40 | 42 / 50 | 5.3 | 418-443 |
| 769 | Breite b. Nürensdorf, Winterthurerstr. - Bassersdorf, Bahnhof | rural | diesel or hybrid | 4.42 / 8.59 | 6 / 11 | 1.31 | 8.0 / 18.0 | 30.0 | 6 / 196 | 137 / 63 | 10.2 | 454-598 |
| 771 | Wallisellen, Schäfligraben - Wallisellen, Bahnhof | suburban (town line) | diesel or hybrid | 2.02 / 2.10 | 8 / 8 | 3.88 | 6.0 / 6.0 | 20.6 | 3 / 45 | 46 / 3 | 5.2 | 430-474 |
| 772 | Wallisellen, Sportzentrum - Wallisellen, Bahnhof | suburban (town line) | diesel or hybrid | 2.33 / 2.08 | 9 / 9 | 4.08 | 8.0 / 7.0 | 17.6 | 8 / 28 | 30 / 6 | 4.2 | 429-456 |
| 773 | Wallisellen, Frohheimstrasse - Wallisellen, Bahnhof | suburban (town line) | diesel or hybrid | 1.54 / 1.28 | 4 / 4 | 2.84 | 4.0 / 4.0 | 21.1 | 4 / 35 | 38 / 2 | 7.9 | 429-466 |
| 774 | Wallisellen, Schulhaus Bürgli - Wallisellen, Bahnhof | suburban (town line) | diesel or hybrid | 0.97 / 1.16 | 4 / 4 | 3.75 | 4.0 / 4.0 | 16.0 | 0 / 41 | 32 / 9 | 9.9 | 430-463 |
| 781 | Glattbrugg, Giebeleichstrasse - Zürich Oerlikon, Bahnhof | urban | diesel or hybrid | 3.89 / 4.28 | 12 / 12 | 2.94 | 12.0 / 13.0 | 19.6 | 22 / 17 | 13 / 26 | 4.7 | 424-443 |
| 787 | Brüttisellen, Obere Wangenstr. - Zürich Oerlikon, Bahnhof | urban | diesel or hybrid | 9.73 / 9.59 | 19 / 19 | 1.97 | 29.0 / 29.0 | 20.0 | 46 / 56 | 56 / 44 | 5.3 | 422-454 |
| 795 | Rümlang, Chilestieg - Rümlang, Ifangstrasse | suburban (village line) | diesel or hybrid | 2.15 / 2.63 (d1: straight lines) | 5 / 7 | 2.51 | 13.0 / 16.0 | 9.9 | 7 / 49 | 37 / 19 | 6.4 | 427-465 |
| 797 | Oberhasli, Dorf - Rümlang, Bahnhof | rural | diesel or hybrid | 4.29 / 6.22 | 4 / 8 | 1.14 | 7.0 / 13.0 | 31.5 | 23 / 58 | 32 / 51 | 7.8 | 419-466 |

"Today": the two BEB lines from section 1. For the other lines, the sources say only that the rest of the VBG fleet runs with hybrid or diesel buses (*Glattalbahn und Fahrzeugflotte*).

**Daily service** (n_j and the km values are estimates, section 5):

| Line | Trips Tue / Sat / Sun | n_j Tue / Sat / Sun | n_j check, Tue (cycle time / headway 07-08) | Vehicle-km per day Tue / Sat / Sun | km per vehicle per day Tue / Sat / Sun | First - last departure, Tue |
|---|---|---|---|---|---|---|
| 739 | 110 / 0 / 0 | 2 / 0 / 0 | 2 (15 min) | 227.8 / 0 / 0 | 113.9 / - / - | 06:15 - 19:53 |
| 742 | 78 / 74 / 70 | 2 / 2 / 2 | 1 (30 min) | 354.6 / 336.5 / 318.3 | 177.3 / 168.2 / 159.1 | 05:24 - 00:25 |
| 749 | 26 / 0 / 0 | 2 / 0 / 0 | 1 (30 min) | 126.5 / 0 / 0 | 63.2 / - / - | 06:01 - 19:16 |
| **759** | 147 / 140 / 79 | **8** / 7 / 4 | 6 (15 min) | 1,935.1 / 1,865.5 / 1,055.8 | 241.9 / 266.5 / 264.0 | 04:42 - 00:34 |
| 766 | 14 / 0 / 0 | 1 / 0 / 0 | 1 (30 min) | 59.0 / 0 / 0 | 59.0 / - / - | 06:41 - 17:35 |
| **768** | 244 / 204 / 194 | **6** / 4 / 4 | 6 (7.5 min) | 1,375.5 / 1,150.2 / 1,093.8 | 229.2 / 287.5 / 273.5 | 04:46 - 00:20 |
| 769 | 18 / 0 / 0 | 3 / 0 / 0 | 2 (30 min) | 119.0 / 0 / 0 | 39.7 / - / - | 06:03 - 19:06 |
| 771 | 136 / 122 / 70 | 3 / 3 / 2 | 2 (15 min) | 272.4 / 250.4 / 144.1 | 90.8 / 83.5 / 72.0 | 05:44 - 00:24 |
| 772 | 84 / 71 / 70 | 2 / 2 / 2 | 2 (15 min) | 180.5 / 148.4 / 121.1 | 90.3 / 74.2 / 60.6 | 05:44 - 00:15 |
| 773 | 20 / 0 / 0 | 1 / 0 / 0 | 1 (30 min) | 28.2 / 0 / 0 | 28.2 / - / - | 06:16 - 19:09 |
| 774 | 62 / 52 / 0 | 2 / 2 / 0 | 1 (30 min) | 66.1 / 55.4 / 0 | 33.0 / 27.7 / - | 05:56 - 21:00 |
| 781 | 243 / 152 / 152 | 6 / 3 / 3 | 5 (7.5 min) | 990.0 / 619.9 / 619.9 | 165.0 / 206.6 / 206.6 | 05:00 - 00:40 |
| 787 | 133 / 120 / 67 | 5 / 5 / 3 | 5 (15 min) | 1,278.2 / 1,153.2 / 643.9 | 255.6 / 230.6 / 214.6 | 05:57 - 23:50 |
| 795 | 23 / 0 / 0 | 2 / 0 / 0 | 2 (30 min) | 52.2 / 0 / 0 | 26.1 / - / - | 06:13 - 19:13 |
| 797 | 64 / 62 / 26 | 3 / 3 / 2 | 1 (30 min) | 336.3 / 325.8 / 107.7 | 112.1 / 108.6 / 53.9 | 05:40 - 21:09 |
| **Sum** | 1,402 / 997 / 728 | 48 / 31 / 22 | | 7,401.4 / 5,905.3 / 4,104.6 | | |

Departures after midnight are written 00:xx here (the CSV keeps the GTFS form, for example 24:25).

**Checks against operator figures:**

| Check | Operator figure | This table |
|---|---|---|
| Line 759, length | "über 27 Kilometer" (over 27 km) for the line (VBG story 14 Mar 2023) | 14.08 + 13.56 = 27.64 km for one round trip: the VBG figure fits the round trip |
| Line 759, km per year | about 635,000 km per year with diesel buses (fact sheet 2021, p. 1, earlier timetable) | 250 × 1,935.1 + 52 × 1,865.5 + 63 × 1,055.8 ≈ 647,000 km (assumption: 250 weekdays, 52 Saturdays, 63 Sundays and holidays) |
| Line 759, vehicles | 8 e-buses (1 pilot bus + 7, media release 8 Mar 2023) | n_j = 8 on a weekday |
| Lines 759 + 768, vehicles | 14 e-buses on 31 Dec 2025 (annual report 2025, p. 4) | 8 + 6 = 14 |
| Line 759, km per vehicle | "bis zu 400 km" (up to 400 km) per vehicle and day required (fact sheet 2021, p. 1) | mean 241.9 km on a weekday |

## 4. Data sources

| Name | URL | Licence | Date accessed | Size | What was used | Evidence basis |
|---|---|---|---|---|---|---|
| VBG, *Innovationslinie 759, Projekt Energieautarker Elektrobus*, fact sheet, 12 Jul 2021 | https://www.vbg.ch/wp-content/uploads/2025/07/innovationslinie-faktenblatt-e-bus-vbg.pdf | operator document, no licence stated | 2026-11-09 | 85,463 B | line 759 plan; battery 200 kWh; 300 kW pantograph charger at Flughafen; 40 kW depot charger; up to 400 km per day; 635,000 km per year with diesel | read in full (2 pp.) |
| VBG, *Geschäftsbericht 2025* (annual report 2025) | https://www.vbg.ch/wp-content/uploads/2026/06/vbg-gb25-final-web.pdf | operator document, no licence stated | 2026-11-09 | 1,205,085 B | p. 4 fleet and network figures; p. 7 line 768 electric; p. 8 passengers per line; p. 12 about 20 more e-buses planned; p. 25 bus contractors and their depots | read in full (26 pp.) |
| PostAuto, media release *Im Knonauer Amt fahren bald die ersten E-Postautos*, 28 Mar 2025 | https://www.postauto.ch/de/ueber-uns-und-aktuelles/medien/medienmitteilungen/2025/e-postautos-im-knonauer-amt | operator web page | 2026-11-09 | 56,752 B | Affoltern am Albis garage; 13 articulated BEB; range; depot charging | read in full |
| PostAuto, news *Das 200. E-Postauto ist auf der Strasse*, 7 Jul 2026 | https://www.postauto.ch/de/ueber-uns-und-aktuelles/aktuelles/2026/200-e-postauto | operator web page | 2026-11-09 | 63,578 B | largest e-bus sites (Affoltern am Albis 13 vehicles); Winkel (ZH) named | read in full |
| Neue Zürcher Zeitung, 22 Aug 2024, p. 13, *Die holprige Fahrt zum CO2-freien Busbetrieb* (S. Hotz), copy on the VZO site | https://vzo.ch/files/files/Neue_Zuercher_Zeitung_20240822_Seite_13.pdf | press article, copyright NZZ | 2026-11-09 | 123,199 B | VZO fleet, e-buses, depots, range | read in full (the VZO article; the page also holds an unrelated article) |
| VBG, story *Volle Fahrt voraus mit neuen Elektro-Gelenkbussen*, 5 Sep 2025 | https://www.vbg.ch/story/volle-fahrt-voraus-mit-neuen-elektro-gelenkbussen/ | operator web page | 2026-11-09 | 134,096 B | line 768: eCitaro G, over 18 m, 550 kWh usable, range about 350 km, CCS depot charging, about 4 h; driver at Eurobus Bassersdorf; about 40 more e-buses by end of 2027; all-electric by 2040 | read in full |
| VBG, page *E-Mobilität* | https://www.vbg.ch/themen-und-projekte/e-mobilitaet/ | operator web page | 2026-11-09 | 132,615 B | line 759 all-electric since end of Feb 2023; opportunity charging at Flughafen, Bahnhof; e-buses replace diesel buses 1:1 | read in full |
| VBG, story *Willkommen auf der längsten Elektrobus-Linie der Schweiz*, 14 Mar 2023 (date from the page metadata) | https://www.vbg.ch/story/willkommen-auf-der-laengsten-elektrobus-linie-der-schweiz/ | operator web page | 2026-11-09 | 133,936 B | line 759: over 27 km; eCitaro standard bus; 264 kWh; range about 100-150 km; pantograph charging at the airport | read in full |
| VBG, page *Partner* (bus contractors), modified 10 Aug 2026 (page metadata) | https://www.vbg.ch/ueber-die-vbg/partner/ | operator web page | 2026-11-09 | 127,167 B | contractors, their sites and the lines each one runs | read in full |
| VBG, media release *VBG-Linie 759 vollständig elektrifiziert*, 8 Mar 2023 | https://www.vbg.ch/news/vbg-linie-759-vollstaendig-elektrifiziert/ | operator web page | 2026-11-09 | 121,344 B | line 759: pilot from autumn 2021; 7 more e-buses at end of Feb 2023; partners Siemens and EvoBus | read in full |
| VBG, page *Glattalbahn und Fahrzeugflotte*, modified 7 Jul 2026 (page metadata) | https://www.vbg.ch/ueber-die-vbg/glattalbahn-und-fahrzeugflotte/ | operator web page | 2026-11-09 | 129,004 B | key figures on 31 Dec 2025 (56 bus lines, 268.5 km, 388 stops); 120 buses of the contractors; e-buses on lines 759 and 768 | read in full |
| SZU, page *E-Bus-Strategie* (undated) | https://www.szu.ch/de/unsere-zukunftsformel/e-bus-strategie/ | operator web page | 2026-11-09 | 30,494 B | Zimmerbergbus: diesel today; first 8-10 e-buses about end of 2026; depot Rütihof over 30 vehicles; depot charging only | read in full |
| Stadt Winterthur, media release *Mehr Platz für Busse im Grüzefeld*, 11 Jun 2024 | https://stadt.winterthur.ch/stadtbus/medien/medienmitteilungen/mehr-platz-fuer-busse-im-gruezefeld | city web page | 2026-11-09 | 68,612 B | depot Grüzefeld: 96 vehicles; battery buses only a long-term need | read in full |
| Stadt Winterthur, media release *Bestellung von vierzehn Elektrobussen für Winterthur*, 27 May 2021 | https://stadt.winterthur.ch/stadtbus/medien/medienmitteilungen/bestellung-von-vierzehn-elektrobussen-fuer-winterthur | city web page | 2026-11-09 | 67,137 B | the 14 Hess vehicles are trolleybuses with Dynamic Charging | read in full |
| PostAuto, media release *2025 annual performance*, 15 Apr 2026 | https://www.postauto.ch/en/about-us-and-news/media/press-releases/2026/2025-annual-performance | operator web page | 2026-11-09 | 70,778 B | 166 of 2,390 PostAuto buses electric at end of 2025; Winkel (ZH) among the sites served mainly by e-buses | read in full |
| VBZ, page *Depots und Garagen*, modified 6 Aug 2026 (page metadata) | https://www.stadt-zuerich.ch/vbz/de/die-vbz/depots-und-garagen.html | city web page | 2026-11-09 | 61,158 B | 256 buses; Hardau: 105 chargers, about 180 vehicles, line list; Hagenholz line list; regional garages | read in full |
| VBZ, page *Nachhaltige Infrastruktur* | https://www.stadt-zuerich.ch/vbz/de/die-vbz/umwelt/nachhaltige-infrastruktur.html | city web page | 2026-11-09 | 28,523 B | nothing used (buildings and solar power only) | read in full |
| VBZ, page *Elektromobilität*, modified 27 Jan 2026 (page metadata) | https://www.stadt-zuerich.ch/vbz/de/mobilitaet-im-wandel/elektromobilitaet.html | city web page | 2026-11-09 | 25,089 B | strategy *eBus VBZ*; battery buses on neighbourhood and standard lines, charged in the garage; trolleybus facts | read in full |
| VBZ, pages *Fahrzeuge* and *Busse* (index pages) | https://www.stadt-zuerich.ch/vbz/de/die-vbz/fahrzeuge.html, .../fahrzeuge/busse.html | city web pages | 2026-11-09 | 17,054 B; 17,470 B | links only | read in full |
| VBZ, page *Batteriebusse*, modified 1 Sep 2025 (page metadata) | https://www.stadt-zuerich.ch/vbz/de/die-vbz/fahrzeuge/busse/batteriebusse.html | city web page | 2026-11-09 | 25,686 B | battery bus types, counts, lengths, batteries, charging | read in full (the data tables are inside the page code) |
| VBZ, page *Midibusse* (path .../busse/ebusse.html), modified 1 Sep 2025 | https://www.stadt-zuerich.ch/vbz/de/die-vbz/fahrzeuge/busse/ebusse.html | city web page | 2026-11-09 | 19,277 B | nothing used (3 diesel midibuses) | read in full |
| SKI+ (Systemaufgaben Kundeninformation, the Swiss public-transport customer-information office), *Timetable 2026 (GTFS2020)*, feed version 20260527 (`feed_info.txt`), valid 14 Dec 2025 - 12 Dec 2026. Copy hosted by the Mobility Database (MobilityData, a non-profit open-data catalogue), snapshot of 29 May 2026, ID mdb-2898, which names as producer URL https://data.opentransportdata.swiss/dataset/timetable-2026-gtfs2020/permalink | https://files.mobilitydatabase.org/mdb-2898/mdb-2898-202605290027/mdb-2898-202605290027.zip | terms of use of opentransportdata.swiss (not read: the site was blocked, see section 6) | 2026-11-09 | 162,613,673 B zipped (stop_times.txt 1,789,073,567 B unzipped) | `agency.txt`, `routes.txt`, `trips.txt`, `stop_times.txt`, `stops.txt`, `calendar.txt`, `calendar_dates.txt`, `feed_info.txt`: all VBG trips (agency_id 773: 65 routes, 25,266 trips, 287,888 stop times, 1,039 stops) | read in full for agency 773 (every row of each file was scanned; the other agencies were skipped) |
| OpenStreetMap, bus route relations of the 15 Bassersdorf lines in the canton of Zurich, through the Overpass API (database state 2026-11-09T16:43:51Z) | https://overpass-api.de/api/interpreter (query in `extract_line_features.py` notes, section 3) | ODbL 1.0 (OpenStreetMap contributors) | 2026-11-09 | 1,076,464 B | road geometry of each line (crowd-sourced, see section 5) | read in full |
| swisstopo, profile service of the federal geoportal API (`profile.json`), 30 requests, one per line and direction, plus one test request (download 37, line 768 on straight lines, not used) | https://api3.geo.admin.ch/rest/services/profile.json (parameters: geometry in LV95, `sr=2056`, `nb_points`, `distinct_points=true`; the full URLs are in `log.tsv` and in `patterns.csv` of the work folder) | swisstopo terms of use (not read) | 2026-11-09 | 9,973-104,151 B each (downloads 38-43, 46-67, 70, 71) | heights (model COMB) along each line | read in full (every sample used) |

## 5. Assumptions and estimates

Each item is an assumption or an estimate of this study, not a figure of an operator.

| Item | Formula or rule | Why |
|---|---|---|
| **Main pattern** of a direction | the most frequent sequence of stop places on the weekday (GTFS `parent_station` merges the platforms of one place) | one representative trip per direction; short trips and variants count in the daily km |
| **Length** (estimate, crowd-sourced geometry) | road length along the OSM bus route relation of the same line number in the ZVV network, cut at the first and the last stop. A relation is accepted when it passes all the stops in order, every intermediate stop within 50 m, both termini within 150 m, and no gap between two ways over 10 m; among accepted relations the shortest road wins. Accepted for 29 of 30 directions; line 795, d1 (a loop), uses straight lines between stops, a lower bound. | the GTFS feed has no road geometry. OSM is crowd-sourced: the lengths are unconfirmed by VBG. The round trip of line 759 (27.64 km) fits the VBG figure "over 27 km". |
| **Mean speed** | (length_d0 + length_d1) / (run time_d0 + run time_d1); run time = median, over the weekday trips of the main pattern, of last arrival - first departure | dwell times at stops are included; layovers at the termini are not |
| **Elevation** | swisstopo profile (model COMB) along the road, one sample about every 10 m; resampled every 20 m and smoothed by a moving average over 100 m. Ascent = sum of the rises, descent = sum of the falls. Steepest grade = largest height change over 100 m of road, in % | the height model is the terrain, not the road deck: bridges and underpasses give false steps; smoothing damps them. Grades over about 8 % (749, 769, 774) should be checked on a map before use |
| **n_j** (estimate) | minimum number of vehicles that serve all trips of the line on that day: n_j = trips - (largest set of pairs "trip i, then trip j on the same vehicle"). A pair is allowed when arrival_i + 3 min + empty-run time ≤ departure_j; empty-run time = 1.3 × straight distance / 20 km/h between two different stop places | the GTFS `block_id` does not give the vehicle blocks (section 6). The line is taken alone: no bus serves two lines in one day, and there is no reserve. With 0 min or 6 min layover, line 759 needs 7 or 8 vehicles and line 768 needs 6 or 8 |
| **n_j check** | n = ceil((run time_d0 + run time_d1 + 2 × 3 min) / headway), headway = 60 / departures of direction d0 between 07:00 and 08:00 on the weekday | a second estimate from the classic cycle-time rule. It is lower where the timetable has extra peak trips or long layovers |
| **Vehicle-km per day** (estimate) | sum over the day's trips of the trip's straight stop-to-stop length × (road length / straight length of the main pattern of its direction) | the road factor of the main pattern is applied to the variants; empty runs to and from the depot are not counted |
| **km per vehicle per day** (estimate) | vehicle-km per day / n_j | a mean; the real vehicle blocks may share km unevenly |
| **Character** (own classification) | urban: at least 40 % of the stops in the city of Zurich (stop names "Zürich, ..." or "Zürich Oerlikon, ...") and mean speed ≤ 22 km/h. Rural: fewer than 1.5 stops per km and villages only. Suburban: the rest; "town line" or "village line" when all stops lie in one municipality | an explicit rule from the table's own indicators. Note: the stops named "Zürich Flughafen" are not counted as city stops |
| **Days** | Tuesday 16 Jun 2026, Saturday 20 Jun 2026, Sunday 21 Jun 2026 | ordinary days of the timetable year, outside the Christmas period |
| **Number of e-buses per line** (inference) | line 768: 14 (VBG total, 31 Dec 2025) - 8 (line 759) = 6 | VBG does not state it; it agrees with n_j = 6 |
| **Depot fleet** (estimate) | sum of n_j on a weekday = 48 vehicles, without reserve and without interlining | the operator's fleet of the depot was not found |

**How to rebuild the table** (all inputs are the downloads of section 4; the script downloads nothing):

```
python -I extract_line_features.py filter <GTFS zip> <work dir> --agency 773
python -I extract_line_features.py geoms <work dir> --lines 739,742,749,759,766,768,769,771,772,773,774,781,787,795,797 --date 20260616 --osm <Overpass JSON>
(download each URL of <work dir>/patterns.csv into its own folder)
python -I extract_line_features.py table <work dir> <30 profile JSON files, in the order of patterns.csv> --lines <same list> --weekday 20260616 --saturday 20260620 --sunday 20260621 --out zurich_line_features.csv
```

Defaults of the `table` step: layover 3 min (`--layover-min`), empty runs at 20 km/h (`--empty-kmh`) with detour factor 1.3 (`--detour`).

## 6. Could not obtain

| Item | Reason |
|---|---|
| GTFS files and data-set pages on opentransportdata.swiss (`data.opentransportdata.swiss`, `opentransportdata.swiss`) | Every request got HTTP 403 from the local antivirus web filter of this workstation (response header `Server: Avast WebShield [malicious URL blocked]`), 2026-11-09. The filter was not changed. The same official feed was taken from the Mobility Database copy (section 4). The user may want to check the filter rule. |
| Vehicle blocks (*Umläufe*) of the VBG lines | The GTFS field `block_id` is filled for some lines only, and mostly joins two trips (line 759 on 16 Jun 2026: 132 of 147 trips with a block id, 66 block ids). It marks trips that run through, not day-long vehicle blocks. n_j is therefore estimated (section 5). |
| Stadt Winterthur PDF media releases of 27 May 2021 and 11 Jun 2024 (`.../20210527-mm-bestellung-elektrobusse.pdf/download`, `.../20260611-mm-depot-gruezefeld.pdf/download`); PostAuto release `.../2026/jahresergebnis-2025` | HTTP 404. The HTML pages of the same releases were read instead. |
| opendata.swiss record of the GTFS data set (licence field) | The CKAN API of opendata.swiss answered, but its licence field is empty; the terms of use page of opentransportdata.swiss was blocked (row 1). |
| Empty mass of the e-bus types (eCitaro, eCitaro G, lighTram 10 Plug, MAN Lion's City 12CE and 18CE, PostAuto articulated buses) | Not given in any source read. A manufacturer data sheet would be needed; none was read. |
| Total fleet of the Bassersdorf depot, and its diesel and hybrid buses per line | Not given by VBG or EUROBUS in the sources read. Section 5 gives a timetable estimate (48 vehicles). |
| Road geometry from an official source | The national GTFS feed has no `shapes.txt`. OpenStreetMap was used instead (unconfirmed, section 5). |
| VBZ: which lines run with battery buses | Not stated on the VBZ pages read. |
| PostAuto Winkel: e-bus count and lines | Not stated in the PostAuto pages read. |
| Downloads 03, 04, 07, 08, 10, 11 of the earlier rounds | The folders are empty and `log.tsv` has no entry for them; their URLs are not recorded. |
| swisstopo profiles of line 759 at the first try (downloads 44, 45, 68, 69: empty folders) | HTTP 400: the service refuses request lines over 4,094 characters. The geometry was simplified more (tolerance 6.8 m) and downloaded again (70, 71). |
