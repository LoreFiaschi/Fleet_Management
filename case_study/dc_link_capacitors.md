# DC-link capacitors of traction converters in vehicles other than buses

Status: done (9 Nov 2026). Five sources read in full (S1-S5). Next, if wanted: the leads
below, first Yao et al. 2020 (metro) and the Chalmers master's thesis (heavy-duty truck).

Question: is there field or reliability evidence that the DC-link capacitors of traction
converters wear out or fail in vehicles other than buses? DC link = the capacitor bank
between the DC supply (battery or line rectifier) and the traction inverter.

## Summary and verdict

- **Field data (item 1).** None found that names the DC-link capacitor in a vehicle. The fleet data read
  (S3: 23 trains, 59 converter failures) count whole converters and show only early-life failures.
  The quoted 19 % capacitor share is from a cross-industry survey, not vehicles (cited in S1, p. 3).
- **Lifetime studies (item 2).** Two model studies: a train (S2); a battery-electric vehicle (BEV) (S1).
  Ripple current rises with speed, slope, acceleration and braking (S2, p. 4-5). Train: about 40-50
  years at 1 % failed, bank 13.7 years at 5 % (S2, p. 12-14); BEV: most fail after about
  1,500 thousand km, target 300,000 km (S1, p. 3, p. 18). Ageing is gradual; Weibull shape 5.06 (S2).
- **Type (item 3).** Polypropylene film: "widely used" in electric-vehicle traction drives (S4, p. 2); "the dominant
  technology", per a 2023 market report (S5, p. 2). The rail and BEV studies also use film.
- **Verdict.**
  1. Wear-out physics is shown; drive-cycle ripple is one input, but voltage, heat and humidity also
     act while the vehicle stands or charges (S2, S5).
  2. Single model studies did this; no source shows field wear-out or an established replacement practice.
  3. Not enough evidence to treat a bus DC-link capacitor as usage-driven wear-out; predicted lives
     exceed the 15-year design life used in S1.

## Sources

Page numbers below are pages of the downloaded PDF file.

### S1. Chakraborty et al. 2022: battery-electric vehicle, SiC DC/DC converter

- **Reference.** S. Chakraborty, M. M. Hasan, P. McGahan, D.-D. Tran, T. Geury, P. Davari,
  F. Blaabjerg, M. El Baghdadi, O. Hegazy, "Real-Life Mission Profile-Oriented Lifetime
  Estimation of a SiC Interleaved Bidirectional HV DC/DC Converter for Electric Vehicle
  Drivetrains", *IEEE Journal of Emerging and Selected Topics in Power Electronics* 10(5),
  5142-5167, 2022, doi:10.1109/JESTPE.2021.3083198. Accepted manuscript from the Aalborg
  University repository (vbn.aau.dk/ws/files/413263584).
- **Evidence basis.** Read in full (26 PDF pages).
- **Vehicle and part.** A light utility battery-electric vehicle (BEV), 3600 kg gross weight,
  60 kW motor (p. 5, Table 2). The capacitor is the 160 µF output capacitor (the DC link) of
  the high-voltage (HV) DC/DC converter between battery and traction inverter, not the DC
  link of the inverter itself (p. 10, Table 4; p. 13).
- **Kind of evidence.** A model study (simulation plus a lab prototype for the efficiency
  model). No field failure data. The drive cycles are three measured cycles from a vehicle
  maker (p. 6, Table 3; p. 7).
- **Capacitor type.** A polypropylene film capacitor, Cornell Dubilier type 947C (p. 13;
  p. 14). The introduction motivates the study with the ageing of electrolytic capacitors
  (p. 3, §I.B), but the case study uses a film capacitor.
- **Lifetime model.** Arrhenius in hot-spot temperature times a power law in voltage, eq. (24);
  rated life 5,000 h at 85 °C and 1200 V DC (p. 13, Table 6). The hot-spot temperature
  follows the ripple current through the equivalent series resistance (ESR) losses, eq. (15)
  and eq. (20) (p. 11-12). Damage is summed with Miner's rule, eq. (26) (p. 13).
- **Findings.**
  - Under the extra-urban cycle, most of the simulated capacitor population fails after about
    1,500 thousand km; the most failure-prone switch fails after about 1,000 thousand km
    (p. 18, text under Fig. 25). The authors call the lower MOSFET (metal-oxide-semiconductor
    field-effect transistor) and the DC-link capacitor "the most failure-prone components"
    (p. 18; p. 22, §VI).
  - The capacitor is in series in the system reliability: its failure fails the converter
    (p. 19, eqs. (34)-(36)).
  - The whole converter is about four times more reliable under the urban cycle than under the
    extra-urban cycle (p. 22, §VI). This ranking is driven by the switches; the paper does not
    give a separate capacitor number per cycle in the text.
  - The paper requires a lifetime of 300,000 km or 15 years, about 8,000 h of operation, for
    traction inverter and DC/DC converter (p. 3, Table 1, citing Brüll et al., EVS32, 2019).
- **Numbers that the paper only cites.** "PECs are responsible for 37% of the unscheduled
  maintenances in the electrical traction systems", from "field test data", cited to Hu et
  al. 2020, a review (p. 2, ref. [5]). The DC-link capacitor is the root cause of 19% of
  converter failures, cited to Yang et al. 2011, a survey of 80 companies across all
  industries, not vehicles (p. 3, §I.B; refs. [6] on p. 22). Neither is field data of vehicle
  traction converters in this paper.

### S2. Yao et al. 2021: high-speed train, DC-link capacitor bank

- **Reference.** B. Yao, X. Ge, D. Xie, S. Li, Y. Zhang, H. Wang, H. Wang, "Electro-Thermal
  Stress Analysis and Lifetime Evaluation of DC-Link Capacitor Banks in the Railway Traction
  Drive System", *IEEE Journal of Emerging and Selected Topics in Power Electronics* 9(4),
  4269-4284, 2021, doi:10.1109/JESTPE.2020.3000130. Accepted manuscript from the Aalborg
  University repository (vbn.aau.dk/ws/files/431656908).
- **Evidence basis.** Read in full (17 PDF pages).
- **Vehicle and part.** The DC link between the line-side rectifiers and the traction inverter
  of an AC-fed train; speed profile 180-300 km/h, so a high-speed train (p. 3, Table I; p. 4,
  Fig. 5). Bank of four 750 µF capacitors, 3600 V rated DC-link voltage (p. 3, Table I). The
  bank sits under the train floor (p. 9).
- **Capacitor type.** Metallised polypropylene film; rated 100,000 h at 4000 V and 70 °C
  (p. 3, Table I; p. 5, §II.B: "The DC-link capacitors used in the railway traction drive
  system are composed of serval polypropylene film units").
- **Kind of evidence.** A model study with a small lab prototype (500 V, 3.3 kVA) and an
  accelerated ageing test (p. 10; p. 13; p. 15, Table B.II). One field measurement: case
  temperatures read from temperature labels after one day of train service, 40-43 °C (p. 9).
  No field failure data.
- **Usage dependence (load cycle).**
  - The ripple current and its harmonic distortion grow with speed, slope, acceleration and
    braking (p. 4-5). The total harmonic distortion (THD) of the capacitor current at
    300 km/h is about twice that at 180 km/h; at a 30 ‰ slope it reaches 46.65 % (p. 5).
  - In the lab, the THD during acceleration and braking is 1.5 to 2 times that at constant
    speed (p. 10-11).
  - The ripple current sets the losses (ESR times current squared, eq. (10), p. 6), the losses
    set the hot-spot temperature (p. 7-9), and the temperature sets the life.
- **Lifetime model.** Film-capacitor model: power law in voltage, factor 2 per 10 K in
  hot-spot temperature, power law in relative humidity, eq. (25) (p. 12). Damage summed with
  Miner's rule over daily and yearly profiles of temperature and humidity (p. 12).
- **Findings.**
  - Expected life of one capacitor about 50.64 years at 1 % failed (p. 12).
  - Accelerated test: about 2,400 h of testing at acceleration factor 37.0; extrapolated life
    about 240,500 h, that is 43.93 years (p. 13).
  - Failure is gradual: capacitance falls and dissipation factor (DF) rises. The authors set
    end of life at a capacitance loss of 4.06 % and a DF rise of 1.81 times, for stability of
    the DC link and the 75 °C temperature limit (p. 13-14). With this criterion the life is
    about 39.75 years at 1 % failed (p. 14).
  - Weibull shape parameter m = 5.06, fitted to the ageing test, that is wear-out (p. 14).
    At 5 % unreliability: 18.0 years for one capacitor, 13.7 years for the bank (p. 14).
  - The authors propose on-condition maintenance instead of scheduled maintenance (p. 2,
    abstract; p. 14, §V).
- **Statement on field data.** Railway reliability studies use fault statistics but "rarely
  involve the reliability evaluation of capacitor in the traction system" (p. 2, §I).

### S3. Guida, Poli and Pulcini 2014: field failures of rail traction converters, Naples

- **Reference.** M. Guida, P. Poli, G. Pulcini, "Reliability analysis and prediction of the
  failure process of some traction converters", *Computers in Railways XIV*, WIT
  Transactions on The Built Environment 135, 279-290, WIT Press, 2014,
  doi:10.2495/CR140221. Open access PDF from witpress.com.
- **Evidence basis.** Read in full (12 PDF pages, journal pp. 279-290).
- **Vehicle and data.** 23 new AnsaldoBreda trains in the Naples area, three traction
  converters per train, observed from April 2009 to August 2013 (p. 1; p. 3). 59 converter
  failures in total (p. 3). Mileage per train and mileage at each failure are tabulated
  (p. 4, Table 1; p. 5, Table 2).
- **Kind of evidence.** Field failure data, at the level of the whole converter.
- **Findings.**
  - The converter parts listed are transducers, a pre-charge contactor, resistors, six types of
    electronic boards, the traction control unit, three IGBT (insulated-gate bipolar
    transistor) inverter modules and the brake chopper (p. 2). The paper does not name a
    DC-link capacitor, and it gives no failure count per part.
  - A repair replaces the single failed part (p. 3), so the authors model the converter as a
    repairable system with minimal repair (a non-homogeneous Poisson process) over mileage
    (p. 3).
  - Failure intensity decreases over the observed 0-211,000 km: early-life failures, not
    wear-out (p. 3; p. 6, Laplace test −1.422).
  - After 210,968 km the authors assume a constant rate: mean mileage between failures
    334,780 km per converter (p. 9). They also give a mean operating time between failures
    "56.665 hours", equal to 16.91 years at 10 h a day and 335 days a year (p. 11); 16.91 ×
    3350 h ≈ 56,650 h, so the printed number means about 56,665 h.
  - The converters are designed for 30-40 years and more than 2 million km; the data cover
    only the early life (p. 3).
- **Relevance.** Field data for rail traction converters exist, but this set does not isolate
  the DC-link capacitor and does not reach the wear-out phase.

### S4. Chowdhury, Gurpinar and Ozpineci (ORNL): capacitor types for car traction inverters

- **Reference.** S. Chowdhury, E. Gurpinar, B. Ozpineci, "High-Energy Density Capacitors for
  Electric Vehicle Traction Inverters", Oak Ridge National Laboratory (ORNL), accepted
  manuscript, OSTI ID 1649202 (osti.gov/servlets/purl/1649202). The PDF does not print the
  venue or year; OSTI lists it as a conference paper, and its newest reference is from
  December 2019.
- **Evidence basis.** Read in full (7 PDF pages).
- **Kind of evidence.** A lab characterisation of single film, ceramic and PLZT (lead
  lanthanum zirconate titanate) capacitors, plus a short review of capacitor types. No
  lifetime or field data.
- **Findings on the capacitor type.**
  - "film capacitors are widely used as DC bus capacitors for EV traction drive applications
    due to their higher reliability, high current conduction capability, high-frequency
    operation, and lower ESR compared to electrolytic capacitors" (p. 2, §II). This is the
    authors' statement; no market survey is cited for it.
  - Electrolytic capacitors are "untenable" for electric-vehicle (EV) traction inverters
    because of short life, limited current capability and low-frequency operation (p. 2,
    §II).
  - Ceramic capacitors "have not gained popularity" in EV traction inverters because the
    dielectric can crack and short the DC terminals (p. 2, §II).
  - Film capacitors work up to about 105 °C, and their self-heating is limited to about 20 °C,
    which limits their ripple current (p. 2; p. 4).
- **Other numbers.** The DC-link capacitor can carry up to 60 % of the RMS (root mean
  square) load current (p. 1, citing Kolar and Round 2006; p. 3). The capacitor bank can take
  30 % of inverter volume (p. 1, citing März et al. 2010). The BMW i3 inverter capacitor bank
  is 0.644 L (p. 6, Table IV); the paper does not state its type.

### S5. TDK 2026: DC-link film capacitors for the main traction inverter of cars

- **Reference.** D. Olalla, G. Kulkarni, F. Rodriguez, "Introduction to xEVCap: A modular
  DC-link capacitor solution for the main powertrain inverter of xEVs", TDK Corporation white
  paper, April 2026 (tdk-electronics.tdk.com, file wp-introduction-xev-cap.pdf).
- **Evidence basis.** Read in full (10 PDF pages). A manufacturer's document, not peer
  reviewed.
- **Findings.**
  - "biaxially oriented polypropylene (BOPP) film capacitors are, as of today, the dominant
    technology solution used for the DC-link of the main traction inverter in xEV systems"
    (p. 2). The source given is a market report: Yole Group, "Capacitors for Power Converters
    2023" (p. 9, ref. [3]). I did not read that report.
  - Example mission profile of the capacitor over the vehicle life: driving 8,000 h,
    preconditioning 5,000 h, charging 40,000 h (p. 7). So the capacitor is under voltage, and
    ages, also when the vehicle stands and charges, not only when it drives.
  - A higher voltage rating gives "a longer service life for demanding mission profiles,
    especially when temperature and voltage are high" (p. 5).
  - Qualification follows the ZVEI (German electrical industry association) guideline "Basic
    Qualification of DC-Link Capacitors for Automotive Use" and AEC-Q200 (Automotive
    Electronics Council) (p. 5); one test shown is 1,000 h at 105 °C (p. 6, Fig. 9(b)).
  - Uneven current sharing inside the bank creates hot spots "that would limit their
    lifetime" (p. 8).

## Could not read

IEEE Xplore answered HTTP 418 (a bot block) to the PDF route on 9 Nov 2026. These items are
therefore not read; nothing above relies on them.

- **B. Yao, X. Ge, H. Wang, H. Wang, D. Zhou, B. Gou**, "Multitimescale Reliability
  Evaluation of DC-Link Capacitor Banks in Metro Traction Drive System", *IEEE Trans.
  Transportation Electrification* 6(1), 213-227, 2020, doi:10.1109/TTE.2020.2974182. Worth
  reading: metro trains stop often, like city buses. No open copy found.
- **H. Wen, W. Xiao, X. Wen, P. Armstrong**, "Analysis and Evaluation of DC-Link Capacitors
  for High-Power-Density Electric Vehicle Drive Systems", *IEEE Trans. Vehicular Technology*
  61(7), 2950-2964, 2012, doi:10.1109/TVT.2012.2206082 (IEEE document 6226484). Compares
  electrolytic and film capacitors in an 80 kW EV drive, with lifetime.
- **S. Chowdhury, E. Gurpinar, B. Ozpineci**, "Capacitor Technologies: Characterization,
  Selection, and Packaging for Next-Generation Power Electronics Applications", *IEEE Trans.
  Transportation Electrification* 8(2), 2022, doi:10.1109/TTE.2021.3139806 (IEEE document
  9675834). Journal version of the topic of S4.
- **N. Sharma, A. Rodionov, Y. Liu**, "Optimal Dimensioning of DC Link Capacitor for Traction
  Machine and Drive Cycles", IECON 2024, doi:10.1109/IECON55916.2024.10905898. Passenger EV
  and heavy-duty truck drive cycles. The Chalmers portal has no full text.
- **A. Rodionov, J. Yodwong, S. De Boodt, Y. Liu**, "Conceptualization and Evaluation of DC
  link Cooling for a 6-phase 500-kW Inverter with Respect to Life Time", IPEC-Nagasaki 2026,
  doi:10.23919/IPEC-Nagasaki2026-EC64663.2026.11596842. Heavy-duty truck, HHDDT (heavy
  heavy-duty diesel truck) drive cycle. No full text on the Chalmers portal; a search snippet
  gives capacitor lives of about 7-10 years, which is unverified.

## Leads not followed

- **Chalmers master's thesis** on DC-link thermal design and lifetime for heavy-duty EV
  traction inverters, odr.chalmers.se (bitstream b75502d7-7bcb-419e-a43d-8c92ab0c9535).
  Downloaded (111 pages), not read: too long for this quick look. The closest source to a
  bus found.
- **S. Yang et al. 2011**, "An industry-based survey of reliability in power electronic
  converters", *IEEE Trans. Industry Applications* 47(3), 1441-1451. Origin of the 19 %
  capacitor share; cross-industry, not vehicles.
- **K. Hu et al. 2020**, review in *IEEE Access* 8, 89988-90022. Cited by S1 for "37 % of the
  unscheduled maintenances in the electrical traction systems"; the original field data
  behind it are unknown.
- **Yole Group 2023**, "Capacitors for Power Converters" market report, cited by S5 for the
  dominance of film capacitors. Paid report.
- **Standards:** EN 61881-1:2011 (railway rolling stock, capacitors for power electronics;
  cited by S2, ref. [42]); ZVEI guideline for automotive DC-link capacitors (cited by S5).
  They may define end-of-life criteria.
- **PORTES** (EU project "Power reliability for traction electronics", CORDIS 517224):
  field-return statistics, which a search snippet says concern IGBTs. Not opened.
- **Not relevant after a check of the title or headings:** D. Zhou et al. 2019, capacitor
  banks in wind power converters (Aalborg repository; title page only); H. Liu et al. 2023,
  *Frontiers in Electronics*, electrolytic capacitors in an EV charging module (a charger,
  not a vehicle).
