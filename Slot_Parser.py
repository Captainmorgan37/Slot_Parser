import streamlit as st
import pandas as pd
import re
from datetime import datetime

st.set_page_config(page_title="OCS vs Fl3xx Slot Compliance", layout="wide")

st.title("🛫 OCS vs Fl3xx Slot Compliance")

st.markdown("""
Upload **Fl3xx CSV(s)** and **OCS CSV(s)** (you may upload GIR format or structured export).
This tool will normalize both formats and compare them against Fl3xx.
Results:
- ✔ Matched
- ⚠ Flights Missing Slots
- ⚠ Misaligned (time/tail)
- ⚠ Stale Slots
""")

WINDOWS_MIN = {"CYYC": 30, "CYVR": 30, "CYYZ": 60, "CYUL": 15}
MONTHS = {m: i for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"], 1)}

# ---------------- OCS Parsing ----------------
ocs_line_re = re.compile(
    r"""^(?P<ignored>\S+)\s+
        (?P<date>\d{2}[A-Z]{3})\s+
        (?P<maxpax>\d{3})(?P<acft>[A-Z0-9]{3,4})\s+
        (?P<link_icao>[A-Z]{4})(?P<slot_time>\d{4}).*?
        RE\.(?P<tail>[A-Z0-9]+).*?
        \.(?P<slot_airport>[A-Z]{4})(?P<movement>[AD])(?P<slot_ref>[A-Z0-9]+)/""",
    re.VERBOSE
)

def parse_gir_file(file):
    df = pd.read_csv(file)
    col = df.columns[0]
    parsed = []
    for line in df[col].astype(str).tolist():
        m = ocs_line_re.search(line)
        if not m: continue
        gd = m.groupdict()
        day = int(gd["date"][:2]); month = MONTHS[gd["date"][2:5]]
        parsed.append({
            "SlotAirport": gd["slot_airport"],
            "Date": (day, month),
            "Movement": "ARR" if gd["movement"] == "A" else "DEP",
            "SlotTimeHHMM": gd["slot_time"].zfill(4),
            "Tail": gd["tail"].upper(),
            "SlotRef": gd["slot_ref"]
        })
    return pd.DataFrame(parsed)

def parse_structured_file(file):
    df = pd.read_csv(file)
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.get("A/P")) or pd.isna(r.get("Date")): continue
        try:
            dt = pd.to_datetime(r["Date"], errors="coerce", dayfirst=True)
            if pd.isna(dt): continue
        except: continue
        # Arrival slot
        if pd.notna(r.get("ATime")) and pd.notna(r.get("ASlotId")):
            rows.append({
                "SlotAirport": r["A/P"],
                "Date": (dt.day, dt.month),
                "Movement": "ARR",
                "SlotTimeHHMM": str(r["ATime"]).zfill(4),
                "Tail": str(r["A/C Reg"]).replace("-","").upper(),
                "SlotRef": str(r["ASlotId"])
            })
        # Departure slot
        if pd.notna(r.get("DTime")) and pd.notna(r.get("DSlotId")):
            rows.append({
                "SlotAirport": r["A/P"],
                "Date": (dt.day, dt.month),
                "Movement": "DEP",
                "SlotTimeHHMM": str(r["DTime"]).zfill(4),
                "Tail": str(r["A/C Reg"]).replace("-","").upper(),
                "SlotRef": str(r["DSlotId"])
            })
    return pd.DataFrame(rows)

def parse_ocs_file(file):
    df = pd.read_csv(file, nrows=5)  # peek at headers
    if "GIR" in df.columns:   # GIR format
        return parse_gir_file(file)
    elif "A/P" in df.columns: # structured format
        return parse_structured_file(file)
    else:
        return pd.DataFrame([])

# ---------------- Fl3xx Parsing ----------------
def parse_fl3xx_file(file):
    df = pd.read_csv(file)
    df["Tail"] = df["Aircraft"].astype(str).str.replace("-","").str.upper()
    df["OnBlock"] = pd.to_datetime(df["On-Block (Est)"], errors="coerce", dayfirst=True)
    dep_cols_try = ["Off-Block (Est)","Out-Block (Est)","STD (UTC)","Scheduled Departure (UTC)"]
    dep_series = None
    for c in dep_cols_try:
        if c in df.columns: dep_series = pd.to_datetime(df[c], errors="coerce", dayfirst=True); break
    df["OffBlock"] = dep_series
    return df

# ---------------- Comparison Logic ----------------
def compare(fl3xx_df, ocs_df):
    results = {"Matched": [], "Missing": [], "Misaligned": []}
    used_slots = set()

    def minutes_diff(a, b):
        return abs(int((a - b).total_seconds() // 60))

    # Build legs from Fl3xx
    legs = []
    for _, r in fl3xx_df.iterrows():
        tail = str(r.get("Tail","")).upper()
        # Arrival
        to_ap = r.get("To (ICAO)")
        if isinstance(to_ap, str) and to_ap in WINDOWS_MIN and pd.notna(r["OnBlock"]):
            legs.append({"Flight": r["Booking"], "Tail": tail, "Airport": to_ap,
                         "Movement": "ARR", "SchedDT": r["OnBlock"]})
        # Departure
        from_ap = r.get("From (ICAO)")
        if isinstance(from_ap, str) and from_ap in WINDOWS_MIN and pd.notna(r["OffBlock"]):
            legs.append({"Flight": r["Booking"], "Tail": tail, "Airport": from_ap,
                         "Movement": "DEP", "SchedDT": r["OffBlock"]})

    for leg in legs:
        ap, move, tail, sched_dt = leg["Airport"], leg["Movement"], leg["Tail"], leg["SchedDT"]
        day, month = sched_dt.day, sched_dt.month
        cand = ocs_df[(ocs_df["SlotAirport"] == ap) &
                      (ocs_df["Movement"] == move) &
                      (ocs_df["Date"].apply(lambda d: d==(day,month)))]
        if cand.empty:
            results["Missing"].append({**leg, "Reason":"No slot for airport/date/movement"})
            continue

        same_tail = cand[cand["Tail"] == tail]
        window = WINDOWS_MIN[ap]

        def ocs_dt(row):
            return datetime(sched_dt.year, month, day,
                            int(row["SlotTimeHHMM"][:2]), int(row["SlotTimeHHMM"][2:]))

        match_found = False
        if not same_tail.empty:
            for idx, s in same_tail.iterrows():
                delta = minutes_diff(sched_dt, ocs_dt(s))
                if delta <= window:
                    results["Matched"].append({**leg, "SlotTime":s["SlotTimeHHMM"],
                                               "DeltaMin":delta,"SlotRef":s["SlotRef"]})
                    used_slots.add(idx); match_found=True; break
            if not match_found:
                nearest = same_tail.iloc[0]
                results["Misaligned"].append({**leg, "NearestSlotTime":nearest["SlotTimeHHMM"],
                                              "Issue":"Outside time window"})
        else:
            nearest = cand.iloc[0]
            delta = minutes_diff(sched_dt, ocs_dt(nearest))
            if delta <= window:
                results["Misaligned"].append({**leg, "NearestSlotTime":nearest["SlotTimeHHMM"],
                                              "Issue":f"Wrong tail ({nearest['Tail']})"})
            else:
                results["Missing"].append({**leg, "Reason":"No usable slot match"})

    stale = ocs_df.drop(index=list(used_slots))
    return results, stale

# ---------------- UI ----------------
fl3xx_files = st.file_uploader("Upload Fl3xx CSV(s)", type="csv", accept_multiple_files=True)
ocs_files = st.file_uploader("Upload OCS CSV(s)", type="csv", accept_multiple_files=True)

if fl3xx_files and ocs_files:
    fl3xx_df = pd.concat([parse_fl3xx_file(f) for f in fl3xx_files], ignore_index=True)
    ocs_df = pd.concat([parse_ocs_file(f) for f in ocs_files], ignore_index=True)
    st.success(f"Loaded {len(fl3xx_df)} flights and {len(ocs_df)} slots.")

    results, stale = compare(fl3xx_df, ocs_df)

    st.subheader("✔ Matched")
    st.dataframe(pd.DataFrame(results["Matched"]))

    st.subheader("⚠ Missing")
    st.dataframe(pd.DataFrame(results["Missing"]))

    st.subheader("⚠ Misaligned")
    st.dataframe(pd.DataFrame(results["Misaligned"]))

    st.subheader("⚠ Stale Slots")
    st.dataframe(stale)

else:
    st.info("Upload both Fl3xx and OCS files to begin.")
