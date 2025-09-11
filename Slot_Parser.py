
import streamlit as st
import pandas as pd
import re
from datetime import datetime, timedelta

st.set_page_config(page_title="OCS vs Fl3xx Slot Compliance", layout="wide")

st.title("🛫 OCS vs Fl3xx Slot Compliance")

st.markdown("""
Upload **Fl3xx flight CSV(s)** and **OCS GIR CSV(s)** (you can upload more than one OCS file).
This app parses the OCS lines, compares to Fl3xx, and classifies:
- ✔ Matched
- ⚠ Flights Missing Slots
- ⚠ Misaligned (time or tail)
- ⚠ Stale Slots (slot exists but no flight)
""")

WINDOWS_MIN = {"CYYC": 30, "CYVR": 30, "CYYZ": 60, "CYUL": 15}

# ---------------- Parsers ----------------
MONTHS = {
    "JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
    "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12
}

ocs_line_re = re.compile(
    r"""^(?P<ignored>\S+)\s+
        (?P<date>\d{2}[A-Z]{3})\s+
        (?P<maxpax>\d{3})
        (?P<acft>[A-Z0-9]{3,4})\s+
        (?P<link_icao>[A-Z]{4})
        (?P<slot_time>\d{4})
        .*?RE\.(?P<tail>[A-Z0-9]+).*?
        \.(?P<slot_airport>[A-Z]{4})(?P<movement>[AD])(?P<slot_ref>[A-Z0-9]+)/
    """,
    re.VERBOSE
)

def parse_ocs_row(line: str):
    if not isinstance(line, str):
        return None
    m = ocs_line_re.search(line)
    if not m:
        return None
    gd = m.groupdict()
    day = int(gd["date"][:2])
    mon_abbr = gd["date"][2:5]
    month = MONTHS.get(mon_abbr, None)
    slot_time_hm = gd["slot_time"].zfill(4)
    return {
        "OCSDate_Day": day,
        "OCSDate_Month": month,
        "MaxPax": int(gd["maxpax"]),
        "AircraftType": gd["acft"],
        "OtherICAO": gd["link_icao"],
        "SlotTimeHHMM": slot_time_hm,
        "Tail": gd["tail"].upper(),
        "SlotAirport": gd["slot_airport"],
        "Movement": "ARR" if gd["movement"] == "A" else "DEP",
        "SlotRef": gd["slot_ref"],
        "Raw": line
    }

def parse_ocs_file(file):
    df = pd.read_csv(file)
    # first column contains GIR lines
    col = df.columns[0]
    parsed = []
    for line in df[col].tolist():
        p = parse_ocs_row(line)
        if p:
            parsed.append(p)
    return pd.DataFrame(parsed)

def parse_fl3xx_file(file):
    df = pd.read_csv(file)
    # Normalize columns
    # tail cleanup
    if "Aircraft" in df.columns:
        df["Tail"] = df["Aircraft"].astype(str).str.replace("-", "", regex=False).str.upper()
    else:
        df["Tail"] = ""
    # parse times
    def parse_dt(col):
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        return pd.Series([pd.NaT]*len(df))
    df["OnBlock"] = parse_dt("On-Block (Est)")
    # Attempt to find a departure time column if present
    dep_cols_try = ["Off-Block (Est)","Out-Block (Est)","Scheduled Departure (UTC)","STD (UTC)","Departure Time"]
    dep_series = None
    for c in dep_cols_try:
        if c in df.columns:
            dep_series = parse_dt(c)
            break
    if dep_series is None:
        df["OffBlock"] = pd.NaT
    else:
        df["OffBlock"] = dep_series
    # Keep key columns
    keep = ["Booking","From (ICAO)","To (ICAO)","Tail","OnBlock","OffBlock","Aircraft Type","Workflow"]
    return df[[c for c in keep if c in df.columns]]

# ---------------- UI ----------------
fl3xx_files = st.file_uploader("Upload Fl3xx CSV(s)", type="csv", accept_multiple_files=True)
ocs_files = st.file_uploader("Upload OCS CSV(s) (GIR lines)", type="csv", accept_multiple_files=True)
assume_utc = st.checkbox("Assume Fl3xx times are UTC", value=True)

if fl3xx_files and ocs_files:
    with st.spinner("Parsing files..."):
        fl3xx_df = pd.concat([parse_fl3xx_file(f) for f in fl3xx_files], ignore_index=True)
        ocs_df = pd.concat([parse_ocs_file(f) for f in ocs_files], ignore_index=True)
        # focus only on slot airports
        slot_airports = set(WINDOWS_MIN.keys())
        ocs_df = ocs_df[ocs_df["SlotAirport"].isin(slot_airports)].copy()

    st.success(f"Loaded {len(fl3xx_df)} Fl3xx rows and {len(ocs_df)} OCS slots.")

    # Build flight legs to check (ARR + DEP if data exists)
    legs = []
    for _, r in fl3xx_df.iterrows():
        tail = str(r.get("Tail","")).upper()
        # ARR leg
        to_ap = r.get("To (ICAO)")
        if isinstance(to_ap, str) and to_ap in slot_airports and pd.notna(r.get("OnBlock")):
            legs.append({
                "Flight": r.get("Booking"),
                "Tail": tail,
                "Airport": to_ap,
                "Movement": "ARR",
                "SchedDT": r.get("OnBlock")
            })
        # DEP leg (only if we have a departure time)
        from_ap = r.get("From (ICAO)")
        if isinstance(from_ap, str) and from_ap in slot_airports and pd.notna(r.get("OffBlock")):
            legs.append({
                "Flight": r.get("Booking"),
                "Tail": tail,
                "Airport": from_ap,
                "Movement": "DEP",
                "SchedDT": r.get("OffBlock")
            })

    legs_df = pd.DataFrame(legs)

    # Comparison
    matched_rows = []
    missing_rows = []
    misaligned_rows = []

    used_slot_idxs = set()

    def minutes_diff(a, b):
        return abs(int((a - b).total_seconds() // 60))

    for i, leg in legs_df.iterrows():
        ap = leg["Airport"]
        move = leg["Movement"]
        tail = leg["Tail"]
        sched_dt = leg["SchedDT"]

        if pd.isna(sched_dt):
            # Can't compare without a time
            missing_rows.append({
                "Flight": leg["Flight"], "Tail": tail, "Airport": ap, "Movement": move,
                "Reason": "No schedule time in Fl3xx"
            })
            continue

        # Filter candidate OCS slots by airport, movement, and date
        day = sched_dt.day
        month = sched_dt.month
        cand = ocs_df[(ocs_df["SlotAirport"] == ap) & (ocs_df["Movement"] == move) &
                      (ocs_df["OCSDate_Day"] == day) & (ocs_df["OCSDate_Month"] == month)]

        if cand.empty:
            missing_rows.append({
                "Flight": leg["Flight"], "Tail": tail, "Airport": ap, "Movement": move,
                "SchedDT": sched_dt, "Reason": "No OCS slots for airport/date/movement"
            })
            continue

        # Split by tail match
        cand_same_tail = cand[cand["Tail"] == tail]
        airport_window = WINDOWS_MIN.get(ap, 30)

        # Build OCS datetime using sched_dt.year (assumes same year)
        def ocs_dt(row):
            hh = int(row["SlotTimeHHMM"][:2])
            mm = int(row["SlotTimeHHMM"][2:])
            return datetime(sched_dt.year, row["OCSDate_Month"], row["OCSDate_Day"], hh, mm)

        # Evaluate time proximity
        best_idx = None
        best_delta = None
        best_slot = None
        for idx, s in cand_same_tail.iterrows():
            dt = ocs_dt(s)
            delta = minutes_diff(sched_dt, dt)
            if (best_delta is None) or (delta < best_delta):
                best_delta = delta
                best_idx = idx
                best_slot = s

        if best_slot is not None and best_delta is not None and best_delta <= airport_window:
            matched_rows.append({
                "Flight": leg["Flight"], "Tail": tail, "Airport": ap, "Movement": move,
                "SchedDT": sched_dt, "SlotTime": best_slot["SlotTimeHHMM"],
                "DeltaMin": best_delta, "SlotRef": best_slot["SlotRef"]
            })
            used_slot_idxs.add(best_idx)
        else:
            # If same-tail exists but outside window, it's a time misalignment
            if not cand_same_tail.empty:
                # choose nearest for context
                nearest = None; nearest_delta = None; nearest_idx = None
                for idx, s in cand_same_tail.iterrows():
                    dt = ocs_dt(s)
                    delta = minutes_diff(sched_dt, dt)
                    if (nearest_delta is None) or (delta < nearest_delta):
                        nearest = s; nearest_delta = delta; nearest_idx = idx
                misaligned_rows.append({
                    "Flight": leg["Flight"], "Tail": tail, "Airport": ap, "Movement": move,
                    "SchedDT": sched_dt, "NearestSlotTime": nearest["SlotTimeHHMM"],
                    "DeltaMin": nearest_delta, "Issue": f"Outside {airport_window} min window", "SlotRef": nearest["SlotRef"]
                })
            else:
                # No same-tail; check if there's a slot within window but wrong tail
                nearest = None; nearest_delta = None; nearest_idx = None
                for idx, s in cand.iterrows():
                    dt = ocs_dt(s)
                    delta = minutes_diff(sched_dt, dt)
                    if (nearest_delta is None) or (delta < nearest_delta):
                        nearest = s; nearest_delta = delta; nearest_idx = idx
                if nearest is not None and nearest_delta is not None and nearest_delta <= airport_window:
                    misaligned_rows.append({
                        "Flight": leg["Flight"], "Tail": tail, "Airport": ap, "Movement": move,
                        "SchedDT": sched_dt, "NearestSlotTime": nearest["SlotTimeHHMM"],
                        "DeltaMin": nearest_delta, "Issue": f"Wrong tail (slot for {nearest['Tail']})", "SlotRef": nearest["SlotRef"]
                    })
                    # (do not mark used; it's not a valid match)
                else:
                    # No usable slot at all
                    missing_rows.append({
                        "Flight": leg["Flight"], "Tail": tail, "Airport": ap, "Movement": move,
                        "SchedDT": sched_dt, "Reason": "No matching tail/time within window"
                    })

    # Stale slots = OCS not used in matches on that date/airport/movement
    stale_mask = pd.Series([True]*len(ocs_df))
    if used_slot_idxs:
        stale_mask.loc[list(used_slot_idxs)] = False
    stale_df = ocs_df[stale_mask].copy()

    # Output tables
    st.subheader("✔ Matched")
    matched_df = pd.DataFrame(matched_rows)
    st.dataframe(matched_df)

    st.subheader("⚠ Flights Missing Slots")
    missing_df = pd.DataFrame(missing_rows)
    st.dataframe(missing_df)

    st.subheader("⚠ Misaligned (time or tail)")
    misaligned_df = pd.DataFrame(misaligned_rows)
    st.dataframe(misaligned_df)

    st.subheader("⚠ Stale Slots")
    st.dataframe(stale_df)

    # Downloads
    def dl(df, name):
        if not df.empty:
            st.download_button(
                f"Download {name} CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"{name.replace(' ','_').lower()}.csv",
                mime="text/csv"
            )

    col1, col2, col3, col4 = st.columns(4)
    with col1: dl(matched_df, "Matched")
    with col2: dl(missing_df, "Missing")
    with col3: dl(misaligned_df, "Misaligned")
    with col4: dl(stale_df, "Stale")

else:
    st.info("Upload both Fl3xx and OCS CSV files to begin.")
