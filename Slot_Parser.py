import streamlit as st
import pandas as pd
import re
import os
from datetime import datetime

st.set_page_config(page_title="OCS vs Fl3xx Slot Compliance", layout="wide")
st.title("🛫 OCS vs Fl3xx Slot Compliance")

st.markdown("""
Upload **Fl3xx CSV(s)** and **OCS CSV(s)** (CYYZ GIR free-text or structured export).
This tool normalizes both formats and compares them against Fl3xx with airport-specific time windows.

**Results**
- ✔ Matched
- ⚠ Missing (no usable slot)
- ⚠ Misaligned (wrong tail or outside time window)
- ⚠ Stale Slots (slot not used by any Fl3xx leg)
""")

# ---------------- Config ----------------
WINDOWS_MIN = {"CYYC": 30, "CYVR": 30, "CYYZ": 60, "CYUL": 15}
MONTHS = {m: i for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"], 1)}

# Pretty month names for displaying Date tuples like (13, 9)
MONTH_ABBR = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

def with_datestr(df: pd.DataFrame, date_col="Date"):
    """Add a human-friendly DateStr column next to the Date tuple."""
    if df is None or df.empty or date_col not in df.columns:
        return df
    out = df.copy()

    def fmt(d):
        try:
            day, month = d
            abbr = MONTH_ABBR.get(int(month), str(month))
            return f"{int(day):02d}-{abbr}"
        except Exception:
            return d

    out["DateStr"] = out[date_col].apply(fmt)

    # place DateStr right after Date
    cols = list(out.columns)
    if "DateStr" in cols and date_col in cols:
        cols.insert(cols.index(date_col) + 1, cols.pop(cols.index("DateStr")))
        out = out[cols]
    return out


SLOT_AIRPORTS = set(WINDOWS_MIN.keys())

# ---------------- Helpers ----------------
def _cyvr_future_exempt(ap: str, sched_dt: pd.Timestamp, threshold_days: int = 4) -> bool:
    """Return True if this should NOT be flagged as Missing:
       CYVR legs that are threshold_days or more days in the future."""
    if ap != "CYVR" or pd.isna(sched_dt):
        return False
    today = pd.Timestamp.utcnow().date()
    days_out = (sched_dt.date() - today).days
    return days_out >= threshold_days

def show_table(df: pd.DataFrame, title: str, key: str):
    st.subheader(title)
    if df is None or df.empty:
        st.write("— no rows —")
        return
    st.dataframe(df, use_container_width=True)
    st.download_button(
        f"Download {title} CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{key}.csv",
        mime="text/csv",
        key=f"dl_{key}"
    )

def _tail_future_exempt(sched_dt: pd.Timestamp, threshold_days: int = 5) -> bool:
    """Hide tail-mismatch items when the leg is >= threshold_days in the future."""
    if pd.isna(sched_dt):
        return False
    today = pd.Timestamp.utcnow().date()
    days_out = (sched_dt.date() - today).days
    return days_out >= threshold_days


# ---------------- Tail filtering ----------------
def load_tails(path="tails.csv"):
    if not os.path.exists(path):
        print(f"No tail list found at {path}, skipping filter")
        return []
    try:
        df = pd.read_csv(path)
        return df["Tail"].astype(str).str.replace("-", "").str.upper().tolist()
    except Exception as e:
        print(f"Error reading tail list: {e}")
        return []

TAILS = load_tails()
if TAILS:
    st.sidebar.success(f"Loaded {len(TAILS)} company tails from tails.csv")
else:
    st.sidebar.warning("No tails.csv found — showing all OCS slots")

# ---------------- Utils ----------------
def _read_csv_reset(file, **kwargs):
    file.seek(0)
    try:
        return pd.read_csv(file, **kwargs)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1", **kwargs)

def _hhmm_str(x):
    """Convert various input formats into zero-padded HHMM string."""
    if pd.isna(x):
        return None
    s = str(x).strip()

    # Handle floats like 15.0, 930.0
    if re.match(r"^\d+\.0$", s):
        s = s[:-2]

    # Handle cases like 9:30 or 09:30
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            hh, mm = parts
            return hh.zfill(2) + mm.zfill(2)

    # Remove non-digits
    s = re.sub(r"\D", "", s)

    # Accept 1–4 digit numbers
    if len(s) == 1:   # "5" → "0005"
        s = s.zfill(4)
    elif len(s) == 2: # "15" → "0015"
        s = s.zfill(4)
    elif len(s) == 3: # "930" → "0930"
        s = s.zfill(4)
    elif len(s) == 4: # "1530" → "1530"
        pass
    else:
        return None

    return s

# ---------------- OCS Parsing ----------------
def parse_gir_file(file):
    df = _read_csv_reset(file)
    col = df.columns[0]
    parsed = []

    for line in df[col].astype(str).tolist():
        # normalize whitespace
        line = line.replace("\u00A0", " ")
        line = re.sub(r"\s+", " ", line.strip())
        parts = line.split(" ")
        if len(parts) < 5:
            continue

        try:
            # 1) Find date token like 13SEP anywhere
            date_idx = next((i for i, p in enumerate(parts) if re.match(r"^\d{2}[A-Z]{3}$", p)), None)
            if date_idx is None:
                continue
            date_str = parts[date_idx]
            day = int(date_str[:2])
            month = MONTHS.get(date_str[2:5].upper())
            if not month:
                continue

            # 2) Find ICAO + time (supports: CYUL0320, 0320CYUL, or split "CYUL","0320" / "0320","CYUL")
            link_icao, slot_time = None, None

            # search tokens after date for ICAO/time
            for i in range(date_idx + 1, len(parts)):
                tok = parts[i]

                m1 = re.match(r"^([A-Z]{4})(\d{3,4})$", tok)   # CYUL0320
                m2 = re.match(r"^(\d{3,4})([A-Z]{4})$", tok)   # 0320CYUL
                if m1:
                    link_icao, slot_time = m1.groups()
                    break
                if m2:
                    slot_time, link_icao = m2.groups()
                    break

                # split across two tokens?
                if i + 1 < len(parts):
                    nxt = parts[i + 1]
                    if re.match(r"^[A-Z]{4}$", tok) and re.match(r"^\d{3,4}$", nxt):
                        link_icao, slot_time = tok, nxt
                        break
                    if re.match(r"^\d{3,4}$", tok) and re.match(r"^[A-Z]{4}$", nxt):
                        slot_time, link_icao = tok, nxt
                        break

            if not slot_time or not link_icao:
                continue

            slot_time = _hhmm_str(slot_time)
            if not slot_time:
                continue

            # 3) Tail token like RE.CFASY
            try:
                tail_token = next(p for p in parts if p.startswith("RE."))
            except StopIteration:
                continue
            tail = tail_token.replace("RE.", "").upper()

            # 4) Slot token like IDA.CYYZAGNN953500/  or IDD.CYYZDGNN027800/
            try:
                slot_token = next(p for p in parts if p.startswith("ID"))
            except StopIteration:
                continue

            mslot = re.match(r"ID[AD]\.(?P<apt>[A-Z]{4})(?P<mov>[AD])(?P<ref>[A-Z0-9]+)/?", slot_token)
            if not mslot:
                continue
            gd = mslot.groupdict()
            
            # Build full slot reference to match structured format (e.g., CYULAGN0396000)
            slot_ref_full = f"{gd['apt']}{gd['mov']}{gd['ref']}"
            
            parsed.append({
                "SlotAirport": gd["apt"],
                "Date": (day, month),
                "Movement": "ARR" if gd["mov"] == "A" else "DEP",
                "SlotTimeHHMM": slot_time,
                "Tail": tail,
                "SlotRef": slot_ref_full  # <-- use full slot id
})
            
        except Exception:
            # keep robust; skip only the bad line
            continue

    print(f"Parsed {len(parsed)} GIR rows out of {len(df)}")
    return pd.DataFrame(parsed, columns=["SlotAirport","Date","Movement","SlotTimeHHMM","Tail","SlotRef"])


def parse_structured_file(file):
    df = _read_csv_reset(file)
    df.columns = [re.sub(r"[^A-Za-z0-9]", "", c).upper() for c in df.columns]

    rows = []
    for _, r in df.iterrows():
        ap = r.get("AP")
        date_val = str(r.get("DATE")).strip()
        if pd.isna(ap) or not date_val:
            continue

        token = date_val.split()[0] if " " in date_val else date_val
        if not re.match(r"\d{2}[A-Z]{3}", token):
            continue
        day = int(token[:2])
        month = MONTHS.get(token[2:5].upper())
        if not month:
            continue

        tail = str(r.get("ACREG", "")).replace("-", "").upper()

        # Arrival
        atime = _hhmm_str(r.get("ATIME"))
        aslot = r.get("ASLOTID")
        if atime and pd.notna(aslot):
            rows.append({
                "SlotAirport": str(ap).upper(),
                "Date": (day, month),
                "Movement": "ARR",
                "SlotTimeHHMM": atime,
                "Tail": tail,
                "SlotRef": str(aslot)
            })

        # Departure
        dtime = _hhmm_str(r.get("DTIME"))
        dslot = r.get("DSLOTID")
        if dtime and pd.notna(dslot):
            rows.append({
                "SlotAirport": str(ap).upper(),
                "Date": (day, month),
                "Movement": "DEP",
                "SlotTimeHHMM": dtime,
                "Tail": tail,
                "SlotRef": str(dslot)
            })

    print(f"Parsed {len(rows)} structured rows out of {len(df)}")
    return pd.DataFrame(rows, columns=["SlotAirport","Date","Movement","SlotTimeHHMM","Tail","SlotRef"])

def parse_ocs_file(file):
    head = _read_csv_reset(file, nrows=5)
    cols_norm = [c.strip().upper() for c in head.columns]
    file.seek(0)
    if "GIR" in cols_norm:
        return parse_gir_file(file)
    hallmark = {"A/P","A/C REG","ATIME","DTIME","ASLOTID","DSLOTID"}
    if hallmark.intersection(set(cols_norm)):
        return parse_structured_file(file)
    return parse_structured_file(file)

# ---------------- Fl3xx Parsing ----------------
def parse_fl3xx_file(file):
    df = _read_csv_reset(file)
    if "Aircraft" in df.columns:
        df["Tail"] = df["Aircraft"].astype(str).str.replace("-", "", regex=False).str.upper()
    else:
        df["Tail"] = ""
    def parse_dt(col):
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        return pd.Series([pd.NaT]*len(df))
    df["OnBlock"] = parse_dt("On-Block (Est)")
    dep_try = ["Off-Block (Est)","Out-Block (Est)","STD (UTC)","Scheduled Departure (UTC)","Departure Time"]
    dep_series = None
    for c in dep_try:
        if c in df.columns:
            dep_series = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            break
    df["OffBlock"] = dep_series
    keep = ["Booking","From (ICAO)","To (ICAO)","Tail","OnBlock","OffBlock","Aircraft Type","Workflow"]
    cols = [c for c in keep if c in df.columns]
    return df[cols].copy()

# ---------------- Comparison ----------------
def compare(fl3xx_df: pd.DataFrame, ocs_df: pd.DataFrame):
    # New: split misaligned into tail vs time
    results = {"Matched": [], "Missing": [], "MisalignedTail": [], "MisalignedTime": []}

    # Build Fl3xx legs at slot airports
    legs = []
    for _, r in fl3xx_df.iterrows():
        tail = str(r.get("Tail", "")).upper()
        to_ap = r.get("To (ICAO)")
        if isinstance(to_ap, str) and to_ap in SLOT_AIRPORTS and pd.notna(r.get("OnBlock")):
            legs.append({"Flight": r.get("Booking"), "Tail": tail, "Airport": to_ap,
                         "Movement": "ARR", "SchedDT": r.get("OnBlock")})
        from_ap = r.get("From (ICAO)")
        if isinstance(from_ap, str) and from_ap in SLOT_AIRPORTS and pd.notna(r.get("OffBlock")):
            legs.append({"Flight": r.get("Booking"), "Tail": tail, "Airport": from_ap,
                         "Movement": "DEP", "SchedDT": r.get("OffBlock")})

    # De-duplicate legs so the same flight isn't processed twice
    if legs:
        legs_df = pd.DataFrame(legs)
        legs_df["SchedDT"] = pd.to_datetime(legs_df["SchedDT"]).dt.floor("min")
        legs_df["LegKey"] = (
            legs_df["Flight"].astype(str) + "|" +
            legs_df["Tail"].astype(str) + "|" +
            legs_df["Airport"].astype(str) + "|" +
            legs_df["Movement"].astype(str) + "|" +
            legs_df["SchedDT"].astype(str)
        )
        legs_df = legs_df.drop_duplicates(subset="LegKey").drop(columns="LegKey")
        legs = legs_df.to_dict("records")

    def minutes_diff(a, b):
        return abs(int((a - b).total_seconds() // 60))

    # --- Leg-side matching (drives Matched / Misaligned* / Missing)
    for leg in legs:
        ap, move, tail, sched_dt = leg["Airport"], leg["Movement"], leg["Tail"], leg["SchedDT"]
        day, month = sched_dt.day, sched_dt.month

        # Candidates: same airport & movement
        cand = ocs_df[
            (ocs_df["SlotAirport"] == ap) &
            (ocs_df["Movement"] == move)
        ].copy()

        if cand.empty:
            if _cyvr_future_exempt(ap, sched_dt):  # keep your CYVR rule
                continue
            results["Missing"].append({**leg, "Reason": "No slot for airport/date/movement"})
            continue

        # Build each slot's absolute datetime using the slot's own (day, month)
        def slot_dt_for_row(row):
            day_r, month_r = row["Date"]
            hhmm = row["SlotTimeHHMM"]
            hh = int(hhmm[:2]); mm = int(hhmm[2:])
            return datetime(sched_dt.year, month_r, day_r, hh, mm)

        cand["_SlotDT"] = cand.apply(slot_dt_for_row, axis=1)

        # Keep slots that are on the same day or ±1 day of the leg (to allow cross-midnight matches)
        cand = cand[cand["_SlotDT"].apply(
            lambda d: abs((d.date() - sched_dt.date()).days) <= 1
        )]

        if cand.empty:
            if _cyvr_future_exempt(ap, sched_dt):
                continue
            results["Missing"].append({**leg, "Reason": "No slot for airport/date/movement (±1 day)"})
            continue

        window = WINDOWS_MIN.get(ap, 30)

        def minutes_diff(a, b):
            return abs(int((a - b).total_seconds() // 60))


        def ocs_dt(row):
            hhmm = row["SlotTimeHHMM"]; hh = int(hhmm[:2]); mm = int(hhmm[2:])
            return datetime(sched_dt.year, month, day, hh, mm)

        # Prefer same-tail
        same_tail = cand[cand["Tail"] == tail]
        if not same_tail.empty:
            deltas = same_tail["_SlotDT"].apply(lambda dt: minutes_diff(sched_dt, dt))
            best_idx = deltas.idxmin()
            best_row = same_tail.loc[best_idx]
            best_delta = int(deltas.loc[best_idx])

            if best_delta <= window:
                results["Matched"].append({
                    **leg,
                    "SlotTime": best_row["SlotTimeHHMM"],
                    "DeltaMin": best_delta,
                    "SlotRef": best_row["SlotRef"]
                })
            else:
                results["MisalignedTime"].append({
                    **leg,
                    "NearestSlotTime": best_row["SlotTimeHHMM"],
                    "DeltaMin": best_delta,
                    "WindowMin": window,
                    "SlotRef": best_row["SlotRef"]
                })
        else:
            deltas_any = cand["_SlotDT"].apply(lambda dt: minutes_diff(sched_dt, dt))
            best_idx = deltas_any.idxmin()
            nearest = cand.loc[best_idx]
            best_delta = int(deltas_any.loc[best_idx])

            if best_delta <= window:
                # Tail mismatch
                if _tail_future_exempt(sched_dt, threshold_days=5):
                    pass  # suppress far-future tail mismatches if you added that helper
                else:
                    results["MisalignedTail"].append({
                        **leg,
                        "NearestSlotTime": nearest["SlotTimeHHMM"],
                        "DeltaMin": best_delta,
                        "WindowMin": window,
                        "SlotTail": nearest["Tail"],
                        "SlotRef": nearest["SlotRef"]
                    })
            else:
                if _cyvr_future_exempt(ap, sched_dt):
                    pass
                else:
                    results["Missing"].append({**leg, "Reason": "No matching tail/time within window"})


    # --- Slot-side evaluation (Stale) — unchanged: a slot is stale only if no leg exists for same airport/movement/date
    def has_any_leg_for_slot(slot_row):
        ap = slot_row["SlotAirport"]
        mv = slot_row["Movement"]
        day, month = slot_row["Date"]
        for lg in legs:
            if (lg["Airport"] == ap and lg["Movement"] == mv and
                lg["SchedDT"].day == day and lg["SchedDT"].month == month):
                return True
        return False

    stale_mask = ~ocs_df.apply(has_any_leg_for_slot, axis=1)
    stale_df = ocs_df[stale_mask].copy()

    return results, stale_df

# ---------------- UI ----------------
fl3xx_files = st.file_uploader("Upload Fl3xx CSV(s)", type="csv", accept_multiple_files=True)
ocs_files = st.file_uploader("Upload OCS CSV(s)", type="csv", accept_multiple_files=True)

if fl3xx_files and ocs_files:
    fl3xx_df = pd.concat([parse_fl3xx_file(f) for f in fl3xx_files], ignore_index=True)
    ocs_list = [parse_ocs_file(f) for f in ocs_files]
    ocs_list = [df for df in ocs_list if not df.empty]
    ocs_df = pd.concat(ocs_list, ignore_index=True) if ocs_list else pd.DataFrame(columns=["SlotAirport","Date","Movement","SlotTimeHHMM","Tail","SlotRef"])

    # Filter OCS slots by tails.csv
    if TAILS:
        before = len(ocs_df)
        ocs_df = ocs_df[ocs_df["Tail"].isin(TAILS)]
        st.info(f"Filtered OCS slots: {before} → {len(ocs_df)} using company tail list")

    st.success(f"Loaded {len(fl3xx_df)} flights and {len(ocs_df)} slots.")

    with st.expander("🔎 Preview parsed OCS (normalized)"):
        st.dataframe(ocs_df.head(20))
    with st.expander("🔎 Preview parsed Fl3xx"):
        st.dataframe(fl3xx_df.head(20))

    results, stale = compare(fl3xx_df, ocs_df)
    
    matched_df       = pd.DataFrame(results["Matched"])
    missing_df       = pd.DataFrame(results["Missing"])
    mis_tail_df      = pd.DataFrame(results["MisalignedTail"])
    mis_time_df      = pd.DataFrame(results["MisalignedTime"])
    stale_df         = with_datestr(stale)  # if you added the pretty date helper
    
    show_table(matched_df,  "✔ Matched",          "matched")
    show_table(missing_df,  "⚠ Missing",          "missing")
    show_table(mis_tail_df, "⚠ Tail mismatch",    "misaligned_tail")
    show_table(mis_time_df, "⚠ Time mismatch",    "misaligned_time")
    show_table(stale_df,    "⚠ Stale Slots",      "stale")


else:
    st.info("Upload both Fl3xx and OCS files to begin.")














