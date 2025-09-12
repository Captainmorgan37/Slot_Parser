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
    # Split misaligned into tail vs time
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

    # Prevent reusing a slot once it is matched to a leg
    allocated_slot_refs = set()
    # Optional: avoid suggesting the same slot for multiple legs
    suggested_slot_refs = set()

    for leg in legs:
        ap, move, tail, sched_dt = leg["Airport"], leg["Movement"], leg["Tail"], leg["SchedDT"]

        # Start with slots for same airport & movement that are NOT already allocated
        cand = ocs_df[
            (ocs_df["SlotAirport"] == ap) &
            (ocs_df["Movement"] == move) &
            (~ocs_df["SlotRef"].astype(str).isin(allocated_slot_refs))
        ].copy()

        if cand.empty:
            if _cyvr_future_exempt(ap, sched_dt):  # your helper
                continue
            results["Missing"].append({**leg, "Reason": "No slot for airport/movement"})
            continue

        # Build each slot's absolute datetime using the slot's own (day, month)
        def slot_dt_for_row(row):
            d, m = row["Date"]
            hhmm = row["SlotTimeHHMM"]; hh = int(hhmm[:2]); mm = int(hhmm[2:])
            return datetime(sched_dt.year, m, d, hh, mm)

        cand["_SlotDT"] = cand.apply(slot_dt_for_row, axis=1)

        # Keep slots on the same day or ±1 day of the leg (cross-midnight tolerance)
        cand = cand[cand["_SlotDT"].apply(lambda d: abs((d.date() - sched_dt.date()).days) <= 1)]
        if cand.empty:
            if _cyvr_future_exempt(ap, sched_dt):
                continue
            results["Missing"].append({**leg, "Reason": "No slot (±1 day)"})
            continue

        window = WINDOWS_MIN.get(ap, 30)

        # Compute best same-tail and best any-tail deltas
        same_tail = cand[cand["Tail"] == tail]
        same_tail_best, same_tail_row = None, None
        if not same_tail.empty:
            deltas_same = same_tail["_SlotDT"].apply(lambda dt: minutes_diff(sched_dt, dt))
            same_idx = deltas_same.idxmin()
            same_tail_best = int(deltas_same.loc[same_idx])
            same_tail_row  = same_tail.loc[same_idx]

        deltas_any = cand["_SlotDT"].apply(lambda dt: minutes_diff(sched_dt, dt))
        any_idx = deltas_any.idxmin()
        any_best = int(deltas_any.loc[any_idx])
        any_row  = cand.loc[any_idx]

        # Decision order:
        # 1) same-tail within window → Matched (allocate; cannot be reused)
        if same_tail_best is not None and same_tail_best <= window:
            results["Matched"].append({
                **leg,
                "SlotTime": same_tail_row["SlotTimeHHMM"],
                "DeltaMin": same_tail_best,
                "SlotRef":  same_tail_row["SlotRef"]
            })
            allocated_slot_refs.add(str(same_tail_row["SlotRef"]))
            continue

        # 2) any-tail within window → Tail mismatch (do NOT allocate; wrong tail booked)
        if any_best <= window:
            if not _tail_future_exempt(sched_dt, threshold_days=5):  # your helper
                if str(any_row["SlotRef"]) not in suggested_slot_refs:
                    results["MisalignedTail"].append({
                        **leg,
                        "NearestSlotTime": any_row["SlotTimeHHMM"],
                        "DeltaMin": any_best,
                        "WindowMin": window,
                        "SlotTail": any_row["Tail"],
                        "SlotRef":  any_row["SlotRef"]
                    })
                    suggested_slot_refs.add(str(any_row["SlotRef"]))
            continue

        # 3) same-tail exists but out of window → Time mismatch (do NOT allocate)
        if same_tail_best is not None:
            results["MisalignedTime"].append({
                **leg,
                "NearestSlotTime": same_tail_row["SlotTimeHHMM"],
                "DeltaMin": same_tail_best,
                "WindowMin": window,
                "SlotRef":  same_tail_row["SlotRef"]
            })
            continue

        # 4) otherwise → Missing (respect CYVR exemption)
        if not _cyvr_future_exempt(ap, sched_dt):
            results["Missing"].append({**leg, "Reason": "No matching tail/time within window"})

    # --- Slot-side evaluation (Stale)
    # A slot is NOT stale if:
    #   (a) there is ANY leg with same airport/movement/TAIL within ±1 day of the slot's date, or
    #   (b) we've already used it (Matched) or suggested it (Tail mismatch), or
    #   (c) it's a far-future wrong-tail case (we suppress tail mismatches 5+ days out).
    def has_leg_for_slot(slot_row):
        ap   = slot_row["SlotAirport"]
        mv   = slot_row["Movement"]
        tail = slot_row["Tail"]
        day, month = slot_row["Date"]

        for lg in legs:
            if lg["Airport"] != ap or lg["Movement"] != mv or lg["Tail"] != tail:
                continue
            # build slot date using leg's year (slots have no year)
            slot_date = datetime(lg["SchedDT"].year, month, day).date()
            if abs((slot_date - lg["SchedDT"].date()).days) <= 1:
                return True
        return False

    def _far_future_wrong_tail(slot_row):
        ap   = slot_row["SlotAirport"]
        mv   = slot_row["Movement"]
        tail = slot_row["Tail"]
        day, month = slot_row["Date"]
        for lg in legs:
            if lg["Airport"] != ap or lg["Movement"] != mv:
                continue
            # slot date in the leg's year
            slot_date = datetime(lg["SchedDT"].year, month, day).date()
            # same day ±1 indicates this slot relates to that leg's operation
            if abs((slot_date - lg["SchedDT"].date()).days) <= 1:
                # wrong tail & we intentionally suppress tail mismatches far in the future
                if lg["Tail"] != tail and _tail_future_exempt(lg["SchedDT"], threshold_days=5):
                    return True
        return False

    # Slots already used (Matched) or suggested (Tail mismatch) should not be stale
    used_or_suggested = set()
    used_or_suggested.update(allocated_slot_refs)
    used_or_suggested.update(suggested_slot_refs)

    stale_df = ocs_df[
        (~ocs_df["SlotRef"].astype(str).isin(used_or_suggested)) &
        (~ocs_df.apply(has_leg_for_slot, axis=1)) &
        (~ocs_df.apply(_far_future_wrong_tail, axis=1))
    ].copy()




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

    # NEW: ensure no duplicate SlotRef remain
    ocs_df = ocs_df.drop_duplicates(subset=["SlotRef"]).reset_index(drop=True)

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



















