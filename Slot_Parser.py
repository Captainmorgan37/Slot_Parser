import streamlit as st
import pandas as pd
import re
from io import StringIO

st.set_page_config(page_title="Slot Compliance Checker", layout="wide")

# ---------- Helpers ----------
def parse_ocs_line(line):
    """Parse a single OCS GIR line into structured fields"""
    if not isinstance(line, str) or len(line.strip()) == 0:
        return None

    slot_id_match = re.match(r"([A-Z0-9]+)", line)
    slot_id = slot_id_match.group(1) if slot_id_match else None

    date_match = re.search(r"(\d{2}[A-Z]{3})", line)
    date = date_match.group(1) if date_match else None

    acft_match = re.search(r"\s([A-Z0-9]{3,4})\s", line)
    acft_type = acft_match.group(1) if acft_match else None

    ap_time_match = re.search(r"([A-Z]{4})(\d{3,4})", line)
    airport, time_str = (ap_time_match.group(1), ap_time_match.group(2)) if ap_time_match else (None, None)

    tail_match = re.search(r"RE\.([A-Z0-9]+)", line)
    tail = tail_match.group(1) if tail_match else None

    return {
        "SlotID": slot_id,
        "Date": date,
        "AircraftType": acft_type,
        "Airport": airport,
        "SlotTime": time_str,
        "Tail": tail,
        "Raw": line
    }

def parse_ocs_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    parsed = [parse_ocs_line(line) for line in df.iloc[:,0]]
    parsed = [row for row in parsed if row is not None]
    return pd.DataFrame(parsed)

def parse_fl3xx_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # Normalize time format
    df["On-Block (Est)"] = pd.to_datetime(df["On-Block (Est)"], errors="coerce", dayfirst=True)
    return df

# ---------- UI ----------
st.title("🛫 Slot Compliance Checker")

st.markdown("""
Upload your **Fl3xx flight exports** and **OCS slot CSVs** below.
The tool will cross-check them and highlight:
- ✔ Slots matched  
- ⚠ Flights missing slots  
- ⚠ Misaligned slots (time/tail)  
- ⚠ Stale slots (slot but no flight)
""")

fl3xx_files = st.file_uploader("Upload Fl3xx CSV(s)", type="csv", accept_multiple_files=True)
ocs_files = st.file_uploader("Upload OCS CSV(s)", type="csv", accept_multiple_files=True)

if fl3xx_files and ocs_files:
    # Combine Fl3xx
    fl3xx_df = pd.concat([parse_fl3xx_file(f) for f in fl3xx_files], ignore_index=True)
    # Combine OCS
    ocs_df = pd.concat([parse_ocs_file(f) for f in ocs_files], ignore_index=True)

    # ---- Comparison Logic (very simplified v1) ----
    results_matched, results_missing, results_misaligned, results_stale = [], [], [], []

    for idx, row in fl3xx_df.iterrows():
        tail = str(row["Aircraft"]).replace("-", "").upper()
        airport = row["From (ICAO)"] if row["From (ICAO)"] in ["CYYC", "CYVR", "CYYZ", "CYUL"] else row["To (ICAO)"]
        sched_time = row["On-Block (Est)"]

        if pd.isna(sched_time) or airport not in ["CYYC","CYVR","CYYZ","CYUL"]:
            continue

        # Find matching OCS slots
        slot_matches = ocs_df[(ocs_df["Airport"] == airport) & (ocs_df["Tail"] == tail)]

        if slot_matches.empty:
            results_missing.append({
                "Flight": row["Booking"],
                "Tail": tail,
                "Airport": airport,
                "SchedTime": sched_time
            })
        else:
            # Check for time misalignments (simple check only for demo)
            matched = False
            for _, slot in slot_matches.iterrows():
                slot_time = slot["SlotTime"]
                if slot_time and sched_time.strftime("%H%M")[:3] in slot_time:  # crude match
                    matched = True
                    results_matched.append({
                        "Flight": row["Booking"],
                        "Tail": tail,
                        "Airport": airport,
                        "SchedTime": sched_time,
                        "SlotTime": slot_time
                    })
                    break
            if not matched:
                results_misaligned.append({
                    "Flight": row["Booking"],
                    "Tail": tail,
                    "Airport": airport,
                    "SchedTime": sched_time,
                    "SlotIDs": ",".join(slot_matches["SlotID"].astype(str))
                })

    # Stale slots (not matched to any Fl3xx flight)
    used_slot_ids = [r["SlotIDs"] if "SlotIDs" in r else r.get("SlotID") for r in results_matched]
    stale = ocs_df[~ocs_df["SlotID"].isin(used_slot_ids)]
    for _, s in stale.iterrows():
        results_stale.append(s.to_dict())

    # ---- Display Results ----
    st.subheader("✔ Matched Slots")
    st.dataframe(pd.DataFrame(results_matched))

    st.subheader("⚠ Missing Slots")
    st.dataframe(pd.DataFrame(results_missing))

    st.subheader("⚠ Misaligned Slots")
    st.dataframe(pd.DataFrame(results_misaligned))

    st.subheader("⚠ Stale Slots")
    st.dataframe(pd.DataFrame(results_stale))

    # Download option
    combined_report = {
        "Matched": pd.DataFrame(results_matched),
        "Missing": pd.DataFrame(results_missing),
        "Misaligned": pd.DataFrame(results_misaligned),
        "Stale": pd.DataFrame(results_stale),
    }
    for key, df in combined_report.items():
        if not df.empty:
            st.download_button(
                f"Download {key} as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"{key}_report.csv",
                mime="text/csv"
            )
