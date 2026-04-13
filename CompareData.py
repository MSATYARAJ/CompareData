import streamlit as st
import pandas as pd
import numpy as np
import io

# 1. Page Configuration
st.set_page_config(page_title="High-Speed Audit", layout="wide")
pd.set_option("styler.render.max_elements", 20000000)
st.title("🚀 Professional Data Comparison Tool")

# 2. Sidebar Controls
with st.sidebar:
    st.header("Controls")
    if st.button("🧹 Clear Comparison Results", use_container_width=True):
        if 'audit' in st.session_state:
            del st.session_state['audit']
        st.success("Results cleared. Files kept.")
    
    st.divider()
    if st.button("🔄 Full Reset (Clear Files)", type="secondary"):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# 3. Data Loading Utilities
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# 4. File Upload Section
c1, c2 = st.columns(2)
with c1:
    file1 = st.file_uploader("Upload Dataset 1", type=["csv", "xlsx"], key="u1")
with c2:
    file2 = st.file_uploader("Upload Second Dataset", type=["csv", "xlsx"], key="u2")

if file1 and file2:
    df1 = load_data(file1)
    df2 = load_data(file2)
    common_cols = [c for c in df1.columns if c in df2.columns]

    # 5. Configuration Panel
    with st.expander("⚙️ Configuration Settings", expanded=True):
        col_k, col_c = st.columns(2)
        with col_k:
            keys = st.multiselect("Select Unique Key(s):", options=common_cols)
        with col_c:
            remaining = [c for c in common_cols if c not in keys]
            select_all = st.checkbox("Select All Columns")
            comps = st.multiselect("Select Columns to Compare:", options=remaining, default=remaining if select_all else [])
        
        # --- NEW: Duplicate Key Check ---
        duplicates_found = False
        if keys:
            d1_dupes = df1[df1.duplicated(subset=keys)].shape[0]
            d2_dupes = df2[df2.duplicated(subset=keys)].shape[0]
            
            if d1_dupes > 0 or d2_dupes > 0:
                duplicates_found = True
                st.warning(f"⚠️ **Duplicate Keys Detected!** (File 1: {d1_dupes}, File 2: {d2_dupes})")
                st.info("Duplicates cause shape mismatch errors or bloated results. Please use unique keys or clean below.")
                clean_dupes = st.checkbox("Automatically clean duplicates (Keep first instance)")
                if clean_dupes:
                    df1 = df1.drop_duplicates(subset=keys)
                    df2 = df2.drop_duplicates(subset=keys)
                    duplicates_found = False # Reset flag if user opted to clean
        
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=duplicates_found and not keys)

    # 6. EXECUTION LOGIC (Using Inner Merge to fix shape issues)
    if run_btn and keys:
        with st.spinner("Executing optimized comparison..."):
            # Standardize Join Keys (strip spaces and convert to string)
            for c in keys:
                df1[c] = df1[c].astype(str).str.strip()
                df2[c] = df2[c].astype(str).str.strip()

            # Align using an Inner Merge (This ensures both arrays have same shape)
            cols_to_use = list(set(keys + comps))
            df_merged = pd.merge(
                df1[cols_to_use], 
                df2[cols_to_use], 
                on=keys, 
                how='inner', 
                suffixes=('_F1', '_F2')
            )
            
            final_report = pd.DataFrame()
            if comps:
                # Prepare matrices for vectorized comparison
                d1_v = df_merged[[f"{c}_F1" for c in comps]].fillna('N/A').values
                d2_v = df_merged[[f"{c}_F2" for c in comps]].fillna('N/A').values
                
                # Check for any mismatch across columns
                mask = (d1_v != d2_v).any(axis=1)
                
                if mask.any():
                    # Build report for mismatched rows
                    report_df = df_merged.loc[mask, keys].copy()
                    for col in comps:
                        v1 = df_merged.loc[mask, f"{col}_F1"]
                        v2 = df_merged.loc[mask, f"{col}_F2"]
                        report_df[f"{col}_F1"] = v1
                        report_df[f"{col}_F2"] = v2
                        report_df[f"{col}_Equal"] = np.where(v1.fillna('N/A') == v2.fillna('N/A'), "Yes", "No")
                    final_report = report_df

            # Calculate Orphans using sets for speed
            s1 = set(df1[keys].itertuples(index=False, name=None))
            s2 = set(df2[keys].itertuples(index=False, name=None))
            
            st.session_state['audit'] = {
                'report': final_report,
                'orphans_f1': df1[~df1[keys].apply(tuple, axis=1).isin(s2)],
                'orphans_f2': df2[~df2[keys].apply(tuple, axis=1).isin(s1)],
                'metrics': (len(df_merged), len(s1 - s2), len(s2 - s1))
            }

    # 7. RESULTS DISPLAY & EXPORT
    if 'audit' in st.session_state:
        res = st.session_state['audit']
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matched Records", res['metrics'][0])
        m2.metric("Only in File 1", res['metrics'][1])
        m3.metric("Only in File 2", res['metrics'][2])
        m4.metric("Mismatch Rows", len(res['report']))

        if not res['report'].empty:
            st.dataframe(
                res['report'].head(3000).style.map(lambda x: 'background-color: #ffcccc; color: #900' if x == "No" else ''),
                use_container_width=True
            )
        else:
            st.success("✅ No differences found in matched records.")

        st.divider()
        st.subheader("🚩 Orphan Records")
        oc1, oc2 = st.columns(2)
        with oc1:
            st.write("**Only in Dataset 1**")
            st.dataframe(res['orphans_f1'].head(100), use_container_width=True)
        with oc2:
            st.write("**Only in Dataset 2**")
            st.dataframe(res['orphans_f2'].head(100), use_container_width=True)

        st.divider()
        st.subheader("📥 Export Audit Report")
        fmt = st.radio("Format:", ["Excel", "CSV"], horizontal=True)

        if fmt == "Excel":
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                if not res['report'].empty: res['report'].to_excel(writer, index=False, sheet_name='Mismatches')
                res['orphans_f1'].to_excel(writer, index=False, sheet_name='Only_F1')
                res['orphans_f2'].to_excel(writer, index=False, sheet_name='Only_F2')
            st.download_button("💾 Download All", buf.getvalue(), "Audit_Report.xlsx")
        else:
            if not res['report'].empty: 
                st.download_button("📄 Mismatches (CSV)", to_csv_bytes(res['report']), "mismatches.csv")
