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
    if st.button(" (Clear Results)", use_container_width=True):
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
        
        # --- Duplicate Key Check ---
        duplicates_found = False
        if keys:
            d1_dupes = df1[df1.duplicated(subset=keys)].shape[0]
            d2_dupes = df2[df2.duplicated(subset=keys)].shape[0]
            
            if d1_dupes > 0 or d2_dupes > 0:
                duplicates_found = True
                st.warning(f"⚠️ **Duplicate Keys Detected!** (File 1: {d1_dupes}, File 2: {d2_dupes})")
                clean_dupes = st.checkbox("Automatically clean duplicates (Keep first instance)")
                if clean_dupes:
                    df1 = df1.drop_duplicates(subset=keys)
                    df2 = df2.drop_duplicates(subset=keys)
                    duplicates_found = False 
        
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=duplicates_found and not keys)

    # 6. EXECUTION LOGIC (Optimized for 150k+ rows)
    if run_btn and keys:
        with st.spinner("Analyzing datasets..."):
            # Standardize Join Keys
            for c in keys:
                df1[c] = df1[c].astype(str).str.strip()
                df2[c] = df2[c].astype(str).str.strip()

            # A. ALIGN: Inner Join for direct comparison
            cols_to_use = list(set(keys + comps))
            df_merged = pd.merge(
                df1[cols_to_use], 
                df2[cols_to_use], 
                on=keys, 
                how='inner', 
                suffixes=('_F1', '_F2')
            )
            
            final_report = pd.DataFrame()
            if not df_merged.empty and comps:
                # Vectorized Matrix Comparison
                d1_v = df_merged[[f"{c}_F1" for c in comps]].fillna('N/A').values
                d2_v = df_merged[[f"{c}_F2" for c in comps]].fillna('N/A').values
                
                # Create boolean mask where rows are different
                diff_mask = (d1_v != d2_v).any(axis=1)
                
                if diff_mask.any():
                    # Filter for mismatches
                    final_report = df_merged[diff_mask].copy()
                    # Optional: Add a "Reason" column indicating which columns differ
                    # This is better than adding "Yes/No" for every single column
                    diff_cols = []
                    for i, col in enumerate(comps):
                        col_diff = (d1_v[:, i] != d2_v[:, i])
                        # If you want to keep the "No" logic:
                        final_report[f"{col}_Diff"] = np.where(col_diff, "DIFF", "")

            # B. ORPHANS: Vectorized check using Indicator Merge (Fastest way)
            indicator_df = pd.merge(df1[keys], df2[keys], on=keys, how='outer', indicator=True)
            
            orphans_f1_keys = indicator_df[indicator_df['_merge'] == 'left_only'][keys]
            orphans_f2_keys = indicator_df[indicator_df['_merge'] == 'right_only'][keys]

            orphans_f1 = pd.merge(orphans_f1_keys, df1, on=keys, how='inner')
            orphans_f2 = pd.merge(orphans_f2_keys, df2, on=keys, how='inner')
            
            st.session_state['audit'] = {
                'report': final_report,
                'orphans_f1': orphans_f1,
                'orphans_f2': orphans_f2,
                'metrics': (len(df_merged), len(orphans_f1), len(orphans_f2))
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
            st.subheader("🚩 Mismatched Data (Matched Keys, Different Values)")
            # Apply styling only to a preview of the first 1000 rows to keep UI fast
            st.dataframe(
                res['report'].head(1000).style.applymap(
                    lambda x: 'background-color: #ffcccc; color: #900' if x == "DIFF" else '',
                    subset=[c for c in res['report'].columns if c.endswith('_Diff')]
                ),
                use_container_width=True
            )
        else:
            st.success("✅ No differences found in matched records.")

        st.divider()
        st.subheader("🔍 Orphan Records")
        oc1, oc2 = st.columns(2)
        with oc1:
            st.write(f"**Only in Dataset 1 ({len(res['orphans_f1'])} rows)**")
            st.dataframe(res['orphans_f1'].head(500), use_container_width=True)
        with oc2:
            st.write(f"**Only in Dataset 2 ({len(res['orphans_f2'])} rows)**")
            st.dataframe(res['orphans_f2'].head(500), use_container_width=True)

        # 8. EXPORT
        st.divider()
        st.subheader("📥 Export Audit Report")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            if not res['report'].empty: 
                res['report'].to_excel(writer, index=False, sheet_name='Mismatches')
            res['orphans_f1'].to_excel(writer, index=False, sheet_name='Only_F1')
            res['orphans_f2'].to_excel(writer, index=False, sheet_name='Only_F2')
        
        st.download_button(
            label="💾 Download Full Audit (.xlsx)",
            data=buf.getvalue(),
            file_name="Audit_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
