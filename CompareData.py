import streamlit as st
import pandas as pd
import numpy as np
import io

# 1. Page Configuration
st.set_page_config(page_title="High-Speed Audit", layout="wide")
pd.set_option("styler.render.max_elements", 20000000)
st.title("🚀 Professional Data Comparison Tool")

# 2. Sidebar - Partial Reset Logic
with st.sidebar:
    st.header("Controls")
    # This button only clears the comparison results, NOT the uploaded files
    if st.button("🧹 Clear Comparison Results", use_container_width=True):
        if 'audit' in st.session_state:
            del st.session_state['audit']
        st.success("Comparison results cleared. Files kept.")
    
    st.divider()
    # Optional: Full reset if they want to change files entirely
    if st.button("🔄 Full Reset (Clear Files)", type="secondary"):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# 3. Optimized Data Loading
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
        
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    # 6. EXECUTION LOGIC (Stored in Session State)
    if run_btn and keys:
        with st.spinner("Executing optimized comparison..."):
            # Standardize Join Keys
            for c in keys:
                df1[c] = df1[c].astype(str).str.strip()
                df2[c] = df2[c].astype(str).str.strip()

            # Fast Orphan Detection using Sets
            s1 = set(df1.set_index(keys).index)
            s2 = set(df2.set_index(keys).index)
            matched_idx = list(s1.intersection(s2))
            
            # Align DataFrames
            df1_m = df1.set_index(keys).loc[matched_idx].sort_index().reset_index()
            df2_m = df2.set_index(keys).loc[matched_idx].sort_index().reset_index()
            
            # Vectorized Matrix Comparison
            final_report = pd.DataFrame()
            if comps:
                d1_v = df1_m[comps].fillna('N/A').values
                d2_v = df2_m[comps].fillna('N/A').values
                mask = (d1_v != d2_v).any(axis=1)
                
                if mask.any():
                    parts = [df1_m.loc[mask, keys].reset_index(drop=True)]
                    for col in comps:
                        v1, v2 = df1_m.loc[mask, col], df2_m.loc[mask, col]
                        status = np.where(v1.fillna('N/A') == v2.fillna('N/A'), "Yes", "No")
                        parts.append(pd.DataFrame({
                            f"{col}_F1": v1.reset_index(drop=True),
                            f"{col}_F2": v2.reset_index(drop=True),
                            f"{col}_Equal": status
                        }))
                    final_report = pd.concat(parts, axis=1)

            # Save results to session state
            st.session_state['audit'] = {
                'report': final_report,
                'orphans_f1': df1.set_index(keys).loc[list(s1 - s2)].reset_index(),
                'orphans_f2': df2.set_index(keys).loc[list(s2 - s1)].reset_index(),
                'metrics': (len(matched_idx), len(s1 - s2), len(s2 - s1))
            }

    # 7. RESULTS DISPLAY & EXPORT
    if 'audit' in st.session_state:
        res = st.session_state['audit']
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matched Records", res['metrics'][0])
        m2.metric("Only in File 1", res['metrics'][1])
        m3.metric("Only in File 2", res['metrics'][2])
        m4.metric("Values Mismatched Records", len(res['report']))

        if not res['report'].empty:
            if len(res['report']) > 3000:
                st.warning("⚠️ High cell count! Displaying first 3,000 rows. Download for full data.")
                display_df = res['report'].head(3000)
            else:
                display_df = res['report']

            st.dataframe(
                display_df.style.map(lambda x: 'background-color: #ffcccc; color: #900' if x == "No" else ''),
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
        fmt = st.radio("Select Format:", ["Excel", "CSV"], horizontal=True)

        if fmt == "Excel":
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                if not res['report'].empty: res['report'].to_excel(writer, index=False, sheet_name='Mismatches')
                res['orphans_f1'].to_excel(writer, index=False, sheet_name='Only_F1')
                res['orphans_f2'].to_excel(writer, index=False, sheet_name='Only_F2')
            st.download_button("💾 Download All as Excel", buf.getvalue(), "Audit_Report.xlsx")
        else:
            ec1, ec2, ec3 = st.columns(3)
            if not res['report'].empty: 
                ec1.download_button("📄 Mismatches (CSV)", to_csv_bytes(res['report']), "mismatches.csv")
            ec2.download_button("📄 File 1 Orphans (CSV)", to_csv_bytes(res['orphans_f1']), "f1_orphans.csv")
            ec3.download_button("📄 File 2 Orphans (CSV)", to_csv_bytes(res['orphans_f2']), "f2_orphans.csv")
