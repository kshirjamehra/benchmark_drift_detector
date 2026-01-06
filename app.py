import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance

# --- CONFIGURATION ---
st.set_page_config(page_title="Benchmark Drift Detector Pro", page_icon="📉", layout="wide")

# Custom CSS for "Research Lab" aesthetic
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #262730;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    /* Make 3D plot background tighter */
    .js-plotly-plot .plotly .gl-container {
        border: 1px solid #303030;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- CLASS: DRIFT ENGINE ---
class DriftAnalyzer:
    def __init__(self):
        # Simulate 50-dimensional Embeddings
        self.benchmark_data = np.random.normal(loc=0, scale=1, size=(500, 50))
        self.prod_data = None 
        
        # Generate fake "filenames" for context on hover
        self.bench_labels = [f"train_set_img_{i:03d}.jpg" for i in range(500)]
        self.prod_labels = []

    def generate_current_data(self, drift_severity, bias_skew):
        # Shift the center (Concept Drift)
        loc_shift = drift_severity * 4.0 # Increased range for dramatic 3D effect
        
        # Change the spread (Covariate Shift / Bias)
        scale_shift = 1.0 - (bias_skew * 0.6) 
        
        self.prod_data = np.random.normal(loc=loc_shift, scale=scale_shift, size=(500, 50))
        self.prod_labels = [f"live_stream_img_{i:03d}.jpg" for i in range(500)]
        return self.prod_data

    def calculate_drift_metrics(self):
        # Wasserstein Distance (Earth Mover's Distance) averaged across dimensions
        drift_scores = [wasserstein_distance(self.benchmark_data[:, i], self.prod_data[:, i]) for i in range(50)]
        avg_drift = np.mean(drift_scores)
        # Normalize score 0-100
        score = min(avg_drift * 40, 100)
        return score

    def get_pca_visualization(self):
        """
        Reduces dimensions down to 3 for interactive 3D plot.
        """
        pca = PCA(n_components=3)
        
        # Combine to fit PCA
        combined = np.vstack((self.benchmark_data, self.prod_data))
        coords = pca.fit_transform(combined)
        
        # Split back up
        bench_coords = coords[:500]
        prod_coords = coords[500:]
        
        # Create DataFrames with 3D coordinates AND labels
        df_bench = pd.DataFrame(bench_coords, columns=['x', 'y', 'z'])
        df_bench['Type'] = 'Benchmark (Stable)'
        df_bench['Source File'] = self.bench_labels
        
        df_prod = pd.DataFrame(prod_coords, columns=['x', 'y', 'z'])
        df_prod['Type'] = 'Live Stream (Drifting)'
        df_prod['Source File'] = self.prod_labels
        
        return pd.concat([df_bench, df_prod])

# --- UI LOGIC ---
def main():
    st.sidebar.title("🔬 Drift Lab Control")
    
    # Initialize session state for first run
    if 'analyzer' not in st.session_state:
        st.session_state['analyzer'] = DriftAnalyzer()
        # Run an initial mild drift simulation so it's not empty
        st.session_state['analyzer'].generate_current_data(0.3, 0.1)
        st.session_state['has_run'] = True

    analyzer = st.session_state['analyzer']
    
    with st.sidebar.form("simulation_controls"):
        st.subheader("Simulate Embedding Shift")
        drift_val = st.slider("Concept Drift (Mean Shift)", 0.0, 1.0, 0.5, help="Simulates data changing topic/appearance over time.")
        bias_val = st.slider("Bias Introduction (Variance Reduction)", 0.0, 1.0, 0.2, help="Simulates loss of diversity in new data.")
        
        run_sim = st.form_submit_button("Update Analysis")
        if run_sim:
             analyzer.generate_current_data(drift_val, bias_val)

    st.title("📉 Benchmark Bias Drift Detector Pro")
    st.markdown("### 3D Embedding Space Analysis & Auditing")
    
    # 1. Run Calculations
    drift_score = analyzer.calculate_drift_metrics()
    df_viz = analyzer.get_pca_visualization()
    
    # 2. Metrics Row
    c1, c2, c3 = st.columns(3)
    c1.metric("Drift Severity Score", f"{drift_score:.1f}/100", delta_color="inverse")
    
    status = "Healthy"
    status_color = "green"
    if drift_score > 35: 
        status = "Warning: Degradation"
        status_color = "orange"
    if drift_score > 75: 
        status = "CRITICAL: Model Failure"
        status_color = "red"
    
    c2.markdown(f"**System Status:** <span style='color:{status_color}; font-weight:bold'>{status}</span>", unsafe_allow_html=True)
    c3.metric("Statistical Metric", "Wasserstein (EMD)")
    
    st.markdown("---")

    # 3. Visualization Main Section
    col_main, col_info = st.columns([3, 1])
    
    with col_main:
        st.subheader("Interactive 3D Embedding Space (PCA)")
        st.caption("Rotate and zoom to inspect the semantic gap between training data and live data.")
        
        # 3D SCATTER PLOT
        fig = px.scatter_3d(
            df_viz, x='x', y='y', z='z',
            color='Type', 
            hover_name='Source File', # Shows fake filename on hover
            color_discrete_map={'Benchmark (Stable)': '#00CC96', 'Live Stream (Drifting)': '#EF553B'},
            opacity=0.7,
            height=600,
        )
        # Tighter layout for 3D view
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis_title='PCA-1',
                yaxis_title='PCA-2',
                zaxis_title='PCA-3'
            ),
            legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_info:
        st.subheader("💡 Live Analysis")
        
        # Dynamic Alerts based on score
        if drift_score < 35:
            st.success("✅ **System Stable.** Live embeddings are clustered tightly with benchmark data.")
            st.markdown("#### Automated Actions")
            st.markdown("- Continue standard monitoring")
        elif drift_score < 75:
            st.warning("⚠️ **Drift Detected.** The semantic gap is widening. Model confidence is likely declining.")
            st.markdown("#### Recommended Actions")
            st.markdown("- **Trigger alert** to ML Ops team")
            st.markdown("- Sample 20% of 'Live Stream' data for labeling")
        else:
            st.error("🚨 **Critical Failure.** Live data is distributionally distinct from training data.")
            st.markdown("#### CRITICAL Actions")
            st.markdown("- **Rollback model** to previous version immediately")
            st.markdown("- Initiate full retraining pipeline")

if __name__ == "__main__":
    main()