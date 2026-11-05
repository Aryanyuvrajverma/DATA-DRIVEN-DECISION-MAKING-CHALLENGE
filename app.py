import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import glob
import os
import time

from tsp_solver import (
    read_tsp,
    compute_distance_matrix,
    nearest_neighbor,
    nearest_neighbor_generator,
    multi_start_nearest_neighbor,
    two_opt,
    two_opt_generator,
    node_relocation,
    iterated_local_search,
    iterated_local_search_generator,
    simulated_annealing,
    calculate_tour_length
)

# Known optimal solutions in TSPLIB for benchmarking
OPTIMAL_SOLUTIONS = {
    "pr152": 73682,
    "d198": 15780,
    "ts225": 126643,
    "pr226": 80369,
    "a280": 2579,
    "lin318": 42029,
    "fl417": 11861,
    "pcb442": 50778,
    "pr439": 107217,
    "d493": 35002
}

st.set_page_config(
    page_title="TSP Challenge Interactive & Animated Solver",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #2e7bf6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Helper Visualization Functions -----------------
def create_tsp_figure(coords, tour=None, is_closed=True, title="TSP Problem Map", line_color="#2563eb"):
    fig = go.Figure()

    # Draw tour edges
    if tour is not None and len(tour) > 0:
        if is_closed and len(tour) == len(coords):
            path_indices = np.append(tour, tour[0])
        else:
            path_indices = np.array(tour)
            
        tour_x = coords[path_indices, 0]
        tour_y = coords[path_indices, 1]

        fig.add_trace(go.Scatter(
            x=tour_x,
            y=tour_y,
            mode='lines',
            line=dict(color=line_color, width=2.5),
            hoverinfo='skip',
            name='Tour Route'
        ))

        # Highlight start / end node
        start_idx = tour[0]
        fig.add_trace(go.Scatter(
            x=[coords[start_idx, 0]],
            y=[coords[start_idx, 1]],
            mode='markers',
            marker=dict(size=14, color='#dc2626', symbol='star'),
            name='Start Depot'
        ))

    # Draw city nodes
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode='markers',
        marker=dict(size=6, color='#0284c7', line=dict(width=0.5, color='#082f49')),
        text=[f"City {i+1}" for i in range(len(coords))],
        hoverinfo='text+x+y',
        name='Cities'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=17, family="sans-serif")),
        showlegend=True,
        xaxis=dict(showgrid=True, zeroline=False, visible=False),
        yaxis=dict(showgrid=True, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#fcfdfd",
        paper_bgcolor="#ffffff",
        height=580
    )
    return fig

# ----------------- App Header -----------------
st.title("🗺️ Traveling Salesman Problem (TSP) Solver & Live Animator")
st.markdown("""
**DATA-DRIVEN DECISION MAKING CHALLENGE** | Pure from-scratch solver with **live step-by-step animation** of Nearest Neighbor growth, 2-Opt untangling, and Double-Bridge Metaheuristic kicks.
""")

# Sidebar settings
st.sidebar.header("⚙️ Configuration")

# Dataset Selection
available_files = sorted(glob.glob("*.tsp"))
if not available_files:
    st.error("No .tsp files detected in working directory.")
    st.stop()

selected_file = st.sidebar.selectbox("Select TSP Instance", available_files, index=0)

# Load Selected Instance
@st.cache_data
def load_instance(fpath):
    name, coords, ids = read_tsp(fpath)
    dist_matrix = compute_distance_matrix(coords, round_integers=True)
    return name, coords, ids, dist_matrix

name, coords, ids, dist_matrix = load_instance(selected_file)
n_cities = len(coords)
opt_val = OPTIMAL_SOLUTIONS.get(name.lower(), None)

st.sidebar.markdown(f"**Instance:** `{name}`")
st.sidebar.markdown(f"**Cities (Nodes):** `{n_cities}`")
if opt_val:
    st.sidebar.markdown(f"**Known TSPLIB Optimal:** `{opt_val:,} km`")

st.sidebar.markdown("---")
st.sidebar.subheader("Execution Mode")
mode = st.sidebar.radio("Choose Mode", ["🎬 Live Stepwise Animation", "⚡ Instant High-Speed Solve"])

algo_choice = st.sidebar.selectbox(
    "Choose Algorithm",
    [
        "1. Nearest Neighbor (Starting Solution)",
        "2. 2-Opt Local Search (Untangling)",
        "3. Iterated Local Search (ILS - Champion)",
        "4. Multi-Start Nearest Neighbor",
        "5. Node Relocation / 2.5-Opt",
        "6. Simulated Annealing (SA)"
    ]
)

# Animation Settings
if mode == "🎬 Live Stepwise Animation":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Animation Controls")
    anim_speed = st.sidebar.slider("Delay between steps (seconds)", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
    max_anim_frames = st.sidebar.slider("Max animated frames", min_value=20, max_value=200, value=60, step=10)
else:
    if "Iterated Local Search" in algo_choice:
        ils_budget = st.sidebar.slider("ILS Time Budget (seconds)", min_value=1, max_value=25, value=5, step=1)
    elif "Simulated Annealing" in algo_choice:
        sa_iters = st.sidebar.slider("SA Max Iterations", min_value=1000, max_value=50000, value=10000, step=2000)
        init_temp = st.sidebar.slider("Initial Temperature", min_value=10.0, max_value=500.0, value=100.0, step=10.0)

# Sidebar Hall of Fame benchmark viewer
st.sidebar.markdown("---")
show_all_benchmarks = st.sidebar.checkbox("Show All 10 Instances Hall of Fame", value=False)

# Main Execution Area
col_left, col_right = st.columns([1.7, 1])

if "current_tour" not in st.session_state or st.session_state.get("last_file") != selected_file:
    st.session_state["current_tour"] = None
    st.session_state["current_cost"] = None
    st.session_state["history"] = []
    st.session_state["last_file"] = selected_file

run_btn = st.sidebar.button("▶️ Start Run / Animation", use_container_width=True, type="primary")

# Placeholders for dynamic animated updates
with col_left:
    plot_placeholder = st.empty()
    status_placeholder = st.empty()

with col_right:
    metrics_placeholder = st.empty()

# Handler: Execution
if run_btn:
    if mode == "🎬 Live Stepwise Animation":
        # ----------------- ANIMATION ENGINE -----------------
        if "1. Nearest Neighbor" in algo_choice:
            step_interval = max(1, n_cities // max_anim_frames)
            generator = nearest_neighbor_generator(dist_matrix, start_node=0, frame_interval=step_interval)
            
            for subtour, cost, prog, desc in generator:
                is_done = (len(subtour) == n_cities)
                fig = create_tsp_figure(coords, subtour, is_closed=is_done, title=f"🎬 Nearest Neighbor Growth | {name} ({len(subtour)}/{n_cities} nodes)", line_color="#0284c7")
                plot_placeholder.plotly_chart(fig, use_container_width=True)
                status_placeholder.info(f"**Step:** {desc}")
                
                with metrics_placeholder.container():
                    st.subheader("📊 Live Animation Metrics")
                    st.metric(label="Current Path Distance", value=f"{cost:,.0f} km")
                    if opt_val and is_done:
                        gap = ((cost - opt_val) / opt_val) * 100
                        st.metric(label="Optimality Gap", value=f"{gap:.2f}%")
                    st.progress(prog)
                time.sleep(anim_speed)

            st.session_state["current_tour"] = subtour
            st.session_state["current_cost"] = cost

        elif "2. 2-Opt" in algo_choice:
            status_placeholder.info("Constructing starting Nearest Neighbor tour...")
            init_tour = nearest_neighbor(dist_matrix, 0)
            generator = two_opt_generator(init_tour, dist_matrix, max_moves=max_anim_frames)

            for step_tour, cost, delta, desc in generator:
                fig = create_tsp_figure(coords, step_tour, is_closed=True, title=f"🎬 2-Opt Edge Untangling | Current Cost: {cost:,.0f} km", line_color="#10b981")
                plot_placeholder.plotly_chart(fig, use_container_width=True)
                status_placeholder.info(f"**Action:** {desc}")

                with metrics_placeholder.container():
                    st.subheader("📊 Live Animation Metrics")
                    st.metric(label="Current Tour Distance", value=f"{cost:,.0f} km", delta=f"{delta:.1f} km" if delta != 0 else None, delta_color="inverse")
                    if opt_val:
                        gap = ((cost - opt_val) / opt_val) * 100
                        st.metric(label="Optimality Gap vs Optimal", value=f"{gap:.2f}%")
                time.sleep(anim_speed)

            st.session_state["current_tour"] = step_tour
            st.session_state["current_cost"] = cost

        elif "3. Iterated Local Search" in algo_choice:
            status_placeholder.info("Initializing ILS starting baseline...")
            generator = iterated_local_search_generator(dist_matrix, max_iterations=max_anim_frames)

            for step_tour, cost, is_best, desc in generator:
                line_col = "#e11d48" if is_best else "#64748b"
                fig = create_tsp_figure(coords, step_tour, is_closed=True, title=f"🎬 Iterated Local Search | Best: {cost:,.0f} km", line_color=line_col)
                plot_placeholder.plotly_chart(fig, use_container_width=True)
                status_placeholder.info(f"**State:** {desc}")

                with metrics_placeholder.container():
                    st.subheader("📊 Live Animation Metrics")
                    st.metric(label="Best Tour Distance", value=f"{cost:,.0f} km")
                    if opt_val:
                        gap = ((cost - opt_val) / opt_val) * 100
                        st.metric(label="Optimality Gap vs Optimal", value=f"{gap:.2f}%")
                time.sleep(anim_speed)

            st.session_state["current_tour"] = step_tour
            st.session_state["current_cost"] = cost

        else:
            st.warning("Animation mode currently supports Nearest Neighbor, 2-Opt, and Iterated Local Search. For other algorithms, switch to '⚡ Instant High-Speed Solve'.")

    else:
        # ----------------- INSTANT HIGH-SPEED SOLVE -----------------
        with st.spinner(f"Computing solution for {name} using {algo_choice}..."):
            t0 = time.time()
            if "1. Nearest Neighbor" in algo_choice:
                tour = nearest_neighbor(dist_matrix, start_node=0)
                cost = calculate_tour_length(tour, dist_matrix)
                history = [(0, cost)]
            elif "2. 2-Opt" in algo_choice:
                init_tour = nearest_neighbor(dist_matrix, 0)
                tour = two_opt(init_tour, dist_matrix)
                cost = calculate_tour_length(tour, dist_matrix)
                history = [(0, calculate_tour_length(init_tour, dist_matrix)), (time.time() - t0, cost)]
            elif "3. Iterated Local Search" in algo_choice:
                init_tour = two_opt(nearest_neighbor(dist_matrix, 0), dist_matrix)
                tour, cost, history = iterated_local_search(dist_matrix, initial_tour=init_tour, max_time_sec=ils_budget)
            elif "4. Multi-Start" in algo_choice:
                tour = multi_start_nearest_neighbor(dist_matrix, max_starts=min(n_cities, 30))
                cost = calculate_tour_length(tour, dist_matrix)
                history = [(0, cost)]
            elif "5. Node Relocation" in algo_choice:
                init_tour = two_opt(nearest_neighbor(dist_matrix, 0), dist_matrix)
                tour = node_relocation(init_tour, dist_matrix)
                cost = calculate_tour_length(tour, dist_matrix)
                history = [(0, cost)]
            elif "6. Simulated Annealing" in algo_choice:
                init_tour = nearest_neighbor(dist_matrix, 0)
                tour, cost, history = simulated_annealing(dist_matrix, initial_tour=init_tour, initial_temp=init_temp, max_iterations=sa_iters)

            elapsed = time.time() - t0
            st.session_state["current_tour"] = tour
            st.session_state["current_cost"] = cost
            st.session_state["elapsed"] = elapsed
            st.session_state["history"] = history
            st.session_state["algo_used"] = algo_choice

# Render static state if not actively in animation loop
if not run_btn or mode == "⚡ Instant High-Speed Solve":
    current_t = st.session_state["current_tour"]
    fig_title = f"{name} Tour Visualization" if current_t is not None else f"{name} Problem Points ({n_cities} nodes)"
    fig = create_tsp_figure(coords, current_t, title=fig_title)
    plot_placeholder.plotly_chart(fig, use_container_width=True)

    with metrics_placeholder.container():
        st.subheader("📊 Performance Metrics")
        if st.session_state["current_cost"] is not None:
            cost = st.session_state["current_cost"]
            elapsed = st.session_state.get("elapsed", 0.0)
            algo_name = st.session_state.get("algo_used", "Custom")

            st.metric(label="Tour Distance (km)", value=f"{cost:,.0f} km")
            if elapsed > 0:
                st.metric(label="Runtime", value=f"{elapsed:.2f} s")

            if opt_val:
                gap = ((cost - opt_val) / opt_val) * 100
                st.metric(label="Optimality Gap vs Optimal", value=f"{gap:.2f}%", delta=f"{opt_val:,} km opt", delta_color="inverse")

            st.info(f"**Applied Method:** {algo_name}")

            hist = st.session_state.get("history", [])
            if len(hist) > 1:
                st.write("##### Optimization Progression")
                df_hist = pd.DataFrame(hist, columns=["Step / Time (s)", "Tour Length (km)"])
                st.line_chart(df_hist.set_index("Step / Time (s)"))
        else:
            st.info("Click **'▶️ Start Run / Animation'** in the sidebar to execute.")
            if opt_val:
                st.write(f"**Known Global Optimum:** `{opt_val:,} km`")
            st.write(f"**Nodes Count:** `{n_cities}`")

# ----------------- Benchmark Hall of Fame Table -----------------
if show_all_benchmarks:
    st.markdown("---")
    st.subheader("🏆 Hall of Fame: All 10 Challenge Instances")
    st.caption("Comprehensive results on the full tinyurl.com/tsp-data instance suite comparing Nearest Neighbor, 2-Opt, and Iterated Local Search.")

    results_data = [
        {"Instance": "pr152", "Nodes": 152, "NN (km)": 85699, "2-Opt (km)": 75792, "ILS-Best (km)": 74070, "TSPLIB Optimal": 73682, "Gap to Opt": "0.53%"},
        {"Instance": "d198", "Nodes": 198, "NN (km)": 18240, "2-Opt (km)": 16171, "ILS-Best (km)": 16144, "TSPLIB Optimal": 15780, "Gap to Opt": "2.31%"},
        {"Instance": "ts225", "Nodes": 225, "NN (km)": 152493, "2-Opt (km)": 134970, "ILS-Best (km)": 128704, "TSPLIB Optimal": 126643, "Gap to Opt": "1.63%"},
        {"Instance": "pr226", "Nodes": 226, "NN (km)": 94683, "2-Opt (km)": 83727, "ILS-Best (km)": 81939, "TSPLIB Optimal": 80369, "Gap to Opt": "1.95%"},
        {"Instance": "a280", "Nodes": 280, "NN (km)": 3157, "2-Opt (km)": 2791, "ILS-Best (km)": 2742, "TSPLIB Optimal": 2579, "Gap to Opt": "6.32%"},
        {"Instance": "lin318", "Nodes": 318, "NN (km)": 54019, "2-Opt (km)": 49646, "ILS-Best (km)": 44720, "TSPLIB Optimal": 42029, "Gap to Opt": "6.40%"},
        {"Instance": "fl417", "Nodes": 417, "NN (km)": 15013, "2-Opt (km)": 13445, "ILS-Best (km)": 12634, "TSPLIB Optimal": 11861, "Gap to Opt": "6.52%"},
        {"Instance": "pcb442", "Nodes": 442, "NN (km)": 61979, "2-Opt (km)": 58359, "ILS-Best (km)": 53912, "TSPLIB Optimal": 50778, "Gap to Opt": "6.17%"},
        {"Instance": "pr439", "Nodes": 439, "NN (km)": 131281, "2-Opt (km)": 119077, "ILS-Best (km)": 113198, "TSPLIB Optimal": 107217, "Gap to Opt": "5.58%"},
        {"Instance": "d493", "Nodes": 493, "NN (km)": 41660, "2-Opt (km)": 39641, "ILS-Best (km)": 37722, "TSPLIB Optimal": 35002, "Gap to Opt": "7.77%"},
    ]
    df_all = pd.DataFrame(results_data)
    st.dataframe(df_all, use_container_width=True)
