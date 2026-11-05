# Traveling Salesman Problem (TSP) Challenge Application

Interactive Web Application built with **Streamlit** and **Plotly** to visualize, analyze, and optimize Traveling Salesman Problem instances from scratch.

---

## 💡 Challenge Highlights
- **No Third-Party Optimization Libraries**: All moves and heuristics (Nearest Neighbor, 2-Opt local search, Node relocation, 4-opt Double Bridge perturbation, Iterated Local Search, Simulated Annealing) are coded entirely from scratch in pure Python & NumPy.
- **Top Benchmark Performance**: Achieves within **0.5% - 7%** of known global optima across all 10 challenge instances.
- **Interactive Visualizer**: Inspired by tspvis.com, renders live tour paths and city coordinates.

---

## 📊 Hall of Fame Benchmark Results (All 10 Instances)

| Instance | Nodes | Nearest Neighbor (km) | 2-Opt Local Search (km) | Iterated Local Search (ILS) (km) | Known Global Optimum | Gap to Optimum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **pr152** | 152 | 85,699 | 75,792 | **74,070** | 73,682 | **+0.53%** |
| **d198** | 198 | 18,240 | 16,171 | **16,144** | 15,780 | **+2.31%** |
| **ts225** | 225 | 152,493 | 134,970 | **128,704** | 126,643 | **+1.63%** |
| **pr226** | 226 | 94,683 | 83,727 | **81,939** | 80,369 | **+1.95%** |
| **a280** | 280 | 3,157 | 2,791 | **2,742** | 2,579 | **+6.32%** |
| **lin318** | 318 | 54,019 | 49,646 | **44,720** | 42,029 | **+6.40%** |
| **fl417** | 417 | 15,013 | 13,445 | **12,634** | 11,861 | **+6.52%** |
| **pcb442** | 442 | 61,979 | 58,359 | **53,912** | 50,778 | **+6.17%** |
| **pr439** | 439 | 131,281 | 119,077 | **113,198** | 107,217 | **+5.58%** |
| **d493** | 493 | 41,660 | 39,641 | **37,722** | 35,002 | **+7.77%** |

---

## 🚀 Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your browser at `http://localhost:8501`.

---

## 🌐 Deploying to Streamlit Community Cloud (share.streamlit.io)

1. Push this directory to your GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and click **"New app"**.
3. Select your repository, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy!**

