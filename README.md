# 🛒 Amazon Sales Insights: A/B Testing, Engagement Analysis & LTV Modeling

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-green)]()

> **End-to-end data science project** analyzing 1,400+ Amazon products to uncover pricing strategy, customer engagement drivers, and lifetime value - using rigorous A/B testing, statistical inference, and CLV modeling.

---

## 📌 Business Impact

| Metric | Finding |
|---|---|
| 💰 Targeting top 25% CLV users vs average | **+237% revenue uplift** ($19.7M → $66.4M per 10k customers) |
| 📣 High-discount vs low-discount engagement | **+6.3% review count uplift** (statistically significant, p < 0.05) |
| ⭐ Price/discount effect on ratings | **No meaningful impact** - product quality drives satisfaction |
| 🎯 LTV concentration | Top 25% of customers → **~81% of total lifetime value** |
| 🏆 Highest LTV categories | Smartphones, Smart TVs, Wearables (high margin + high volume) |

**Bottom line:** Discount strategies move engagement metrics but do not improve ratings. The real lever for revenue growth is identifying and retaining high-CLV customers - the top quartile is worth 3.4× the average customer.

---

## 📂 Project Structure

```
amazon-sales-insights/
├── data/                          # Raw and processed datasets
│   └── amazon.csv                 # Source: Amazon product listings
├── notebooks/
│   └── full_analysis.ipynb        # Original exploratory notebook
├── src/
│   ├── data/
│   │   └── loader.py              # Data loading & cleaning pipeline
│   ├── analysis/
│   │   └── eda.py                 # EDA: distributions, correlations, category insights
│   ├── ab_testing/
│   │   └── ab_test.py             # A/B test engine, bootstrap CI, power analysis, bandit
│   ├── ltv/
│   │   └── ltv_model.py           # CLV modeling, segmentation, scenario analysis
│   └── utils/
│       └── stats_utils.py         # Shared statistical helpers
├── outputs/                       # Generated charts and summary reports
├── tests/
│   └── test_ltv.py                # Unit tests for LTV calculations
├── main.py                        # CLI entry point - runs full pipeline
├── requirements.txt
└── README.md
```

---

## 🔬 Analysis Modules

### 1. Data Cleaning & Feature Engineering (`src/data/loader.py`)
- Parses Indian Rupee price strings, normalizes discount percentages
- Imputes missing rating counts with median
- Engineers derived features: `price_discount_amount`, `discount_ratio`, `log_rating_count`
- Outputs a clean, analysis-ready DataFrame

### 2. Exploratory Data Analysis (`src/analysis/eda.py`)
- Price distribution & quartile segmentation
- Top categories by count, rating, and weighted engagement score
- Pearson & Spearman correlation matrices
- Category-level pricing and discount benchmarking

### 3. A/B Testing Engine (`src/ab_testing/ab_test.py`)
- **Hypothesis:** High discounts (>50%) → higher customer engagement (rating_count)
- Two-sample t-test on log-transformed engagement metric
- **Bootstrap 95% CI** for uplift: –19.5% to +31.3%
- **Statistical power analysis:** Cohen's d, required sample sizes
- **Multi-armed bandit** (ε-greedy) simulation for adaptive allocation
- Chi-square tests across discount, price, and rating level combinations

### 4. LTV Modeling (`src/ltv/ltv_model.py`)
- Infinite-horizon CLV formula: `CLV = (margin × freq) / (1 + discount_rate − retention)`
- Rating-adjusted purchase frequency & retention rate
- Product-level and category-level LTV aggregation
- **Pareto/Lorenz curve** - top 25% users → 81% of LTV
- **Value-based cohort segmentation** (Low / Medium / High / Very High)
- **Scenario analysis:** Conservative / Base Case / Optimistic assumptions

---

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/amazon-sales-insights.git
cd amazon-sales-insights

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your data
cp /path/to/amazon.csv data/amazon.csv

# 4. Run the full pipeline
python main.py

# Optional flags
python main.py --input data/amazon.csv --output outputs/
```

---

## 📊 Key Visualizations

The pipeline automatically generates and saves:

| Chart | Description |
|---|---|
| Price & discount distributions | Histograms + KDE for all numeric features |
| Category benchmarks | Count, avg rating, avg discount by category |
| Correlation heatmaps | Pearson & Spearman matrices |
| A/B test boxplots | Engagement by discount group |
| CLV distribution | Per-customer and total product LTV histograms |
| Lorenz curve | LTV concentration across customer percentiles |
| Scenario comparison | CLV under conservative/base/optimistic assumptions |

---

## 📈 Statistical Methods

| Test | Purpose | Result |
|---|---|---|
| Two-sample t-test | Discount → Rating | p < 0.05; effect negligible (d = −0.20) |
| Two-sample t-test | Price → Rating | p = 0.43; no significant effect |
| Two-sample t-test (log) | Discount → Engagement | p < 0.05; +6.3% uplift |
| Bootstrap CI | Uplift uncertainty | 95% CI: [−19.5%, +31.3%] |
| Chi-square | Discount × Rating level | Significant relationship |
| Chi-square | Price × Rating level | No significant relationship |
| Power analysis | A/B sample adequacy | 3,264 samples/group needed for 15% MDE |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🛠 Tech Stack

| Library | Use |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scipy` | Statistical tests |
| `statsmodels` | Power analysis |
| `matplotlib` / `seaborn` | Visualization |

---

## 📋 Data Dictionary

| Column | Description |
|---|---|
| `product_id` | Unique product identifier |
| `product_name` | Product title |
| `category` | Product category (cleaned, lowercase) |
| `discounted_price` | Sale price (₹) |
| `actual_price` | Original price (₹) |
| `discount_percentage` | Normalized 0–1 discount rate |
| `rating` | Average customer rating (1–5) |
| `rating_count` | Number of customer reviews |

---

## 💡 Business Recommendations

1. **Prioritize high-CLV customer retention** - the top 25% drives 81% of revenue. Invest disproportionately in loyalty programs for this segment.
2. **Discounts are an engagement lever, not a quality signal** - use them to drive initial reviews and visibility, not to compensate for product issues.
3. **Electronics & mobile accessories are your highest-LTV categories** - they combine high margins with large review volumes. Prioritize inventory and promotions here.
4. **Run larger A/B experiments** - current sample sizes (n ≈ 700 per group) are underpowered for 15% MDE. Target 3,264+ per group for reliable results.
5. **Consider adaptive allocation (bandit)** - in live product environments, ε-greedy strategies can reduce regret vs fixed 50/50 splits.

---

## 📄 License

MIT - free to use, adapt, and distribute with attribution.
