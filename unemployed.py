# ================================================================
#  Unemployment Rate Analysis During Covid-19 — India
#
#  Libraries: pandas, numpy, matplotlib, seaborn
# ================================================================
#
#  HOW TO RUN
#  ----------
#  1. Download the dataset from Kaggle (link above).
#  2. Place both CSV files in the SAME folder as this script:
#       • "Unemployment in India.csv"
#       • "Unemployment_Rate_upto_11_2020.csv"
#  3. pip install pandas numpy matplotlib seaborn
#  4. python unemployment_analysis.py
#
#  OUTPUT : 8 PNG files saved in the same folder.
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

# ── Global style ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f9f9f9",
    "axes.edgecolor":    "#cccccc",
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.8,
})

COVID_DATE  = pd.Timestamp("2020-03-25")
PRE_COLOR   = "#2980b9"
POST_COLOR  = "#e74c3c"
LINE_COLOR  = "#c0392b"
GREEN_COLOR = "#27ae60"
NAVY        = "#1a237e"


# ════════════════════════════════════════════════════════════
#  1. LOAD & CLEAN
# ════════════════════════════════════════════════════════════
def load_and_clean():
    df_m = pd.read_csv("Unemployment in India.csv")
    df_s = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

    for df in [df_m, df_s]:
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes("object").columns:
            df[col] = df[col].str.strip()

    df_m["Date"] = pd.to_datetime(df_m["Date"], dayfirst=True)
    df_s["Date"] = pd.to_datetime(df_s["Date"], dayfirst=True)

    rename_map = {
        "Estimated Unemployment Rate (%)":         "Unemp_Rate",
        "Estimated Employed":                      "Employed",
        "Estimated Labour Participation Rate (%)": "LPR",
        "Region":                                  "State",
    }
    df_m.rename(columns=rename_map, inplace=True)
    df_s.rename(columns=rename_map, inplace=True)

    df_m.sort_values("Date", inplace=True)
    df_s.sort_values("Date", inplace=True)
    df_m.reset_index(drop=True, inplace=True)
    df_s.reset_index(drop=True, inplace=True)

    for df in [df_m, df_s]:
        df["Period"] = df["Date"].apply(
            lambda d: "Post-Covid" if d >= COVID_DATE else "Pre-Covid")

    return df_m, df_s


df_monthly, df_state = load_and_clean()

# ── National monthly averages (one clean value per date) ─────
nat = df_monthly.groupby("Date").agg(
    Unemp_Rate=("Unemp_Rate", "mean"),
    LPR=("LPR", "mean")
).reset_index()
nat["Period"] = nat["Date"].apply(
    lambda d: "Post-Covid" if d >= COVID_DATE else "Pre-Covid")

# ── State-level aggregates ───────────────────────────────────
state_avg = (df_state.groupby(["State", "Period"])["Unemp_Rate"]
             .mean().reset_index())
state_post = (state_avg[state_avg["Period"] == "Post-Covid"]
              .sort_values("Unemp_Rate", ascending=False)
              .reset_index(drop=True))

# ── Summary print ────────────────────────────────────────────
pre_vals  = nat[nat["Period"] == "Pre-Covid"]["Unemp_Rate"]
post_vals = nat[nat["Period"] == "Post-Covid"]["Unemp_Rate"]
peak_row  = nat.loc[nat["Unemp_Rate"].idxmax()]

print("=" * 55)
print("  Covid-19 Unemployment Analysis — Summary")
print("=" * 55)
print(f"  Date range  : {nat['Date'].min().date()} → {nat['Date'].max().date()}")
print(f"  Pre-Covid   : mean = {pre_vals.mean():.2f}%  |  max = {pre_vals.max():.2f}%")
print(f"  Post-Covid  : mean = {post_vals.mean():.2f}%  |  max = {post_vals.max():.2f}%")
print(f"  Peak        : {peak_row['Unemp_Rate']:.2f}%  ({peak_row['Date'].date()})")
print("=" * 55)


# ════════════════════════════════════════════════════════════
#  HELPER – axis date formatting
# ════════════════════════════════════════════════════════════
def fmt_xaxis(ax, interval=2):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def add_lockdown(ax, ymax_frac=0.88):
    ax.axvline(COVID_DATE, color=NAVY, linestyle="--", linewidth=1.6,
               label="Covid Lockdown (Mar 2020)", zorder=5)
    ylo, yhi = ax.get_ylim()
    ax.text(COVID_DATE + pd.Timedelta(days=6),
            ylo + (yhi - ylo) * ymax_frac,
            "Covid\nLockdown", fontsize=8, color=NAVY,
            va="top", fontweight="bold")


# ════════════════════════════════════════════════════════════
#  PLOT 1 – National Unemployment Trend
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5))

ax.fill_between(nat["Date"], nat["Unemp_Rate"],
                alpha=0.18, color=LINE_COLOR, zorder=2)
ax.plot(nat["Date"], nat["Unemp_Rate"],
        color=LINE_COLOR, linewidth=2.5,
        marker="o", markersize=5, zorder=3,
        label="Unemployment Rate (%)")

add_lockdown(ax)
fmt_xaxis(ax)

ax.set_title("India – National Unemployment Rate Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Unemployment Rate (%)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("01_national_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_national_trend.png")


# ════════════════════════════════════════════════════════════
#  PLOT 2 – Pre vs Post Covid (Histogram + Boxplot)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = {"Pre-Covid": PRE_COLOR, "Post-Covid": POST_COLOR}

for period, color in colors.items():
    vals = nat[nat["Period"] == period]["Unemp_Rate"]
    axes[0].hist(vals, bins=8, alpha=0.7, label=period,
                 color=color, edgecolor="white")
axes[0].set_title("Distribution – Unemployment Rate")
axes[0].set_xlabel("Unemployment Rate (%)")
axes[0].set_ylabel("Frequency")
axes[0].legend()

bp = sns.boxplot(data=nat, x="Period", y="Unemp_Rate",
                 palette=colors, width=0.45,
                 order=["Pre-Covid", "Post-Covid"], ax=axes[1],
                 linewidth=1.5)
axes[1].set_title("Pre-Covid vs Post-Covid Comparison")
axes[1].set_xlabel("")
axes[1].set_ylabel("Unemployment Rate (%)")

# Annotate medians
for i, period in enumerate(["Pre-Covid", "Post-Covid"]):
    med = nat[nat["Period"] == period]["Unemp_Rate"].median()
    axes[1].text(i, med + 0.3, f"{med:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold",
                 color="white" if period == "Post-Covid" else "black")

fig.suptitle("Impact of Covid-19 on Unemployment Rate",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("02_pre_post_covid.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_pre_post_covid.png")


# ════════════════════════════════════════════════════════════
#  PLOT 3 – State-wise Heatmap
# ════════════════════════════════════════════════════════════
pivot = df_state.pivot_table(
    values="Unemp_Rate", index="State", columns="Date", aggfunc="mean"
)
pivot.columns = [d.strftime("%b %Y") for d in pivot.columns]
pivot = pivot.fillna(pivot.mean())

fig, ax = plt.subplots(figsize=(18, 11))
sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.25, linecolor="white",
            ax=ax, annot=False,
            cbar_kws={"label": "Avg Unemployment Rate (%)", "shrink": 0.6})
ax.set_title("State-wise Monthly Unemployment Rate Heatmap")
ax.set_xlabel("Month")
ax.set_ylabel("State")
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
fig.tight_layout()
fig.savefig("03_statewise_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_statewise_heatmap.png")


# ════════════════════════════════════════════════════════════
#  PLOT 4 – Top 10 Most-Affected States (Post-Covid)
# ════════════════════════════════════════════════════════════
top10 = state_post.head(10).copy()

palette = sns.color_palette("Reds_r", len(top10))
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(top10["State"], top10["Unemp_Rate"],
               color=palette, edgecolor="white", height=0.65)
ax.bar_label(bars, fmt="%.1f%%", padding=5, fontsize=9)
ax.set_title("Top 10 States – Avg Unemployment Rate (Post-Covid)")
ax.set_xlabel("Average Unemployment Rate (%)")
ax.invert_yaxis()
ax.set_xlim(0, top10["Unemp_Rate"].max() * 1.15)
fig.tight_layout()
fig.savefig("04_top10_states.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_top10_states.png")


# ════════════════════════════════════════════════════════════
#  PLOT 5 – Urban vs Rural Unemployment
# ════════════════════════════════════════════════════════════
if "Area" in df_state.columns:
    area_trend = (df_state.groupby(["Date", "Area"])["Unemp_Rate"]
                  .mean().reset_index())

    fig, ax = plt.subplots(figsize=(13, 5))
    area_colors = {"Urban": "#e67e22", "Rural": "#27ae60"}
    for area, color in area_colors.items():
        sub = area_trend[area_trend["Area"] == area]
        ax.plot(sub["Date"], sub["Unemp_Rate"],
                label=area, color=color, linewidth=2.5,
                marker="o", markersize=4)
        ax.fill_between(sub["Date"], sub["Unemp_Rate"],
                        alpha=0.1, color=color)

    add_lockdown(ax)
    fmt_xaxis(ax)
    ax.set_title("Urban vs Rural Unemployment Rate Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("05_urban_vs_rural.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 05_urban_vs_rural.png")


# ════════════════════════════════════════════════════════════
#  PLOT 6 – Correlation Matrix
# ════════════════════════════════════════════════════════════
cols_avail = [c for c in ["Unemp_Rate", "Employed", "LPR"]
              if c in df_state.columns]
corr = df_state[cols_avail].corr()
labels = {"Unemp_Rate": "Unemployment\nRate",
          "Employed":   "Employed",
          "LPR":        "Labour\nParticipation"}
corr.index   = [labels.get(c, c) for c in corr.index]
corr.columns = [labels.get(c, c) for c in corr.columns]

fig, ax = plt.subplots(figsize=(7, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, mask=mask, square=True,
            linewidths=1.2, ax=ax,
            annot_kws={"size": 12, "weight": "bold"},
            cbar_kws={"shrink": 0.75})
ax.set_title("Correlation Matrix – Key Indicators")
fig.tight_layout()
fig.savefig("06_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06_correlation.png")


# ════════════════════════════════════════════════════════════
#  PLOT 7 – Monthly Average Bar Chart (color-coded)
# ════════════════════════════════════════════════════════════
monthly_state_avg = (df_state.groupby("Date")["Unemp_Rate"]
                     .mean().reset_index())
monthly_state_avg["Color"] = monthly_state_avg["Date"].apply(
    lambda d: POST_COLOR if d >= COVID_DATE else PRE_COLOR)

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(monthly_state_avg["Date"],
       monthly_state_avg["Unemp_Rate"],
       color=monthly_state_avg["Color"],
       width=22, edgecolor="white", zorder=3)

add_lockdown(ax)
fmt_xaxis(ax)
ax.set_title("Monthly Average Unemployment Rate Across All States")
ax.set_xlabel("Month")
ax.set_ylabel("Avg Unemployment Rate (%)")
legend_elems = [Patch(facecolor=PRE_COLOR,  label="Pre-Covid"),
                Patch(facecolor=POST_COLOR, label="Post-Covid")]
ax.legend(handles=legend_elems, loc="upper left")
fig.tight_layout()
fig.savefig("07_monthly_avg_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07_monthly_avg_bar.png")


# ════════════════════════════════════════════════════════════
#  PLOT 8 – Unemployment vs Labour Participation (dual axis)
# ════════════════════════════════════════════════════════════
if "LPR" in nat.columns:
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()
    ax2.set_facecolor("none")   # keep bg consistent

    ax1.fill_between(nat["Date"], nat["Unemp_Rate"],
                     alpha=0.12, color=LINE_COLOR)
    l1, = ax1.plot(nat["Date"], nat["Unemp_Rate"],
                   color=LINE_COLOR, linewidth=2.5,
                   marker="o", markersize=4,
                   label="Unemployment Rate (%)")

    l2, = ax2.plot(nat["Date"], nat["LPR"],
                   color=GREEN_COLOR, linewidth=2.5,
                   linestyle="--", marker="s", markersize=4,
                   label="Labour Participation Rate (%)")

    ax1.axvline(COVID_DATE, color=NAVY, linestyle=":",
                linewidth=1.6, label="_nolegend_")
    ylo, yhi = ax1.get_ylim()
    ax1.text(COVID_DATE + pd.Timedelta(days=6),
             ylo + (yhi - ylo) * 0.85,
             "Covid\nLockdown", fontsize=8, color=NAVY,
             va="top", fontweight="bold")

    fmt_xaxis(ax1)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Unemployment Rate (%)", color=LINE_COLOR)
    ax2.set_ylabel("Labour Participation Rate (%)", color=GREEN_COLOR)
    ax1.tick_params(axis="y", labelcolor=LINE_COLOR)
    ax2.tick_params(axis="y", labelcolor=GREEN_COLOR)

    ax1.legend(handles=[l1, l2], loc="upper left")
    ax1.set_title("Unemployment Rate vs Labour Participation Rate")
    fig.tight_layout()
    fig.savefig("08_unemployment_vs_lpr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 08_unemployment_vs_lpr.png")


# ════════════════════════════════════════════════════════════
#  Final summary
# ════════════════════════════════════════════════════════════
print("\nAll 8 charts saved successfully.")
print("Pre-Covid  mean unemployment :", round(pre_vals.mean(), 2), "%")
print("Post-Covid mean unemployment :", round(post_vals.mean(), 2), "%")
print(f"Increase                     : +{post_vals.mean() - pre_vals.mean():.2f} percentage points")
