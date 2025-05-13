import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load & filter (outliers)
df = pd.read_csv("./datasets/training_set_VU_DM_reduced.csv")
# define percentiles
p_lower, p_upper = df['price_usd'].quantile([0.1, 0.9])
d_lower, d_upper = df['orig_destination_distance'].quantile([0.1, 0.9])
# filter
mask = (
    df['price_usd'].between(p_lower, p_upper) &
    df['orig_destination_distance'].between(d_lower, d_upper)
)
filtered = df[mask].copy()
orig_filtered = filtered.copy()   # for fair before/after comparison

# 2. Missing‐flagging
to_impute = ['visitor_hist_starrating',
             'visitor_hist_adr_usd',
             'prop_review_score',
             'price_usd']
for col in to_impute + ['orig_destination_distance']:
    filtered[f"{col}_missing"] = filtered[col].isnull().astype(int)

# 3. Distribution‐preserving imputation
rng = np.random.default_rng(42)
for col in to_impute:
    non_null = filtered[col].dropna().values
    n_missing = filtered[col].isnull().sum()
    sampled = rng.choice(non_null, size=n_missing, replace=True)
    filtered.loc[filtered[col].isnull(), col] = sampled

# distance: fill unknown with -1
filtered['orig_destination_distance'].fillna(-1, inplace=True)

# 4. Recompute derived features
filtered['price_diff_from_history'] = filtered['price_usd'] - filtered['visitor_hist_adr_usd']
filtered['starrating_diff']      = filtered['prop_starrating'] - filtered['visitor_hist_starrating']
filtered['price_rank']           = filtered.groupby('srch_id')['price_usd'].rank()
filtered['location_score1_rank'] = filtered.groupby('srch_id')['prop_location_score1'].rank(pct=True)
filtered['booking_or_click']     = filtered['booking_bool'] + 0.15 * filtered['click_bool']

# 5. Plot overlays for imputed columns
def plot_overlay(orig, proc, title, xlabel):
    plt.figure(figsize=(8,4))
    orig.dropna().plot.kde(label='Filtered Original', linestyle='--')
    proc.dropna().plot.kde(label='Imputed (sampled)', linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()

plot_overlay(orig_filtered['visitor_hist_adr_usd'], 
             filtered['visitor_hist_adr_usd'],
             "Visitor ADR: Original vs Sample‐Imputed",
             "visitor_hist_adr_usd")

plot_overlay(orig_filtered['prop_review_score'],
             filtered['prop_review_score'],
             "Prop Review Score: Original vs Sample‐Imputed",
             "prop_review_score")

plot_overlay(orig_filtered['price_usd'],
             filtered['price_usd'],
             "Price USD: Original vs Sample‐Imputed",
             "price_usd")

# 6. Plot engineered‐feature distributions
plt.figure(figsize=(8,4))
filtered['price_diff_from_history'].hist(bins=100, edgecolor='black')
plt.title("Price Difference from Visitor History")
plt.xlabel("price_diff_from_history")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,4))
filtered['starrating_diff'].hist(bins=10, edgecolor='black')
plt.title("Star Rating Difference from Visitor History")
plt.xlabel("starrating_diff")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,4))
filtered['location_score1_rank'].hist(bins=20, edgecolor='black')
plt.title("Location Score1 Rank (Percentile)")
plt.xlabel("location_score1_rank")
plt.ylabel("Frequency")
plt.show()

# 7. Summary statistics & correlations
print("=== Engineered Feature Summary ===\n")
print(filtered[['price_diff_from_history', 'starrating_diff', 'location_score1_rank']].describe())

print("\n=== Correlation with booking_or_click ===\n")
print(filtered[['price_diff_from_history', 'starrating_diff', 'location_score1_rank', 'booking_or_click']]
      .corr()['booking_or_click'])




