import pandas as pd
import matplotlib.pyplot as plt

# Load datasets (make sure the files are in the same directory)
train = pd.read_csv("./datasets/training_set_VU_DM.csv")
test = pd.read_csv("./datasets/test_set_VU_DM.csv")

# 1. Basic Info
print("Training Set Shape:", train.shape)
print("Test Set Shape:", test.shape)

print("\nTraining Set Columns:")
print(train.columns)

# 2. Missing Values Summary
missing_values = train.isnull().mean().sort_values(ascending=False) * 100
print("\nMissing Values (% of rows):")
print(missing_values[missing_values > 0])

# 3. Descriptive Statistics
print("\nDescriptive Statistics:")
print(train.describe())

# 4. Plot: Property Star Rating Distribution
plt.figure(figsize=(8, 5))
train['prop_starrating'].hist(bins=6, edgecolor='black')
plt.title("Distribution of Property Star Ratings")
plt.xlabel("Star Rating")
plt.ylabel("Number of Listings")
plt.grid(False)
plt.show()

# 5. Plot: Search Booking Window
plt.figure(figsize=(8, 5))
train['srch_booking_window'].hist(bins=50, edgecolor='black')
plt.title("Distribution of Booking Window (Days in Advance)")
plt.xlabel("Days")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()

# 6. Plot: Origin-Destination Distance
plt.figure(figsize=(8, 5))
train['orig_destination_distance'].dropna().hist(bins=100, edgecolor='black')
plt.title("Distribution of Origin-Destination Distance")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()

# 7. Plot: Booking vs. Non-Booking
booking_counts = train['booking_bool'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(['Not Booked', 'Booked'], booking_counts.values, color=['grey', 'green'])
plt.title("Booking vs Non-Booking")
plt.ylabel("Number of Records")
plt.show()

# 8. Plot: Click vs. No Click
click_counts = train['click_bool'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(['No Click', 'Clicked'], click_counts.values, color=['blue', 'orange'])
plt.title("Click vs No Click")
plt.ylabel("Number of Records")
plt.show()
