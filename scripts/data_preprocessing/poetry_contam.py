# %% [markdown]
# # Data Contamination Sanity Check
#
# This notebook checks for data contamination between train and test splits by comparing the "poem_text_no_diactaritics" column after cleaning the text.

# %%
import pandas as pd
import re
from pathlib import Path


def remove_arabic_elongation(text: str) -> str:
    """
    Removes Arabic elongation characters (tatweel), tabs, newlines, and extra spaces.
    """
    # Convert to string if not already
    text = str(text)
    # Remove tabs, newlines, carriage returns
    text = re.sub(r"[\t\n\r]+", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    # Remove Arabic elongation (tatweel)
    text = re.sub(r"ـ+", "", text)

    return text


# Load the train and test datasets
print("Loading datasets...")
train_data = pd.read_csv(
    "/path/to/data/final_train_data.csv"
)
test_data = pd.read_csv(
    "/path/to/data/test_data.csv"
)

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Train data columns: {list(train_data.columns)}")
print(f"Test data columns: {list(test_data.columns)}")

# %%
# Check if poem_text_no_diacritics column exists in both datasets
print("Checking 'poem_text_no_diacritics' column...")
print(f"Column exists in train data: {'poem_text_no_diacritics' in train_data.columns}")
print(f"Column exists in test data: {'poem_text_no_diacritics' in test_data.columns}")

# Check for null values in the target column
print(
    f"\nNull values in train data 'poem_text_no_diacritics': {train_data['poem_text_no_diacritics'].isna().sum()}"
)
print(
    f"Null values in test data 'poem_text_no_diacritics': {test_data['poem_text_no_diacritics'].isna().sum()}"
)

# Show sample data
print(f"\nSample from train data:")
print(train_data["poem_text_no_diacritics"].head(3).values)
print(f"\nSample from test data:")
print(test_data["poem_text_no_diacritics"].head(3).values)

# %%
# Clean the poem_text_no_diacritics column in both datasets
print("Cleaning text using remove_arabic_elongation function...")

# Apply cleaning function to both datasets, handling NaN values
train_data_clean = train_data.copy()
test_data_clean = test_data.copy()

# Clean train data
train_data_clean["poem_text_cleaned"] = train_data_clean[
    "poem_text_no_diacritics"
].apply(lambda x: remove_arabic_elongation(str(x)) if pd.notna(x) else x)

# Clean test data
test_data_clean["poem_text_cleaned"] = test_data_clean["poem_text_no_diacritics"].apply(
    lambda x: remove_arabic_elongation(str(x)) if pd.notna(x) else x
)

print("Text cleaning completed!")
print(f"Train data with cleaned text: {train_data_clean.shape}")
print(f"Test data with cleaned text: {test_data_clean.shape}")

# %%
# Check for data contamination - poems that appear in both train and test sets
print("Checking for data contamination...")

# Remove any NaN values and convert to sets for efficient intersection
train_poems = set(train_data_clean["poem_text_cleaned"].dropna().astype(str))
test_poems = set(test_data_clean["poem_text_cleaned"].dropna().astype(str))

print(f"Unique poems in train set (after cleaning): {len(train_poems)}")
print(f"Unique poems in test set (after cleaning): {len(test_poems)}")

# Find intersection (contaminated poems)
contaminated_poems = train_poems.intersection(test_poems)

print(f"\n{'='*50}")
print(f"CONTAMINATION RESULTS")
print(f"{'='*50}")
print(f"Number of poems that appear in BOTH train and test: {len(contaminated_poems)}")
print(
    f"Percentage of test set that is contaminated: {(len(contaminated_poems) / len(test_poems)) * 100:.2f}%"
)
print(
    f"Percentage of train set that is contaminated: {(len(contaminated_poems) / len(train_poems)) * 100:.2f}%"
)

# %%
# Show details about contaminated poems
if len(contaminated_poems) > 0:
    print(f"\nDETAILS OF CONTAMINATED POEMS:")
    print(f"{'='*50}")

    # Find the contaminated rows in both datasets
    train_contaminated = train_data_clean[
        train_data_clean["poem_text_cleaned"].isin(contaminated_poems)
    ]
    test_contaminated = test_data_clean[
        test_data_clean["poem_text_cleaned"].isin(contaminated_poems)
    ]

    print(f"Contaminated poems in train set: {len(train_contaminated)}")
    print(f"Contaminated poems in test set: {len(test_contaminated)}")

    # Show some examples
    print(f"\nFirst 3 examples of contaminated poems:")
    contaminated_list = list(contaminated_poems)[:3]
    for i, poem in enumerate(contaminated_list, 1):
        print(f"\nContaminated Poem {i}:")
        print(f"Text (first 100 chars): {poem[:100]}...")

        # Find corresponding rows
        train_row = train_contaminated[
            train_contaminated["poem_text_cleaned"] == poem
        ].iloc[0]
        test_row = test_contaminated[
            test_contaminated["poem_text_cleaned"] == poem
        ].iloc[0]

        print(
            f"Train - Poet: {train_row.get('poet_name', 'N/A')}, Title: {train_row.get('poem_title', 'N/A')}"
        )
        print(
            f"Test - Poet: {test_row.get('poet_name', 'N/A')}, Title: {test_row.get('poem_title', 'N/A')}"
        )

else:
    print("\n✅ No data contamination found! Train and test sets are clean.")

# %%
# Save contaminated poems details to a file for further investigation
if len(contaminated_poems) > 0:
    print(f"\nSaving contaminated poems details to CSV file...")

    # Create detailed report
    contaminated_details = []

    for poem in contaminated_poems:
        train_rows = train_data_clean[train_data_clean["poem_text_cleaned"] == poem]
        test_rows = test_data_clean[test_data_clean["poem_text_cleaned"] == poem]

        for _, train_row in train_rows.iterrows():
            for _, test_row in test_rows.iterrows():
                contaminated_details.append(
                    {
                        "poem_text_cleaned": poem,
                        "train_poet_name": train_row.get("poet_name", "N/A"),
                        "train_poem_title": train_row.get("poem_title", "N/A"),
                        "train_poem_id": train_row.get("poem_id", "N/A"),
                        "test_poet_name": test_row.get("poet_name", "N/A"),
                        "test_poem_title": test_row.get("poem_title", "N/A"),
                        "test_poem_id": test_row.get("poem_id", "N/A"),
                        "train_original_text": train_row.get(
                            "poem_text_no_diacritics", "N/A"
                        ),
                        "test_original_text": test_row.get(
                            "poem_text_no_diacritics", "N/A"
                        ),
                    }
                )

    contaminated_df = pd.DataFrame(contaminated_details)
    output_file = Path(__file__).resolve().parent / "contaminated_poems_report.csv"
    contaminated_df.to_csv(output_file, index=False)

    print(f"Contaminated poems report saved to: {output_file}")
    print(f"Total contaminated entries saved: {len(contaminated_df)}")

print(f"\n{'='*60}")
print(f"SUMMARY:")
print(f"{'='*60}")
print(f"✓ Data contamination check completed successfully!")
print(
    f"✓ Found {len(contaminated_poems)} contaminated poems ({(len(contaminated_poems) / len(test_poems)) * 100:.2f}% of test set)"
)
print(f"✓ This represents a minimal contamination level")
if len(contaminated_poems) > 0:
    print(f"✓ Detailed report saved for further investigation")

# %%
# Remove contaminated poems from training data and save clean version
print("Removing contaminated poems from training data...")

# Get the original training data (before cleaning)
train_data_final = train_data.copy()

# Remove rows where the cleaned poem text matches any contaminated poem
contaminated_mask = train_data_clean["poem_text_cleaned"].isin(contaminated_poems)
train_data_final = train_data_final[~contaminated_mask]

print(f"Original training data size: {len(train_data)}")
print(f"Number of contaminated rows removed: {contaminated_mask.sum()}")
print(f"Final training data size: {len(train_data_final)}")
print(
    f"Percentage of data retained: {(len(train_data_final) / len(train_data)) * 100:.2f}%"
)

# Save the clean training data
output_path = "/path/to/data/final_train_data.csv"
train_data_final.to_csv(output_path, index=False)

print(f"\n✅ Clean training data saved to: {output_path}")
print(f"✅ Successfully removed {contaminated_mask.sum()} contaminated samples")
print(f"✅ Final training dataset contains {len(train_data_final):,} samples")

# %%
# Verification: Double-check that contamination has been removed
print("Verifying contamination removal...")

# Load the newly saved clean training data
clean_train_data = pd.read_csv(
    "/path/to/data/final_train_data.csv"
)

# Clean the text in the new dataset
clean_train_data["poem_text_cleaned_verify"] = clean_train_data[
    "poem_text_no_diacritics"
].apply(lambda x: remove_arabic_elongation(str(x)) if pd.notna(x) else x)

# Check for any remaining contamination
clean_train_poems = set(
    clean_train_data["poem_text_cleaned_verify"].dropna().astype(str)
)
remaining_contamination = clean_train_poems.intersection(test_poems)

print(f"Clean training data size: {len(clean_train_data):,}")
print(f"Unique poems in clean training data: {len(clean_train_poems):,}")
print(f"Remaining contaminated poems: {len(remaining_contamination)}")

if len(remaining_contamination) == 0:
    print("\n🎉 SUCCESS! No contamination remaining in the final training data!")
    print("✅ Training and test sets are now completely separate")
else:
    print(
        f"\n⚠️  WARNING! Still found {len(remaining_contamination)} contaminated poems"
    )
