import json
from sklearn.model_selection import KFold

"""
    Used to create 5-fold cross-validation splits from the dataset.
    Saves train and validation splits for each fold in the data/ directory.
"""

# Load all data
with open("data/train.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)

print(f"Total documents: {len(all_data)}")

# Create 5-fold splits
kfold = KFold(n_splits=5, shuffle=True, random_state=42) # For reproducibility

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(all_data)):
    print(f"\nFold {fold_idx}: Train={len(train_idx)}, Val={len(val_idx)}")
    
    # Extract data for this fold
    train_data = [all_data[i] for i in train_idx]
    val_data = [all_data[i] for i in val_idx]
    
    # Save train data
    with open(f"data/fold_{fold_idx}_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # Save val data
    with open(f"data/fold_{fold_idx}_val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

print("\nCreated 5 folds:")
print("   fold_0_train.json, fold_0_val.json")
print("   fold_1_train.json, fold_1_val.json")
print("   fold_2_train.json, fold_2_val.json")
print("   fold_3_train.json, fold_3_val.json")
print("   fold_4_train.json, fold_4_val.json")