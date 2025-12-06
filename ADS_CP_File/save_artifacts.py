import joblib
import os
import sys

print("--- 1. Verifying Environment ---")

# Check if the model 'bst' exists
if 'bst' not in globals():
    print("❌ Error: The LightGBM model variable 'bst' is missing.")
    print("   -> Please SCROLL UP and RUN the cell where you train the model (lgbm.train).")
else:
    print("✅ Model 'bst' found in memory.")

# Check if the preprocessor 'ct' exists
if 'ct' not in globals():
    print("❌ Error: The preprocessor variable 'ct' is missing.")
    print("   -> Please SCROLL UP and RUN the cell where you define 'ct = ColumnTransformer(...)'.")
else:
    print("✅ Preprocessor 'ct' found in memory.")

# Check threshold
if 'threshold' not in globals():
    print("⚠️  Threshold variable missing. Defaulting to 34.00.")
    threshold = 34.00
else:
    print(f"✅ Threshold found: {threshold}")


print("\n--- 2. Saving Files ---")

# Only proceed if critical variables exist
if 'bst' in globals() and 'ct' in globals():
    try:
        # Save Model
        joblib.dump(bst, 'lgbm_cinema_model.pkl')
        print(f"💾 Saved: lgbm_cinema_model.pkl")

        # Save Preprocessor
        joblib.dump(ct, 'preprocessor.pkl')
        print(f"💾 Saved: preprocessor.pkl")

        # Save Threshold
        joblib.dump(threshold, 'threshold.pkl')
        print(f"💾 Saved: threshold.pkl")
        
        print("\n🎉 SUCCESS! You can now run 'streamlit run cinema_app.py'")
        
    except Exception as e:
        print(f"\n❌ Save Failed: {e}")
else:
    print("\n⛔ Cannot save files because variables are missing. See errors above.")