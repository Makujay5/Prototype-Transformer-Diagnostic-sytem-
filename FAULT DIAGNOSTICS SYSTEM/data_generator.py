"""
=============================================================================
  SMART FAULT DIAGNOSTIC SYSTEM FOR DISTRIBUTION TRANSFORMER
  Module 1: Prototype Data Generator
=============================================================================
  Authors : Maku James Oluwatosin (20201749)
           & Eniyangbagbe Oluwaniyomi Enoch (20201740)
  University: Federal University of Agriculture, Abeokuta (FUNAAB)

  PROTOTYPE SPECIFICATIONS (Lab/Bench Scale):
  ─────────────────────────────────────────────
  Transformer : Single-phase, step-down, 1 kVA (lab prototype)
  Primary     : 240 V AC (mains)
  Secondary   : 12 V AC (low-voltage output for safety)
  Rated current (secondary) : I = P/V = 1000/12 ≈ 83 A  (or 5A for smaller)
  
  NOTE ON SENSOR LIMITS:
  • ACS712-20A  : measures ±20 A max — safe for prototype secondary
  • ZMPT101B    : measures 0–250 V AC — suitable for 240V primary side
  • DS18B20     : -55°C to +125°C — adequate for lab transformer temperatures
  • HC-SR04     : 2–400 cm — suitable for small oil tank monitoring

  WHY PROTOTYPE (NOT FULL-SCALE 200 kVA):
  • ACS712 (20A) cannot measure 278A rated current of a 200 kVA transformer
  • ZMPT101B measures up to 250V — not suitable for 11kV primary direct
  • Lab prototype demonstrates all fault types safely within sensor limits
  • Full-scale deployment would require CT (current transformer) and PT 
    (potential transformer) to step down signals before measurement

  Fault Classes (5 total for prototype):
    0 - Normal Operation
    1 - Overheating (temperature fault)
    2 - Low Oil Level
    3 - Short Circuit / Overcurrent
    4 - Over Voltage / Under Voltage
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

SEED = 42
np.random.seed(SEED)

# ── Prototype Transformer Specifications ─────────────────────────────────────
RATED_VA        = 1000          # 1 kVA prototype transformer
PRIMARY_V       = 230.0         # Primary voltage (V) — standard Nigerian mains
SECONDARY_V     = 12.0          # Secondary voltage (V) — safe lab output
RATED_CURRENT_A = RATED_VA / SECONDARY_V   # ≈ 83.3 A secondary rated current

# Sensor limits (must stay within these during operation)
ACS712_MAX_A    = 20.0          # ACS712-20A maximum measurable current
ZMPT101B_MAX_V  = 250.0         # ZMPT101B maximum measurable voltage

# For prototype: we measure on secondary side
# Secondary rated: 12V, 83A — but ACS712 is 20A
# So for lab demo we use a smaller transformer: 5A rated secondary
# This is realistic for a bench demonstration unit
PROTO_RATED_I   = 5.0           # Realistic rated current for ACS712 range
PROTO_RATED_V   = 12.0          # Secondary rated voltage (V)

# ── Fault class definitions ───────────────────────────────────────────────────
FAULT_LABELS = {
    0: "Normal",
    1: "Overheating",
    2: "Low_Oil_Level",
    3: "Overcurrent_Short_Circuit",
    4: "Voltage_Fault",
}
N_CLASSES = len(FAULT_LABELS)


def generate_prototype_data(n_samples_per_class: int = 1000,
                             save_csv: bool = True,
                             output_dir: str = ".") -> pd.DataFrame:
    """
    Generate synthetic sensor readings for a 1 kVA single-phase
    step-down prototype transformer (240V primary → 12V secondary).

    All values are within the physical sensor measurement limits:
      - ACS712-20A  : 0 to 20 A
      - ZMPT101B    : 0 to 250 V
      - DS18B20     : 20°C to 110°C (practical transformer range)
      - HC-SR04     : measures oil depth as percentage
    """
    records = []

    for label, fault_name in FAULT_LABELS.items():
        n = n_samples_per_class

        # ── Baseline normal readings ──────────────────────────────────────
        # Primary voltage (ZMPT101B on 240V mains side)
        v_primary   = np.random.normal(230.0, 3.0, n)   # Nigerian mains ≈ 230V

        # Secondary voltage (ZMPT101B on 12V output)
        v_secondary = np.random.normal(12.0,  0.3, n)

        # Secondary current (ACS712-20A) — normal load ≈ 60% of rated
        i_load      = np.random.normal(PROTO_RATED_I * 0.60, 0.15, n)  # ≈ 3A

        # Oil temperature (DS18B20 immersed in oil)
        temp_oil    = np.random.normal(42.0, 3.0, n)    # typical lab transformer temp

        # Winding temperature (DS18B20 on winding surface)
        temp_winding = np.random.normal(55.0, 4.0, n)

        # Ambient temperature (room temperature Nigeria)
        temp_ambient = np.random.normal(32.0, 2.5, n)

        # Oil level (HC-SR04 ultrasonic — percentage of tank full)
        oil_level   = np.random.normal(85.0, 3.0, n)

        # ── Fault perturbations ───────────────────────────────────────────

        if label == 0:  # NORMAL — no change
            pass

        elif label == 1:  # OVERHEATING
            # Temperature rises — from overload or poor ventilation
            temp_winding += np.random.uniform(20, 45, n)   # 75–100°C
            temp_oil     += np.random.uniform(12, 28, n)   # 54–70°C
            # Load is higher (causing heat)
            i_load       += np.random.uniform(1.5, 4.0, n) # 4.5–7A (within ACS712)
            oil_level    -= np.random.uniform(2.0, 6.0, n) # slight drop from expansion

        elif label == 2:  # LOW OIL LEVEL
            oil_level    -= np.random.uniform(30, 55, n)  # drops to 30–55%
            temp_winding += np.random.uniform(8,  20, n)  # rises from poor cooling
            temp_oil     += np.random.uniform(5,  15, n)
            # Current/voltage unaffected

        elif label == 3:  # OVERCURRENT / SHORT CIRCUIT
            # ACS712-20A: max 20A — short circuit drives to near maximum
            sc_factor     = np.random.uniform(2.5, 3.8, n)  # 2.5x to 3.8x rated
            i_load        = i_load * sc_factor               # 7.5A to 19A — stays <20A
            # Voltage sags under heavy fault current
            v_secondary  -= np.random.uniform(1.5, 5.0, n)  # sags to 7–10.5V
            v_primary    -= np.random.uniform(5.0, 20.0, n) # sags on primary too
            temp_winding += np.random.uniform(15, 40, n)    # rapid heating
            temp_oil     += np.random.uniform(8,  22, n)

        elif label == 4:  # VOLTAGE FAULT (over or under voltage)
            # Mix of overvoltage (60%) and undervoltage (40%) cases
            ov_mask = np.random.random(n) < 0.60
            # Overvoltage: primary rises above 253V (+10%)
            v_primary = np.where(ov_mask,
                                  v_primary + np.random.uniform(15, 40, n),   # 245–270V
                                  v_primary - np.random.uniform(25, 60, n))   # 170–205V
            # Secondary tracks primary change
            v_secondary = np.where(ov_mask,
                                    v_secondary + np.random.uniform(0.8, 2.2, n),  # 12.8–14.2V
                                    v_secondary - np.random.uniform(1.5, 4.0, n))  # 8–10.5V
            temp_winding += np.random.uniform(3, 12, n)   # mild temp rise from core losses

        # ── Derived features ──────────────────────────────────────────────
        # Voltage ratio (actual secondary/primary × turns ratio constant)
        # For a 230V:12V transformer, ratio ≈ 0.0522
        turns_ratio      = PROTO_RATED_V / PRIMARY_V              # 12/230
        v_ratio_actual   = v_secondary / (v_primary + 1e-9)
        v_ratio_deviation = np.abs(v_ratio_actual - turns_ratio) / turns_ratio * 100  # %

        # Load percentage (as fraction of ACS712 max for safety indicator)
        load_pct = (i_load / ACS712_MAX_A) * 100

        # Temperature differential
        delta_temp = temp_winding - temp_oil

        # Power (approximate, single phase)
        apparent_power = v_secondary * i_load   # VA

        # Clip to physical limits
        v_primary    = np.clip(v_primary,    100.0, ZMPT101B_MAX_V)
        v_secondary  = np.clip(v_secondary,    0.0,  25.0)
        i_load       = np.clip(i_load,          0.0, ACS712_MAX_A)
        temp_winding = np.clip(temp_winding,   20.0, 115.0)
        temp_oil     = np.clip(temp_oil,        20.0, 100.0)
        oil_level    = np.clip(oil_level,        0.0, 100.0)
        load_pct     = np.clip(load_pct,         0.0, 100.0)

        for idx in range(n):
            records.append({
                # ── Primary sensor readings ─────────────────────────────
                "v_primary_V":       round(float(v_primary[idx]),    2),  # ZMPT101B primary
                "v_secondary_V":     round(float(v_secondary[idx]),  2),  # ZMPT101B secondary
                "i_load_A":          round(float(i_load[idx]),       3),  # ACS712
                "temp_oil_C":        round(float(temp_oil[idx]),     2),  # DS18B20 oil
                "temp_winding_C":    round(float(temp_winding[idx]), 2),  # DS18B20 winding
                "temp_ambient_C":    round(float(temp_ambient[idx]), 2),  # DS18B20 ambient
                "oil_level_pct":     round(float(oil_level[idx]),    2),  # HC-SR04
                # ── Derived / engineered features ───────────────────────
                "v_ratio_dev_pct":   round(float(v_ratio_deviation[idx]), 3),
                "load_pct":          round(float(load_pct[idx]),     2),
                "delta_temp_C":      round(float(delta_temp[idx]),   2),
                "apparent_power_VA": round(float(apparent_power[idx]),2),
                # ── Target ──────────────────────────────────────────────
                "fault_label":       label,
                "fault_name":        fault_name,
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "prototype_fault_data.csv")
        df.to_csv(path, index=False)
        print(f"[DataGenerator] Saved → {path}")
        print(f"[DataGenerator] Shape: {df.shape}")
        print(f"\n[DataGenerator] Class distribution:")
        print(df['fault_name'].value_counts().to_string())
        print(f"\n[DataGenerator] Mean values per class:")
        print(df.groupby("fault_name")[[
            "v_primary_V","v_secondary_V","i_load_A",
            "temp_winding_C","oil_level_pct"
        ]].mean().round(2).to_string())

    return df


def load_and_preprocess(df: pd.DataFrame,
                         test_size:   float = 0.20,
                         val_size:    float = 0.10,
                         scaler_path: str   = "./models/scaler.pkl"):
    """Split data and apply StandardScaler."""
    feature_cols = [c for c in df.columns if c not in ("fault_label","fault_name")]
    X = df[feature_cols].values
    y = df["fault_label"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=SEED, stratify=y)
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - rel_val), random_state=SEED, stratify=y_temp)

    scaler   = StandardScaler()
    X_train  = scaler.fit_transform(X_train)
    X_val    = scaler.transform(X_val)
    X_test   = scaler.transform(X_test)

    os.makedirs(os.path.dirname(scaler_path) if os.path.dirname(scaler_path) else ".", exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[Preprocessing] Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler


if __name__ == "__main__":
    df = generate_prototype_data(n_samples_per_class=1000, save_csv=True, output_dir=".")
