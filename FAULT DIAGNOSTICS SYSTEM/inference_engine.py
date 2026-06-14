"""
=============================================================================
  SMART FAULT DIAGNOSTIC SYSTEM — PROTOTYPE
  Module 3: Real-Time Inference Engine
=============================================================================
  Prototype specs:
    Transformer  : 1 kVA, single-phase, 240V primary → 12V secondary
    Current      : ACS712-20A (range 0–20A)
    Voltage      : ZMPT101B (range 0–250V primary, 0–25V secondary)
    Temperature  : DS18B20 (two probes: winding + oil)
    Oil Level    : HC-SR04 ultrasonic
=============================================================================
"""

import os, json, joblib, datetime
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List

MODEL_DIR  = "./models"
REPORT_DIR = "./reports"
os.makedirs(REPORT_DIR, exist_ok=True)

FAULT_LABELS = {
    0: "Normal",
    1: "Overheating",
    2: "Low_Oil_Level",
    3: "Overcurrent_Short_Circuit",
    4: "Voltage_Fault",
}

FAULT_SEVERITY = {
    0: "INFO",
    1: "CRITICAL",
    2: "HIGH",
    3: "CRITICAL",
    4: "HIGH",
}

FAULT_ACTIONS = {
    0: [
        "Prototype operating normally.",
        "Continue logging data. Schedule next inspection.",
    ],
    1: [
        "CRITICAL: Switch off prototype transformer immediately.",
        "Allow transformer to cool for at least 30 minutes before restart.",
        "Check ventilation around the enclosure — ensure airflow is not blocked.",
        "Check load — reduce connected load below 60% of rated.",
        "Inspect winding insulation after cooling.",
    ],
    2: [
        "WARNING: Oil level is low — insulation and cooling at risk.",
        "Top up oil in prototype tank to the marked fill level.",
        "Inspect tank seals and container for visible leakage.",
        "Do not run transformer continuously until oil level is restored.",
    ],
    3: [
        "CRITICAL: Overcurrent or short circuit detected — disconnect load immediately.",
        "Check connected load for short circuits before restarting.",
        "Inspect secondary winding terminals for signs of arcing or burn marks.",
        "Verify fuse/circuit breaker has not blown.",
        "Do NOT restart until fault cause is identified and cleared.",
    ],
    4: [
        "WARNING: Input voltage outside safe operating range.",
        "Measure mains supply with a calibrated voltmeter.",
        "If overvoltage: disconnect transformer from mains — insulation may be at risk.",
        "If undervoltage: check supply source and connections.",
        "Consider using a voltage stabiliser on the input supply.",
    ],
}

# ── Physical alarm thresholds (prototype-scale, sensor-appropriate) ──────────
# Based on: ACS712-20A, ZMPT101B, DS18B20, HC-SR04
THRESHOLDS = {
    "temp_winding_C":  {"warn": 75,   "crit": 90,   "low": False},  # DS18B20
    "temp_oil_C":      {"warn": 62,   "crit": 75,   "low": False},  # DS18B20
    "oil_level_pct":   {"warn": 60,   "crit": 45,   "low": True},   # HC-SR04
    "i_load_A":        {"warn": 15.0, "crit": 18.0, "low": False},  # ACS712-20A (>75%, >90% of 20A)
    "v_primary_V":     {"warn": 253,  "crit": 264,  "low": False},  # ZMPT101B overvoltage (+10%, +15%)
    "v_secondary_V":   {"warn": 13.5, "crit": 14.5, "low": False},  # secondary overvoltage
    "v_ratio_dev_pct": {"warn": 5.0,  "crit": 10.0, "low": False},  # voltage ratio deviation
    "load_pct":        {"warn": 75.0, "crit": 90.0, "low": False},  # % of ACS712 max
}


@dataclass
class SensorReading:
    """One complete sensor reading from the prototype transformer."""
    timestamp:       str   = field(default_factory=lambda: datetime.datetime.now().isoformat())
    transformer_id:  str   = "PROTO-001"
    # Primary sensor readings
    v_primary_V:     float = 0.0   # ZMPT101B: mains input voltage (V)
    v_secondary_V:   float = 0.0   # ZMPT101B: output voltage (V)
    i_load_A:        float = 0.0   # ACS712-20A: load current (A)
    temp_oil_C:      float = 0.0   # DS18B20: oil temperature (°C)
    temp_winding_C:  float = 0.0   # DS18B20: winding temperature (°C)
    temp_ambient_C:  float = 0.0   # DS18B20: ambient room temperature (°C)
    oil_level_pct:   float = 0.0   # HC-SR04: oil level (%)


@dataclass
class DiagnosticReport:
    timestamp:            str
    transformer_id:       str
    fault_code:           int
    fault_name:           str
    severity:             str
    confidence_pct:       float
    class_probabilities:  dict
    threshold_violations: list
    recommended_actions:  list
    computed_features:    dict


class PrototypeDiagnosticEngine:
    """
    Loads the trained ML model and diagnoses prototype transformer
    sensor readings in real time.
    """

    def __init__(self,
                 model_path:  str = f"{MODEL_DIR}/best_model.pkl",
                 scaler_path: str = f"{MODEL_DIR}/scaler.pkl",
                 meta_path:   str = f"{MODEL_DIR}/metadata.pkl"):
        self.model        = joblib.load(model_path)
        self.scaler       = joblib.load(scaler_path)
        meta              = joblib.load(meta_path)
        self.feature_names = meta["feature_names"]
        self.class_names   = meta["class_names"]
        specs = meta.get("prototype_specs", {})
        print(f"[Engine] Model loaded: {meta['best_model_name']}")
        print(f"[Engine] Prototype: {specs.get('transformer','1 kVA')}")
        print(f"[Engine] Sensors: {specs.get('current_sensor','')} | "
              f"{specs.get('voltage_sensor','')}")

    @staticmethod
    def engineer_features(r: SensorReading) -> dict:
        """Compute derived features from raw sensor readings."""
        PRIMARY_V   = 230.0
        PROTO_RATED_V = 12.0
        ACS712_MAX  = 20.0

        turns_ratio       = PROTO_RATED_V / PRIMARY_V
        v_ratio_actual    = r.v_secondary_V / (r.v_primary_V + 1e-9)
        v_ratio_dev_pct   = abs(v_ratio_actual - turns_ratio) / turns_ratio * 100

        load_pct          = (r.i_load_A / ACS712_MAX) * 100
        delta_temp        = r.temp_winding_C - r.temp_oil_C
        apparent_power_VA = r.v_secondary_V * r.i_load_A

        return {
            "v_primary_V":       r.v_primary_V,
            "v_secondary_V":     r.v_secondary_V,
            "i_load_A":          r.i_load_A,
            "temp_oil_C":        r.temp_oil_C,
            "temp_winding_C":    r.temp_winding_C,
            "temp_ambient_C":    r.temp_ambient_C,
            "oil_level_pct":     r.oil_level_pct,
            "v_ratio_dev_pct":   round(v_ratio_dev_pct,  3),
            "load_pct":          round(load_pct,          2),
            "delta_temp_C":      round(delta_temp,        2),
            "apparent_power_VA": round(apparent_power_VA, 2),
        }

    @staticmethod
    def check_thresholds(features: dict) -> list:
        violations = []
        for param, limits in THRESHOLDS.items():
            val = features.get(param)
            if val is None: continue
            low = limits.get("low", False)
            if low:
                if val <= limits["crit"]:
                    violations.append({"param":param,"value":val,"level":"CRITICAL","threshold":limits["crit"]})
                elif val <= limits["warn"]:
                    violations.append({"param":param,"value":val,"level":"WARNING", "threshold":limits["warn"]})
            else:
                if val >= limits["crit"]:
                    violations.append({"param":param,"value":val,"level":"CRITICAL","threshold":limits["crit"]})
                elif val >= limits["warn"]:
                    violations.append({"param":param,"value":val,"level":"WARNING", "threshold":limits["warn"]})
        return violations

    def diagnose(self, reading: SensorReading) -> DiagnosticReport:
        features = self.engineer_features(reading)
        X_row    = np.array([[features[f] for f in self.feature_names]])
        X_sc     = self.scaler.transform(X_row)
        pred     = int(self.model.predict(X_sc)[0])
        proba    = self.model.predict_proba(X_sc)[0]
        conf     = float(proba[pred]) * 100
        proba_d  = {self.class_names[i]: round(float(p)*100,2) for i,p in enumerate(proba)}
        return DiagnosticReport(
            timestamp            = reading.timestamp,
            transformer_id       = reading.transformer_id,
            fault_code           = pred,
            fault_name           = FAULT_LABELS[pred],
            severity             = FAULT_SEVERITY[pred],
            confidence_pct       = round(conf, 2),
            class_probabilities  = proba_d,
            threshold_violations = self.check_thresholds(features),
            recommended_actions  = FAULT_ACTIONS[pred],
            computed_features    = features,
        )

    @staticmethod
    def save_report(report: DiagnosticReport) -> str:
        ts   = report.timestamp.replace(":","-").replace(".","-")
        path = os.path.join(REPORT_DIR, f"report_{report.transformer_id}_{ts}.json")
        with open(path,"w") as f: json.dump(asdict(report), f, indent=2)
        return path

    @staticmethod
    def print_alert(report: DiagnosticReport):
        SEP = "─"*58
        print(f"\n{SEP}")
        print(f"  PROTOTYPE TRANSFORMER DIAGNOSTIC REPORT")
        print(SEP)
        print(f"  ID         : {report.transformer_id}")
        print(f"  Time       : {report.timestamp}")
        print(f"  Fault      : [{report.fault_code}] {report.fault_name}")
        print(f"  Severity   : {report.severity}")
        print(f"  Confidence : {report.confidence_pct:.1f}%")
        print(f"\n  Sensor Readings:")
        f = report.computed_features
        print(f"    V_primary  = {f.get('v_primary_V',0):.1f} V  (ZMPT101B, rated 230V)")
        print(f"    V_secondary= {f.get('v_secondary_V',0):.1f} V  (ZMPT101B, rated 12V)")
        print(f"    I_load     = {f.get('i_load_A',0):.2f} A  (ACS712-20A, max 20A)")
        print(f"    Temp_wind  = {f.get('temp_winding_C',0):.1f} °C  (DS18B20)")
        print(f"    Temp_oil   = {f.get('temp_oil_C',0):.1f} °C  (DS18B20)")
        print(f"    Oil level  = {f.get('oil_level_pct',0):.1f} %   (HC-SR04)")
        print(f"\n  Class Probabilities:")
        for cls, prob in report.class_probabilities.items():
            bar = "█" * int(prob/5)
            print(f"    {cls:<30} {prob:5.1f}%  {bar}")
        if report.threshold_violations:
            print(f"\n  ⚠ Threshold Violations:")
            for v in report.threshold_violations:
                print(f"    [{v['level']}] {v['param']} = {v['value']:.2f}  (limit: {v['threshold']})")
        print(f"\n  Recommended Actions:")
        for i,a in enumerate(report.recommended_actions, 1):
            print(f"    {i}. {a}")
        print(SEP)


def demo_inference():
    """Run 5 test scenarios covering all fault classes."""
    engine = PrototypeDiagnosticEngine()

    scenarios = [
        # 1. Normal operation
        SensorReading(transformer_id="PROTO-001-FUNAAB",
            v_primary_V=229.5, v_secondary_V=11.9, i_load_A=2.8,
            temp_oil_C=42.0, temp_winding_C=55.0, temp_ambient_C=31.0,
            oil_level_pct=85.0),

        # 2. Overheating (poor ventilation + overload)
        SensorReading(transformer_id="PROTO-002-FUNAAB",
            v_primary_V=231.0, v_secondary_V=11.7, i_load_A=6.5,
            temp_oil_C=74.0, temp_winding_C=96.0, temp_ambient_C=35.0,
            oil_level_pct=82.0),

        # 3. Low oil level
        SensorReading(transformer_id="PROTO-003-FUNAAB",
            v_primary_V=230.0, v_secondary_V=12.0, i_load_A=3.0,
            temp_oil_C=56.0, temp_winding_C=68.0, temp_ambient_C=32.0,
            oil_level_pct=38.0),

        # 4. Overcurrent / short circuit
        # ACS712 reads 17.5A (within 20A limit but 3.5x normal)
        SensorReading(transformer_id="PROTO-004-FUNAAB",
            v_primary_V=215.0, v_secondary_V=7.8, i_load_A=17.5,
            temp_oil_C=65.0, temp_winding_C=88.0, temp_ambient_C=32.0,
            oil_level_pct=84.0),

        # 5. Voltage fault (overvoltage from unstable mains)
        SensorReading(transformer_id="PROTO-005-FUNAAB",
            v_primary_V=261.0, v_secondary_V=14.1, i_load_A=3.2,
            temp_oil_C=46.0, temp_winding_C=60.0, temp_ambient_C=32.0,
            oil_level_pct=85.0),
    ]

    reports = []
    for s in scenarios:
        r = engine.diagnose(s)
        engine.print_alert(r)
        path = engine.save_report(r)
        print(f"  [Saved] {path}")
        reports.append(r)
    return reports


if __name__ == "__main__":
    demo_inference()
