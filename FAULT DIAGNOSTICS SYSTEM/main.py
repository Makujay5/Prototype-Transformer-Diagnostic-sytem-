"""
=============================================================================
  SMART FAULT DIAGNOSTIC SYSTEM — PROTOTYPE
  main.py — Full Pipeline Runner
=============================================================================
  Run: python main.py
=============================================================================
"""

import time

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   SMART FAULT DIAGNOSTIC SYSTEM — PROTOTYPE                  ║
║   1 kVA | 230V Primary → 12V Secondary                       ║
║                                                              ║
║   Sensors:                                                   ║
║     ACS712-20A     (current,  0–20 A)                        ║
║     ZMPT101B       (voltage,  0–250 V)                       ║
║     DS18B20 × 2    (temperature, -55 to +125°C)              ║
║     HC-SR04        (oil level, ultrasonic)                   ║
║                                                              ║
║   Fault Classes:                                             ║
║     Normal | Overheating | Low Oil Level                     ║
║     Overcurrent/Short Circuit | Voltage Fault                ║
║                                                              ║
║   Authors: Maku James Oluwatosin    (20201749)               ║
║            Eniyangbagbe Oluwaniyomi Enoch (20201740)         ║
║   FUNAAB — Electrical & Electronics Engineering              ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    print("="*62)
    print("  PHASE 1 — DATASET GENERATION & MODEL TRAINING")
    print("="*62)
    t0 = time.time()
    from train_models import train_all_models
    trained, results, feature_names, scaler, X_test, y_test = train_all_models()
    print(f"\n  ✓ Training complete ({time.time()-t0:.1f}s)")

    print("\n"+"="*62)
    print("  PHASE 2 — REAL-TIME INFERENCE DEMO (5 prototype scenarios)")
    print("="*62)
    from inference_engine import demo_inference
    reports = demo_inference()
    print(f"\n  ✓ {len(reports)} diagnostic reports generated")

    print("\n"+"="*62)
    print("  RESULTS SUMMARY")
    print("="*62)
    print(f"\n  {'Model':<22}  {'Accuracy':>9}  {'F1-Macro':>9}")
    print(f"  {'-'*44}")
    for name, res in results.items():
        print(f"  {name:<22}  {res['accuracy']:>9.4f}  {res['f1_macro']:>9.4f}")

    best = max(results, key=lambda k: results[k]["f1_macro"])
    print(f"\n  ★ Best Model  : {best}")
    print(f"  ★ Accuracy    : {results[best]['accuracy']:.4f}")
    print(f"  ★ F1 (macro)  : {results[best]['f1_macro']:.4f}")
    print(f"\n  Models saved  : ./models/")
    print(f"  Plots saved   : ./plots/")
    print(f"  Reports saved : ./reports/")
    print(f"\n  To start the API server, run: python api_server.py\n")


if __name__ == "__main__":
    main()
