"""
=============================================================================
  SMART FAULT DIAGNOSTIC SYSTEM — PROTOTYPE
  api_server.py — Flask REST API
=============================================================================
  Prototype: 1 kVA, 240V→12V single-phase transformer
  Sensors  : ACS712-20A, ZMPT101B, DS18B20 (×2), HC-SR04

  HOW TO RUN LOCALLY:
      python api_server.py

  HOW TO RUN ON RENDER:
      gunicorn api_server:app --bind 0.0.0.0:$PORT

  SEND TEST READING (Normal):
  curl -X POST http://localhost:5000/diagnose \
    -H "Content-Type: application/json" \
    -d "{\"transformer_id\":\"PROTO-001\",\"v_primary_V\":230,\"v_secondary_V\":12.0,\"i_load_A\":3.0,\"temp_oil_C\":42,\"temp_winding_C\":55,\"temp_ambient_C\":31,\"oil_level_pct\":85}"

  SEND TEST READING (Overcurrent):
  curl -X POST http://localhost:5000/diagnose \
    -H "Content-Type: application/json" \
    -d "{\"transformer_id\":\"PROTO-001\",\"v_primary_V\":215,\"v_secondary_V\":7.8,\"i_load_A\":17.5,\"temp_oil_C\":65,\"temp_winding_C\":88,\"temp_ambient_C\":32,\"oil_level_pct\":84}"
=============================================================================
"""

import os
import datetime
import requests as req_lib
from flask import Flask, request, jsonify
from inference_engine import PrototypeDiagnosticEngine, SensorReading

app = Flask(__name__)

print("="*58)
print("  Loading Prototype Diagnostic Engine ...")
engine = PrototypeDiagnosticEngine(
    model_path  = "./models/best_model.pkl",
    scaler_path = "./models/scaler.pkl",
    meta_path   = "./models/metadata.pkl",
)
print("  Ready.")
print("="*58)

# ── SMS Configuration (Termii — Nigerian SMS API) ─────────────────────────────
# Leave TERMII_API_KEY empty "" to disable SMS (system still works normally)
TERMII_API_KEY = ""               # paste your Termii key here
ENGINEER_PHONE = "+2348012345678" # replace with real phone number


def send_sms_alert(report):
    """Send SMS via Termii when CRITICAL or HIGH fault is detected."""
    if not TERMII_API_KEY:
        return
    if report.severity in ["CRITICAL", "HIGH"]:
        msg = (
            f"ALERT [{report.severity}]\n"
            f"Unit: {report.transformer_id}\n"
            f"Fault: {report.fault_name}\n"
            f"Confidence: {report.confidence_pct:.1f}%\n"
            f"Action: {report.recommended_actions[0]}"
        )
        try:
            r = req_lib.post("https://api.ng.termii.com/api/sms/send",
                json={"to":ENGINEER_PHONE,"from":"ProtoTXF","sms":msg,
                      "type":"plain","channel":"generic","api_key":TERMII_API_KEY},
                timeout=10)
            print(f"[SMS] {'Sent' if r.status_code==200 else f'Failed ({r.status_code})'}")
        except Exception as e:
            print(f"[SMS] Error: {e}")


@app.route("/", methods=["GET"])
def home():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<html><head><title>Prototype Transformer Diagnostic</title>
    <meta http-equiv="refresh" content="30">
    <style>body{{font-family:Arial;background:#0d1117;color:#58a6ff;padding:40px}}
    h1{{color:#fff}}.box{{background:#161b22;border:1px solid #30363d;
    border-radius:8px;padding:20px;margin:10px 0}}.ok{{color:#3fb950;font-weight:bold}}
    .url{{color:#f78166;font-family:monospace}}</style></head><body>
    <h1>Prototype Transformer Fault Diagnostic System</h1>
    <div class="box">
      <p>Status: <span class="ok">ONLINE</span></p>
      <p>Prototype: 1 kVA Single-Phase | 230V Primary → 12V Secondary</p>
      <p>Sensors: ACS712-20A | ZMPT101B | DS18B20 (×2) | HC-SR04</p>
      <p>Fault Classes: Normal | Overheating | Low Oil Level |
         Overcurrent/Short Circuit | Voltage Fault</p>
      <p>Last check: {now}</p>
    </div>
    <div class="box"><h3>API</h3>
      <p>POST to <span class="url">/diagnose</span> with JSON sensor data.</p>
      <p>GET <span class="url">/status</span> for health check.</p>
    </div>
    <div class="box">
      <p>Maku James Oluwatosin (20201749)</p>
      <p>Eniyangbagbe Oluwaniyomi Enoch (20201740)</p>
      <p>FUNAAB — Electrical & Electronics Engineering</p>
    </div></body></html>"""


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status":    "online",
        "prototype": "1kVA 230V→12V single-phase",
        "sensors":   ["ACS712-20A","ZMPT101B","DS18B20(x2)","HC-SR04"],
        "classes":   5,
        "timestamp": datetime.datetime.now().isoformat(),
    })


@app.route("/diagnose", methods=["POST"])
def diagnose():
    """
    Diagnose transformer fault from sensor readings.

    Required JSON fields:
      transformer_id  : string  (any label)
      v_primary_V     : float   (0–250V, ZMPT101B on mains side)
      v_secondary_V   : float   (0–25V,  ZMPT101B on output side)
      i_load_A        : float   (0–20A,  ACS712-20A)
      temp_oil_C      : float   (°C, DS18B20 in oil)
      temp_winding_C  : float   (°C, DS18B20 on winding)
      temp_ambient_C  : float   (°C, DS18B20 ambient)
      oil_level_pct   : float   (0–100%, HC-SR04)
    """
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    required = ["transformer_id","v_primary_V","v_secondary_V","i_load_A",
                "temp_oil_C","temp_winding_C","temp_ambient_C","oil_level_pct"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error":"Missing fields","missing":missing}), 400

    # Sensor range validation
    warnings_list = []
    if data.get("i_load_A", 0) > 20.0:
        warnings_list.append("i_load_A exceeds ACS712-20A maximum (20A) — reading unreliable")
    if data.get("v_primary_V", 0) > 250.0:
        warnings_list.append("v_primary_V exceeds ZMPT101B maximum (250V) — reading unreliable")

    try:
        reading = SensorReading(**data)
        report  = engine.diagnose(reading)
        engine.print_alert(report)
        send_sms_alert(report)

        response = {
            "transformer_id": report.transformer_id,
            "timestamp":      report.timestamp,
            "fault_code":     report.fault_code,
            "fault":          report.fault_name,
            "severity":       report.severity,
            "confidence_pct": report.confidence_pct,
            "probabilities":  report.class_probabilities,
            "violations":     report.threshold_violations,
            "actions":        report.recommended_actions,
        }
        if warnings_list:
            response["sensor_warnings"] = warnings_list
        return jsonify(response)

    except Exception as e:
        return jsonify({"error":"Diagnosis failed","detail":str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Server: http://localhost:{port}")
    print("  Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=False)
