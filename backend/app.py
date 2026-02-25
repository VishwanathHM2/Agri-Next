import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_from_directory
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from config import Config
from db import db, User, SoilRecord, CropHistory, MarketPrice, CropSchedule
from ml.recommender import predict_recommendation, load_model
from ml.disease_detector import predict_disease
from utils import load_lang, fertilizer_advice, generate_crop_calendar
import datetime
import google.generativeai as genai
from ml.rainfall import predict_rainfall

# Configure Gemini directly 
genai.configure(api_key="AIzaSyD2V81KAVFuv1KitjSrbbYWwjWBNDjGu38")

UPLOAD_EXTENSIONS = ['.jpg', '.png', '.jpeg']
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "templates"))
app.config.from_object(Config)

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.before_request
def create_tables():
    if not hasattr(app, 'tables_created'):
        db.create_all()
        app.tables_created = True

@app.route("/")
def index():
    lang = load_lang(request.args.get("lang", "en"))
    return render_template("index.html", lang=lang)

@app.route("/login", methods=["GET", "POST"])
def login():
    lang = load_lang(request.args.get("lang", "en"))
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid credentials")
    return render_template("login.html",lang=lang)

@app.route("/register", methods=["GET","POST"])
def register():
    lang = load_lang(request.args.get("lang", "en"))
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        full_name = request.form.get('full_name', '')
        if User.query.filter_by(username=username).first():
            flash("User exists")
            return redirect(url_for("register"))
        user = User(username=username, password_hash=generate_password_hash(password), full_name=full_name)
        db.session.add(user)
        db.session.commit()
        flash("Registered. Please login.")
        return redirect(url_for("login"))
    return render_template("login.html", lang=lang)

@app.route("/dashboard")
@login_required
def dashboard():
    lang = load_lang(current_user.preferred_language or "en")
    history = SoilRecord.query.filter_by(user_id=current_user.id).order_by(SoilRecord.recorded_at.desc()).limit(10).all()
    return render_template("dashboard.html", lang=lang, history=history)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# API: crop recommendation (POST JSON)
@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json() or {}
    # expected keys: N,P,K,pH,temp,humidity,rainfall,market_score
    try:
        inp = {
            "N": float(data.get("N", 0)),
            "P": float(data.get("P", 0)),
            "K": float(data.get("K", 0)),
            "pH": float(data.get("pH", 7.0)),
            "temp": float(data.get("temp", 25.0)),
            "humidity": float(data.get("humidity", 60.0)),
            "rainfall": float(data.get("rainfall", 0.0)),
            "market_score": float(data.get("market_score", 0.0))
        }
    except Exception as e:
        return jsonify({"error": "Invalid numeric inputs", "details": str(e)}), 400
    try:
        res = predict_recommendation(inp)
    except Exception as e:
        return jsonify({"error": "Model error", "details": str(e)}), 500
    return jsonify(res)

# API: fertilizer advice
@app.route("/api/fertilizer", methods=["POST"])
def api_fertilizer():
    data = request.get_json() or {}
    n = float(data.get("N", 0.0))
    p = float(data.get("P", 0.0))
    k = float(data.get("K", 0.0))
    ph = float(data.get("pH", 7.0))
    crop = data.get("crop")
    advice = fertilizer_advice(n, p, k, ph, crop)
    return jsonify({"advice": advice})

# API: disease detection (image upload)
@app.route("/api/detect_disease", methods=["POST"])
def api_detect_disease():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    img = request.files['image']
    filename = secure_filename(img.filename)
    if filename == '':
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(filename):
        return jsonify({"error": "Unsupported file type"}), 400
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img.save(save_path)
    try:
        preds = predict_disease(save_path, top_k=3)
    except Exception as e:
        return jsonify({"error": "Disease model error", "details": str(e)}), 500
    # map to pesticide suggestion (basic mapping — replace with agronomy data)
    pesticide_map = {
        "Apple_scab": "Apply Copper-based fungicide; follow label rates.",
        "Apple_black_rot": "Use Captan or appropriate fungicide; practice sanitation.",
        # ...
    }
    suggestions = []
    for p in preds:
        suggestions.append({"class": p["class"], "prob": p["prob"], "pesticide": pesticide_map.get(p["class"], "Use recommended pesticide for this disease; consult extension services.")})
    return jsonify({"predictions": suggestions})

# 📌 Page route: render the calendar page
@app.route("/calendar")
def calendar_page():
    return render_template("calendar.html", lang={"calendar_title": "Crop Calendar"})

# API: crop calendar
@app.route("/api/calendar", methods=["POST"])
def api_calendar():
    data = request.get_json() or {}
    crop = data.get("crop", "GenericCrop")
    sow_date = data.get("sowing_date")  # ISO format

    if sow_date:
        sow = datetime.date.fromisoformat(sow_date)
    else:
        sow = datetime.date.today()

    duration = int(data.get("duration_days", 120))
    harvest = sow + datetime.timedelta(days=duration)

    # Generate calendar schedule
    cal = generate_crop_calendar(sow, crop, duration)

    # Convert schedule to readable notes
    notes = "\n".join([f"{item['date']}: {item['task']}" for item in cal])

    # Save to database
    record = CropCalendar(
        user_id=current_user.id,
        crop=crop,
        sow_date=sow,
        harvest_date=harvest,
        notes=notes
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({
        "crop": crop,
        "schedule": cal
    })

# API: market trends - query top profitable crops by latest price
@app.route("/api/market_trends", methods=["GET"])
def api_market_trends():
    # simple aggregation
    results = []
    rows = MarketPrice.query.order_by(MarketPrice.date.desc()).limit(100).all()
    # aggregate latest price per crop
    latest = {}
    for r in rows:
        if r.crop not in latest:
            latest[r.crop] = {"price": r.price, "date": r.date.isoformat(), "source": r.source}
    for crop, val in latest.items():
        results.append({"crop": crop, **val})
    results = sorted(results, key=lambda x: -x["price"])[:20]
    return jsonify({"trends": results})

# Frontend pages: recommend and disease forms
@app.route("/recommend", methods=["GET","POST"])
@login_required
def recommend_page():
    lang = load_lang(current_user.preferred_language or "en")
    if request.method == "POST":
        n = float(request.form.get("N", 0))
        p = float(request.form.get("P", 0))
        k = float(request.form.get("K", 0))
        ph = float(request.form.get("pH", 7.0))
        temp = float(request.form.get("temp", 25.0))
        humidity = float(request.form.get("humidity", 60.0))
        rainfall = float(request.form.get("rainfall", 0.0))
        market_score = float(request.form.get("market_score", 0.0))
        payload = {"N": n, "P": p, "K": k, "pH": ph, "temp": temp, "humidity": humidity, "rainfall": rainfall, "market_score": market_score}
        rec = predict_recommendation(payload)
        # save soil record
        rec_model = SoilRecord(user_id=current_user.id, n=n, p=p, k=k, ph=ph, weather_temp=temp, weather_humidity=humidity, rainfall=rainfall, crop_predicted=str(rec.get("prediction")))
        db.session.add(rec_model)
        db.session.commit()
        return render_template("recommend.html", lang=lang, result=rec, input=payload)
    return render_template("recommend.html", lang=lang)

@app.route("/disease", methods=["GET","POST"])
@login_required
def disease_page():
    lang = load_lang(current_user.preferred_language or "en")
    if request.method == "POST":
        file = request.files.get("image")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            preds = predict_disease(path)
            return render_template("disease.html", lang=lang, preds=preds, img_url=url_for('uploaded_file', filename=filename))
        else:
            flash("Upload a valid image file")
    return render_template("disease.html", lang=lang)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.route("/delete_history/<int:id>", methods=["POST"])
@login_required
def delete_history(id):
    record = SoilRecord.query.get_or_404(id)
    # Ensure the user owns this record
    if record.user_id != current_user.id:
        flash("Unauthorized action", "danger")
        return redirect(url_for("dashboard"))
    db.session.delete(record)
    db.session.commit()
    flash("Soil record deleted successfully.", "success")
    return redirect(url_for("dashboard"))



@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "")
    
    # Call Gemini API with your key
    import requests
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=AIzaSyD2V81KAVFuv1KitjSrbbYWwjWBNDjGu38"
    headers = {"Content-Type": "application/json"}
    payload = { "contents": [ { "parts": [ { "text": f"Ans in brief,I am a farmer,{user_msg}" } ] } ] }

    try:
        res = requests.post(url, headers=headers, json=payload)
        res.raise_for_status()
        output = res.json()
        reply = output["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        reply = f"Error: {e}"

    return {"reply": reply}

#rainfall
@app.route("/api/rainfall", methods=["POST"])
def api_rainfall():
    data = request.get_json() or {}
    region = data.get("region")
    if not region:
        return jsonify({"error": "Region is required"}), 400
    result = predict_rainfall(region)
    return jsonify(result)
   


#..........

if __name__ == "__main__":
    app.run(debug=True)
