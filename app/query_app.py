from flask import Flask, url_for, render_template, request, jsonify
import json
from collections import defaultdict
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("root.html"), 200

@app.route("/query", methods=["GET", "POST"])
def query():
    key = str(request.form["user_input"])

    try:
        predicted_cities = set(date_lookup[key][2])
    except IndexError:
        predicted_cities = []

    try:
        critical_cities = set(date_lookup[key][1])
    except IndexError:
        critical_cities = []

    elevated_cities = set(date_lookup[key][0])
    for city in critical_cities:
        elevated_cities.remove(city)
    predicted_locs = [city_lookup[city]["location"] for city in predicted_cities]
    critical_locs = [city_lookup[city]["location"] for city in critical_cities]
    elevated_locs = [city_lookup[city]["location"] for city in elevated_cities]

    predicted_cities = sorted(zip(predicted_cities, predicted_locs))
    critical_cities = sorted(zip(critical_cities, critical_locs))
    elevated_cities = sorted(zip(elevated_cities, elevated_locs))
    return render_template(
        "query.html",
        query_date=key,
        predicted_cities=predicted_cities,
        critical_cities=critical_cities,
        elevated_cities=elevated_cities,
        num_pred=len(predicted_cities),
        num_crit=len(critical_cities),
        num_elev=len(elevated_cities),
        root_link=url_for("root")
    ), 200

@app.route("/map_query", methods=["GET", "POST"])
def map():
    return render_template("basic_map.html"), 200


if __name__ == '__main__':
    with open("date_lookup.json") as f:
        date_lookup = json.load(f)

    with open("city_lookup.json") as f:
        city_lookup = json.load(f)

    print date_lookup.keys()

    app.run(host="0.0.0.0", port=8080, debug=True)
    
