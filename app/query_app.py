from flask import Flask, url_for, render_template, request, jsonify
import json
from collections import defaultdict
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def map():
    try:
        key = str(request.form["user_input"])
    except:
        key = ""

    try:
        date_lookup[key]
    except KeyError:
        return render_template(
            "map_root.html",
            query_date=key,
            predicted_cities=[],
            critical_cities=[],
            elevated_cities=[],
            predicted_locs=[],
            critical_locs=[],
            elevated_locs=[],
            num_pred=0,
            num_crit=0,
            num_elev=0,
            root_link=url_for("map")
        ), 200

    try:
        predicted_cities = set(date_lookup[key][2])
    except IndexError:
        predicted_cities = []

    try:
        critical_cities = set(date_lookup[key][1])
    except IndexError:
        critical_cities = []

    try:
        elevated_cities = set(date_lookup[key][0])
    except IndexError:
        elevated_cities = []

    for city in critical_cities:
        elevated_cities.remove(city)

    predicted_locs = [city_lookup[city]["location"] for city in predicted_cities]
    critical_locs = [city_lookup[city]["location"] for city in critical_cities]
    elevated_locs = [city_lookup[city]["location"] for city in elevated_cities]

    predicted_combos = sorted(zip(predicted_cities, predicted_locs))
    critical_combos = sorted(zip(critical_cities, critical_locs))
    elevated_combos = sorted(zip(elevated_cities, elevated_locs))

    return render_template(
        "map_root.html",
        query_date=key,
        predicted_cities=predicted_combos,
        critical_cities=critical_combos,
        elevated_cities=elevated_combos,
        predicted_locs=predicted_locs,
        critical_locs=critical_locs,
        elevated_locs=elevated_locs,
        num_pred=len(predicted_cities),
        num_crit=len(critical_cities),
        num_elev=len(elevated_cities),
        root_link=url_for("map")
    ), 200


if __name__ == '__main__':
    with open("date_lookup.json") as f:
        date_lookup = json.load(f)

    with open("city_lookup.json") as f:
        city_lookup = json.load(f)

    app.run(host="0.0.0.0", port=8080, debug=False)
