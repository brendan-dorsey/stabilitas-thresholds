from flask import Flask, flash url_for, render_template, request, jsonify
import json
from collections import defaultdict
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def map():
    try:
        key = str(request.form["user_input"])
    except:
        key = "2016-12-20"

    try:
        date_lookup[key]
    except KeyError:
        flash("Invalid query")
        return render_template(
            "dashboard_root.html",
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

    predicted_probas = []
    for city in predicted_cities:
        try:
            predicted_probas.append(round(city_lookup[city][key][0], 3))
        except:
            predicted_probas.append(0)

    critical_probas = []
    for city in critical_cities:
        try:
            critical_probas.append(round(city_lookup[city][key][0], 3))
        except:
            critical_probas.append(0)

    elevated_probas = []
    for city in elevated_cities:
        try:
            elevated_probas.append(round(city_lookup[city][key][0], 3))
        except:
            elevated_probas.append(0)

    predicted_titles = []
    for city in predicted_cities:
        try:
            predicted_titles.append(city_lookup[city][key][1])
        except:
            predicted_titles.append("None predicted critical")

    critical_titles = []
    for city in critical_cities:
        try:
            critical_titles.append(city_lookup[city][key][1])
        except:
            critical_titles.append("None predicted critical")

    elevated_titles = []
    for city in elevated_cities:
        try:
            elevated_titles.append(city_lookup[city][key][1])
        except:
            elevated_titles.append("None predicted critical")

    predicted_combos = sorted(
        zip(predicted_cities, predicted_probas, predicted_titles),
        key=lambda x: x[1],
        reverse=True
    )
    critical_combos = sorted(
        zip(critical_cities, critical_probas, critical_titles),
        key=lambda x: x[1],
        reverse=True
    )
    elevated_combos = sorted(
        zip(elevated_cities, elevated_probas, elevated_titles),
        key=lambda x: x[1],
        reverse=True
    )

    return render_template(
        "dashboard_root.html",
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

@app.route("/dist/<path:path>")
def static_proxy(path):
    return app.send_static_file("dist/"+path)

if __name__ == '__main__':
    with open("date_lookup.json") as f:
        date_lookup = json.load(f)

    with open("city_lookup.json") as f:
        city_lookup = json.load(f)

    # print date_lookup.keys()
    # print len(date_lookup.keys())

    app.run(host="0.0.0.0", port=8080, debug=False)
