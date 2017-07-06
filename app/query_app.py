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
        critical_cities = date_lookup[key][1]
    except IndexError:
        critical_cities = []
    elevated_cities = date_lookup[key][0]
    for city in critical_cities:
        elevated_cities.remove(city)
    critical_locs = [city_lookup[city]["location"] for city in critical_cities]
    elevated_locs = [city_lookup[city]["location"] for city in elevated_cities]
    critical_cities = zip(critical_cities, critical_locs)
    elevated_cities = zip(elevated_cities, elevated_locs)
    return render_template(
        "query.html",
        query_date=key,
        critical_cities=critical_cities,
        elevated_cities=elevated_cities,
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

    app.run(host="0.0.0.0", port=5050, debug=True)
