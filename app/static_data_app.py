from flask import Flask, url_for, render_template, request, jsonify
import json
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8">
            <title>Stabilitas Thresholds v1.0</title>
        </head>
        <body>
            <h1>Stabilitas Thresholds v1.0</h1>
            <p>
                Welcome! This site will allow you to enter a date and see
                cities with population 300,000 or more that had elevated
                and critical levels of risk reporting on that day. Current
                date range is 12 DEC 2016 to 26 DEC 2016.
            </p>
            <h3>Please enter a date in the format "YYYY-MM-DD"</h3>
            <form action="/query" method='POST' >
                <input type="text" name="user_input" />
                <input type="submit" />
            </form>
        </body>
    </html>
    """

@app.route("/query", methods=["GET", "POST"])
def query():
    key = str(request.form["user_input"])
    critical_cities = date_lookup[key][1]
    elevated_cities = date_lookup[key][0]
    return """
    Cities with critical risk levels on {0}:

    {1}

    Cities with elevated risk levels on {0}:

    {2}
    <a href={3}>Submit another query</a>!
    """.format(key, critical_cities, elevated_cities, url_for("root"))


if __name__ == '__main__':
    with open("date_lookup.json") as f:
        date_lookup = json.load(f)

    app.run(host="0.0.0.0", port=5050, debug=True)
