from flask import Flask, url_for, render_template, request, jsonify
import cPickle as pickle
app = FLask(__name__)

@app.route("/", methods=["GET", "POST"]):
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
        </body>
    </html>
    """
