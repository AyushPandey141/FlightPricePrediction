# Project:Flight Price Prediction Web Application
# Program By:Ayush Pandey
# Email Id:1805290@kiit.ac.in
# DATE:29-Oct-2021
# Python Version:3.7
# CAVEATS:None
# LICENSE:None


from csv import DictWriter
from pandas import json_normalize
from bson.json_util import dumps
import csv
import json
from os import cpu_count
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import pymongo

client = pymongo.MongoClient()
FlightPrediction = client['FlightPrediction']
Flight1 = FlightPrediction['Flight1']

# CSV TO JSON for the database


def csv_to_json(csvFilePath, jsonFilePath):
    jsonArray = []

    # read csv file
    with open(csvFilePath, encoding='utf-8') as csvf:
        # load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf)

        # convert each csv row into python dict
        for row in csvReader:
            # add this python dict to json array
            jsonArray.append(row)

    # convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)


csvFilePath = r'ModifiedFlightPrice.csv'
jsonFilePath = r'Flights_JSON.json'

csv_to_json(csvFilePath, jsonFilePath)

# Inserting data into the database
with open('Flights_JSON.json') as f:
    data = json.load(f)
Flight1.insert_many(data)


pipe = pickle.load(open('FlightPricePrediction.pkl', 'rb'))

df = pd.read_csv('ModifiedFlightPrice.csv')


app = Flask(__name__)

# Prediction requirements page


@app.route('/')
def main():
    airline = df['Airline'].unique()
    source = df['Source'].unique()
    destination = df['Destination'].unique()
    stops = df['Total_Stops'].sort_values().unique()
    return render_template("index.html", airline=airline, source=source, destination=destination, stops=stops)


# Prediction Result Page
@app.route('/predict', methods=["POST"])
def get():
    if(request.method == 'POST'):
        airline = request.form['airline']
        source = request.form['source']
        destination = request.form['destination']
        stops = int(request.form['stops'])
        departure = request.form['departure']
        arrival = request.form['arrival']
        Journey_Date = int(pd.to_datetime(
            departure, format="%Y-%m-%dT%H:%M").day)
        Journey_Month = int(pd.to_datetime(
            departure, format="%Y-%m-%dT%H:%M").month)
        Dep_Hour = int(pd.to_datetime(departure, format="%Y-%m-%dT%H:%M").hour)
        Dep_Min = int(pd.to_datetime(
            departure, format="%Y-%m-%dT%H:%M").minute)

        Arrival_Hour = int(pd.to_datetime(
            arrival, format="%Y-%m-%dT%H:%M").hour)
        Arrival_Min = int(pd.to_datetime(
            arrival, format="%Y-%m-%dT%H:%M").minute)

        Duration_Hour = Arrival_Hour-Dep_Hour
        if(Duration_Hour < 0):
            Duration_Hour = Duration_Hour+24

        Duration_Min = Arrival_Min-Dep_Min
        if(Duration_Min < 0):
            Duration_Min = 60+Duration_Min

        print(airline, source, destination, stops, Journey_Date, Journey_Month, Dep_Hour, Dep_Min, Arrival_Hour,
              Arrival_Min, Duration_Hour, Duration_Min)
        arr = [[airline, source, destination, stops, Journey_Date, Journey_Month, Dep_Hour, Dep_Min, Arrival_Hour,
                Arrival_Min, Duration_Hour, Duration_Min]]

        if(source == destination):
            ans = float(0)
            pred = [ans]
            print(pred)
            return render_template('prediction.html', data=pred)

        pred = pipe.predict(pd.DataFrame(
            arr, columns=['Airline', 'Source', 'Destination', 'Total_Stops',
                          'Journey_Date', 'Journey_Month', 'Dep_Hour', 'Dep_Min', 'Arrival_Hour',
                          'Arrival_Min', 'Duration_Hour', 'Duration_Min']))
        print(pred)
        return render_template('prediction.html', data=pred)


# Displaying the csv file in the form of table page
@app.route('/table')
def get1():
    Airline = df['Airline']
    Source = df['Source']
    Destination = df['Destination']
    Total_Stops = df['Total_Stops']
    Price = df['Price']
    Journey_Date = df['Journey_Date']
    Journey_Month = df['Journey_Month']
    Dep_Hour = df['Dep_Hour']
    Dep_Min = df['Dep_Min']
    Arrival_Hour = df['Arrival_Hour']
    Arrival_Min = df['Arrival_Min']
    Duration_Hour = df['Duration_Hour']
    Duration_Min = df['Duration_Min']
    z = []
    length = len(Airline)
    count = 0
    for i in range(length-1, 0, -1):
        q = []
        if(count == 40):
            break
        q.append(Airline[i])
        q.append(Source[i])
        q.append(Destination[i])
        q.append(Total_Stops[i])
        q.append(Price[i])
        q.append(Journey_Date[i])
        q.append(Journey_Month[i])
        q.append(Dep_Hour[i])
        q.append(Dep_Min[i])
        q.append(Arrival_Hour[i])
        q.append(Arrival_Min[i])
        q.append(Duration_Hour[i])
        q.append(Duration_Min[i])
        z.append(q)
        count += 1
    return(render_template("table.html", z=z, length=40))

# Page containing the Report


@app.route('/Home')
def index():
    return render_template('Home.html')


# Page about analysis and visualization
@app.route('/First')
def get2():
    return render_template('First.html')

# Page for report conclusion


@app.route('/Conclusion')
def get3():
    return render_template('Conclusion.html')

# Page for getting the data


@app.route('/Insert')
def get4():
    airline = df['Airline'].unique()
    source = df['Source'].unique()
    destination = df['Destination'].unique()
    stops = df['Total_Stops'].unique()
    return render_template("Update.html", airline=airline, source=source, destination=destination, stops=stops)

# Page for adding the data in the csv file and database


@app.route('/Update', methods=['POST'])
def get5():
    if(request.method == 'POST'):
        airline = request.form['airline']
        source = request.form['source']
        destination = request.form['destination']
        stops = int(request.form['stops'])
        price = int(request.form['price'])
        departure = request.form['departure']
        arrival = request.form['arrival']
        Journey_Date = int(pd.to_datetime(
            departure, format="%Y-%m-%dT%H:%M").day)
        Journey_Month = int(pd.to_datetime(
            departure, format="%Y-%m-%dT%H:%M").month)
        Dep_Hour = int(pd.to_datetime(departure, format="%Y-%m-%dT%H:%M").hour)
        Dep_Min = int(pd.to_datetime(
            departure, format="%Y-%m-%dT%H:%M").minute)

        Arrival_Hour = int(pd.to_datetime(
            arrival, format="%Y-%m-%dT%H:%M").hour)
        Arrival_Min = int(pd.to_datetime(
            arrival, format="%Y-%m-%dT%H:%M").minute)

        Duration_Hour = Arrival_Hour-Dep_Hour
        if(Duration_Hour < 0):
            Duration_Hour = Duration_Hour+24

        Duration_Min = Arrival_Min-Dep_Min
        if(Duration_Min < 0):
            Duration_Min = 60+Duration_Min
        q = {}
        q['Airline'] = airline
        q['Source'] = source
        q['Destination'] = destination
        q['Total_Stops'] = stops
        q['Price'] = price
        q['Journey_Date'] = Journey_Date
        q['Journey_Month'] = Journey_Month
        q['Dep_Hour'] = Dep_Hour
        q['Dep_Min'] = Dep_Min
        q['Arrival_Hour'] = Arrival_Hour
        q['Arrival_Min'] = Arrival_Min
        q['Duration_Hour'] = Duration_Hour
        q['Duration_Min'] = Duration_Min

        print(q)
        Flight1.insert(q)

        w = []
        w.append(airline)
        w.append(source)
        w.append(destination)
        w.append(stops)
        w.append(price)
        w.append(Journey_Date)
        w.append(Journey_Month)
        w.append(Dep_Hour)
        w.append(Dep_Min)
        w.append(Arrival_Hour)
        w.append(Arrival_Min)
        w.append(Duration_Hour)
        w.append(Duration_Min)

        z = []
        z.append(w)
        length = len(z)

        def append_dict_as_row(file_name, dict_of_elem, field_names):
            # Open file in append mode
            with open(file_name, 'a') as f_object:

                dictwriter_object = DictWriter(
                    f_object, fieldnames=field_names, extrasaction='ignore')

                # Pass the dictionary as an argument to the Writerow()
                dictwriter_object.writerow(dict_of_elem)
                f_object.close()

        field_names = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Price',
                       'Journey_Date', 'Journey_Month', 'Dep_Hour', 'Dep_Min', 'Arrival_Hour',
                       'Arrival_Min', 'Duration_Hour', 'Duration_Min']

        append_dict_as_row('ModifiedFlightPrice.csv', q, field_names)

        return(render_template("table.html", z=z, length=length))


if __name__ == "__main__":
    app.run(debug=True)
