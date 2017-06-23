# Stabilitas Thresholds

Capstone Project for Galvanize Data Science Immersive in conjunction with Stabilitas.

Business Question:
  How can Stabilitas alert customers to likely critical events as quickly as possible?
  
Business Answer:
    Deploy a layered model that detects initial signals from possible critical events, investigates these signals for their
  likelihood of correlating to more anomalous signals in the near future, and alerts Stabilitas when these likelihoods are
  high enough to warrant human action.
  
EDA:
    The EDA notebook covers my initial exploratory data analysis. Some of the cells are quite time consuming, but have been 
  commented out and replaced with either hard coding or imports from the data/ directory to make things simpler. This EDA
  was conducted on a two week sample of data centered around the December 2016 Market Attack in Berlin, Germany. Topics 
  covered include initial data visualizations, clustering by two methods, and some initial time series analysis.
