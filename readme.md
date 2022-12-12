Daily prediction 
  * Neural network  
    * Input layer (LTSM with 30 nodes)
    * Hidden layer 1 (LTSM with 30 nodes)
    * Hidden layer 2 (Dense with 30 nodes)
    * Output layer (Dense with 1 node)
  * A look back window of 30 days is analyzed and a prediction is made
  * Network trained for 100 epochs
  * Model predicted the price within +- 5% 82.71% of the times

Hourly prediction
  * Neural network 
    * Input layer (LTSM with 168 nodes)
    * Hidden layer 1 (LTSM with 168 nodes)
    * Hidden layer 2 (Dense with 168 nodes)
    * Output layer (Dense with 1 node)
  * A look back window of 168 hours is analyzed and a prediction is made
  * Network trained for 10 epochs
  * Model predicted the price within +- 5% 99.94% of the times

How to run
  * From IDE ->
    * Open DailyPredictions.py / HourlyPredictions.py
    * Run
  * From CLI
    * Navigate to the folder where the data/python files are
    * Run desired file by "python <filename.py>"
  * NB! Make sure required libraries are installed:
    * Pandas
    * Numpy
    * Matplotlib
    * sklearn
    * Keras