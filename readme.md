Daily prediction 
  * Neural network  
    * Input layer (LTSM with 30 nodes)
    * Hidden layer 1 (LTSM with 30 nodes)
    * Hidden layer 2 (Dense with 30 nodes)
    * Output layer (Dense with 1 node)
  * A look back window of 30 days is analyzed and a prediction is made
  * Network trained for 20 epochs
  * Model predicted the price within +- 5% 77.07% of the times

Hourly prediction
  * Neural network 
    * Input layer (LTSM with 168 nodes)
    * Hidden layer 1 (LTSM with 168 nodes)
    * Hidden layer 2 (Dense with 168 nodes)
    * Output layer (Dense with 1 node)
  * A look back window of 168 hours is analyzed and a prediction is made
  * Network trained for 3 epochs
  * Model predicted the price within +- 5% 98.83% of the times

How to run 
  * From IDE ->
    * Open DailyPredictions.py / HourlyPredictions.py
    * Run
  * From CLI
    * Navigate to the folder where the data/python files are
    * Run desired file by "python <filename.py>"