"use client";
import React, { useState, useEffect } from 'react';
import Header from '../components/ui/header';
import PredictionChart from './predictionChart';
import { fetchPredictions } from './api';

const App = () => {
  const [prophetPredictions, setProphetPredictions] = useState([]);
  const [xgbPredictions, setXGBPredictions] = useState([]);
  const [lstmPredictions, setLSTMPredictions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPredictions()
      .then((data) => {
        console.log(data);
        setProphetPredictions(data.prophet_predictions);
        setXGBPredictions(data.xgb_predictions);  // Updated this line
        setLSTMPredictions(data.lstm_predictions);  // Updated this line
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching predictions:', error);
        setLoading(false);
      });
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 text-black">
      <Header />
      <div className="container mx-auto p-4">
        {loading ? (
          <div className="text-center mt-10">
            <p className="text-xl">Loading predictions...</p>
          </div>
        ) : (
          <PredictionChart
            prophetPredictions={prophetPredictions}
            xgbPredictions={xgbPredictions}
            lstmPredictions={lstmPredictions}
          />
        )}
      </div>
    </div>
  );
};

export default App;
