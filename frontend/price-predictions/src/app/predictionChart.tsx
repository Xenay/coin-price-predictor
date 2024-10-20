// components/PredictionChart.jsx

import React from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title as ChartTitle,
    Tooltip,
    Legend,
    Filler,
  } from 'chart.js';
  ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    ChartTitle,
    Tooltip,
    Legend,
    Filler
  );
const PredictionChart = ({ prophetPredictions, xgbPredictions, lstmPredictions }) => {
  const dates = prophetPredictions.map((item) => item.date);

  const prophetPrices = prophetPredictions.map((item) => item.predicted_price);
  const xgbPrices = xgbPredictions.map((item) => item.predicted_price);
  const lstmPrices = lstmPredictions.map((item) => item.predicted_price);

  const chartData = {
    labels: dates,
    datasets: [
      {
        label: 'Prophet Prediction',
        data: prophetPrices,
        borderColor: 'rgba(75, 192, 192, 1)',
        fill: false,
        tension: 0.4,
      },
      {
        label: 'XGBoost Prediction',
        data: xgbPrices,
        borderColor: 'rgba(255, 99, 132, 1)',
        fill: false,
        tension: 0.4,
      },
      {
        label: 'LSTM Prediction',
        data: lstmPrices,
        borderColor: 'rgba(54, 162, 235, 1)',
        fill: false,
        tension: 0.4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: 'gray',
        },
      },
      title: {
        display: true,
        text: 'Next 60 Days Price Prediction',
        color: 'gray',
      },
      tooltip: {
        enabled: true,
        mode: 'nearest',
        intersect: false,
        callbacks: {
          label: function (context) {
            return `Price: $${context.parsed.y.toFixed(2)}`;
          },
        },
      },
    },
    scales: {
      x: {
        ticks: {
          color: 'gray',
          maxRotation: 45,
          minRotation: 45,
        },
      },
      y: {
        ticks: {
          color: 'gray',
        },
        title: {
          display: true,
          text: 'Price (USD)',
          color: 'gray',
        },
      },
    },
  };

  return (
    <div className="mt-8">
      <Line data={chartData} options={chartOptions} />
    </div>
  );
};

export default PredictionChart;
