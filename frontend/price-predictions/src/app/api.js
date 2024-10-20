import axios from 'axios';
import process from 'process';

export const fetchPredictions = async () => {
  try {
    const response = await axios.post('http://localhost:5000' + '/predict_all');
    return {"prophet_predictions": response.data.prophet_predictions, "xgb_predictions": response.data.xgb_predictions,"lstm_predictions": response.data.lstm_predictions};
  } catch (error) {
    console.error('Error fetching predictions:', error);
    throw error;
  }
};
