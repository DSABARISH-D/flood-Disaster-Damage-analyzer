import axios from 'axios';
import axiosRetry from 'axios-retry';

const API_BASE_URL = 'http://localhost:8000/api';

// Create an Axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds (AI models take time)
});

// Configure robust exponential backoff retry logic
// Why? If the user has a poor connection during a large image upload,
// this ensures the request is retried silently before throwing an error.
axiosRetry(apiClient, { 
    retries: 3, 
    retryDelay: axiosRetry.exponentialDelay,
    retryCondition: (error) => {
        // Retry on network errors or 5xx server errors
        return axiosRetry.isNetworkOrIdempotentRequestError(error) || error.response?.status >= 500;
    }
});

export interface PredictionResponse {
  flood: boolean;
  confidence: number;
  flood_percentage: number;
  severity: string;
  recommendation: string;
  objects: {
    [key: string]: number;
  };
  people_at_risk: number;
  processed_image?: string;
  pdf_report?: string;
  resnet_classification?: string;
}

export const apiService = {
  /**
   * Uploads an image to the backend for flood damage assessment.
   * Handles multipart/form-data.
   */
  async uploadImage(file: File): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('ImageUpload', file);
    formData.append('user_id', 'anonymous-user-id');

    const response = await apiClient.post<PredictionResponse>('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },
};
