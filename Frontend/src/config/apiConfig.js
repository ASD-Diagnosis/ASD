
// Configuration for backend API endpoints
const API_BASE_URL = {
  development: 'http://localhost:5000',
  staging: 'https://staging-api.example.com',
  production: 'https://asd-5m1p.onrender.com'
};

// Get the appropriate API URL based on environment
export const getApiUrl = (endpoint, environment = 'production') => {
  const env = environment || 'production';
  
  if (endpoint === 'PREDICT') {
    return `${API_BASE_URL[env]}/api/predict`;
  }
  
  // For other endpoints, use the standard format
  const endpoints = API_ENDPOINTS[endpoint];
  return endpoints ? endpoints[env] : null;
};

// Additional API endpoints
export const API_ENDPOINTS = {
  USER_ASSESSMENT: {
    development: `${API_BASE_URL.development}/user-assessment`,
    staging: `${API_BASE_URL.staging}/user-assessment`,
    production: `${API_BASE_URL.production}/user-assessment`
  },
  
  ADDITIONAL_SERVICE: {
    development: `${API_BASE_URL.development}/additional-service`,
    staging: `${API_BASE_URL.staging}/additional-service`,
    production: `${API_BASE_URL.production}/additional-service`
  }
};
