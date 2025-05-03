
// Configuration for backend API endpoints
export const API_ENDPOINTS = {
  // Primary autism screening prediction endpoint
  PREDICT: {
    // development: 'http://localhost:5000/predict', 
    development: 'https://asd-5m1p.onrender.com/api/', 

    staging: 'https://staging-api.example.com/predict',
    production: 'https://api.autismaid.com/predict'
  },
  
  // Additional endpoints can be added here
  USER_ASSESSMENT: {
    development: 'http://localhost:5000/user-assessment',
    staging: 'https://staging-api.example.com/user-assessment',
    production: 'https://api.autismaid.com/user-assessment'
  },
  
  // Example of additional service endpoints
  ADDITIONAL_SERVICE: {
    development: 'http://localhost:5000/additional-service',
    staging: 'https://staging-api.example.com/additional-service',
    production: 'https://api.autismaid.com/additional-service'
  }
};

// Utility to get the correct environment
export const getApiUrl = (service, environment = 'development') => {
  return API_ENDPOINTS[service][environment];
};
