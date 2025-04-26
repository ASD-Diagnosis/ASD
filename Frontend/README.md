
# AutismAid - Autism Screening Frontend Application

AutismAid is a frontend application designed to work with a neural network backend for autism screening. Users can upload files and receive analysis results without requiring registration.

## Features

- Light/dark theme support
- File upload interface for autism screening
- Resources page with educational content
- About page with project information
- Responsive design for all screen sizes

## Technologies Used

- React
- React Router
- Tailwind CSS
- Lucide React (for icons)

## Project Structure

```
src/
├── components/      # Reusable UI components
├── context/         # Context providers (theme)
├── pages/           # Application pages
└── assets/          # Static assets
```

## How to Run the Project

1. Clone the repository

```bash
git clone <repository-url>
cd autism-aid-frontend-app
```

2. Install dependencies

```bash
npm install
```

3. Start the development server

```bash
npm run dev
```

4. Open your browser and navigate to http://localhost:8080

## Integrating with Your Backend

To integrate this frontend with your neural network backend:

1. Update the API endpoint in `src/components/FileUploader.jsx`
2. Modify the file upload function to communicate with your backend
3. Adjust the response handling logic to match your backend's format

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally

## Notes for Backend Integration

This frontend is designed to be easily integrated with your autism screening neural network model. The file uploader component is ready to be connected to your backend API. Update the `handleSubmit` function in `src/components/FileUploader.jsx` to connect to your backend endpoint.
