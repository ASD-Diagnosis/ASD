
import FileUploader from "../components/FileUploader";
import { Info } from "lucide-react";

const Upload = () => {
  return (
    <div className="fade-in max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-12">
        <h1 className="text-3xl md:text-4xl font-bold mb-4">Autism Screening Tool</h1>
        <p className="text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
          Upload your file for analysis by our neural network model. 
          We support various file formats including images, audio recordings, and behavioral data.
        </p>
      </div>

      {/* Information notice */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 mb-10 max-w-3xl mx-auto">
        <div className="flex">
          <div className="flex-shrink-0">
            <Info className="h-5 w-5 text-blue-500 dark:text-blue-400" />
          </div>
          <div className="ml-3 text-sm">
            <p className="text-blue-700 dark:text-blue-300">
              <span className="font-semibold">Privacy Notice:</span> All uploaded files are processed securely. 
              We do not store your data beyond the analysis session unless you explicitly opt-in.
            </p>
            <p className="mt-2 text-blue-700 dark:text-blue-300">
              <span className="font-semibold">Important:</span> This tool provides preliminary screening and is not a substitute for professional medical diagnosis.
            </p>
          </div>
        </div>
      </div>

      {/* File upload component */}
      <FileUploader />

      {/* Additional information */}
      <div className="mt-16 max-w-3xl mx-auto">
        <h2 className="text-2xl font-semibold mb-4">About Our Screening Method</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          Our neural network model has been trained on diverse datasets to identify patterns associated with autism spectrum characteristics.
          The algorithm analyzes various indicators and provides a preliminary assessment based on the uploaded data.
        </p>
        
        <div className="mt-8 grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-2">Supported File Types</h3>
            <ul className="space-y-2 text-gray-600 dark:text-gray-400">
              <li>• Images (JPEG, PNG)</li>
              <li>• Documents (PDF)</li>
              <li>• Data Files (CSV)</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-2">Accuracy & Limitations</h3>
            <ul className="space-y-2 text-gray-600 dark:text-gray-400">
              <li>• ~90% accuracy in preliminary screening</li>
              <li>• Best used as an initial assessment tool</li>
              <li>• Always consult healthcare professionals</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Upload;
