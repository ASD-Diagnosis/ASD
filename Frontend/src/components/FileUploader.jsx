import React, { useState, useRef } from "react";
import { Upload, File, X, CheckCircle, AlertCircle } from "lucide-react";
import { getApiUrl } from '../config/apiConfig';
import UploadProgress from "./UploadProgress";
import { useToast } from "@/hooks/use-toast";

const FileUploader = () => {
  const [file, setFile] = useState(null);
  const [age, setAge] = useState('');
  const [sex, setSex] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);
  const { toast } = useToast();

  const allowedFileTypes = ["image/jpeg", "image/png", "application/pdf", "text/csv", "application/octet-stream", "text/plain"];
  const maxFileSize = 10 * 1024 * 1024; // 10MB

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const validateFile = (file) => {
    const fileExtension = file.name.split('.').pop().toLowerCase();
    
    if (fileExtension !== "1d") {
      toast({
        title: "Unsupported File Type",
        description: "Only .1D files are supported for processing.",
        variant: "destructive",
      });
      return false;
    }

    if (!allowedFileTypes.includes(file.type) && fileExtension !== "1d") {
      setError("Invalid file type. Please upload jpeg, png, pdf, csv, or 1D files.");
      return false;
    }

    if (file.size > maxFileSize) {
      setError(`File is too large. Maximum size is ${maxFileSize / (1024 * 1024)}MB.`);
      return false;
    }

    setError("");
    return true;
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (validateFile(droppedFile)) {
        setFile(droppedFile);
        setResult(null); // Reset result when new file is uploaded
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      if (validateFile(selectedFile)) {
        setFile(selectedFile);
        setResult(null); // Reset result when new file is uploaded
      }
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    setError("");
    setResult(null);
    // Reset the file input
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      setError("Please select a file first.");
      return;
    }

    if (!age) {
      setError("Please enter the age.");
      return;
    }

    if (!sex) {
      setError("Please select the sex.");
      return;
    }

    setLoading(true);
    setError("");
    setUploadProgress(0);

    try {
      // Get API endpoint from configuration
      const predictionUrl = getApiUrl('PREDICT');
      
      const formData = new FormData();
      formData.append('file', file);
      // formData.append('age', age);
      // formData.append('sex', sex);

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          const newProgress = prev + (5 * Math.random());
          return newProgress >= 95 ? 95 : newProgress;
        });
      }, 300);

      const response = await fetch(predictionUrl, {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        throw new Error('Failed to process file');
      }

      const data = await response.json();
      const formattedResult = {
        prediction: data.prediction,
        asdProbability: data.probabilities?.ASD || 0,
        tdProbability: data.probabilities?.TD || 0
      };
      
      setResult(formattedResult);
      
      // Show toast notification based on result
      if (formattedResult.asdProbability > formattedResult.tdProbability) {
        toast({
          title: "⚠️ Assessment Complete",
          description: "Indicators of autism spectrum characteristics detected",
          variant: "destructive",
        });
      } else {
        toast({
          title: "✅ Assessment Complete",
          description: "No significant indicators detected",
        });
      }
    } catch (error) {
      console.error("Error processing file:", error);
      setError("Error processing the file. Please try again.");
      toast({
        title: "Error",
        description: "Failed to process your file. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      <div className="mb-6 grid grid-cols-2 gap-4">
        <div>
          <label htmlFor="age" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Age (years)
          </label>
          <input
            type="number"
            id="age"
            min="0"
            max="100"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-700 rounded-md focus:ring-autism-purple focus:border-autism-purple bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
            placeholder="Enter age"
          />
        </div>
        <div>
          <label htmlFor="sex" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Sex
          </label>
          <select
            id="sex"
            value={sex}
            onChange={(e) => setSex(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-700 rounded-md focus:ring-autism-purple focus:border-autism-purple bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
          >
            <option value="">Select sex</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
          </select>
        </div>
      </div>

      <div
        className={`border-2 border-dashed rounded-lg p-6 transition-colors
          ${isDragging ? 'bg-autism-purple/10 border-autism-purple' : 'bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-700'}
          ${file ? 'bg-gray-50 dark:bg-gray-800' : ''}
        `}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {!file ? (
          <div className="flex flex-col items-center justify-center py-6 text-center">
            <Upload className="h-12 w-12 text-gray-400 dark:text-gray-500 mb-4" />
            <p className="text-lg font-medium text-gray-700 dark:text-gray-300">
              Drag & drop your file here
            </p>
            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
              or click to browse
            </p>
            <p className="mt-4 text-xs text-gray-500 dark:text-gray-400">
              Supports JPEG, PNG, PDF, CSV, and 1D files up to 10MB
            </p>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="mt-4 px-4 py-2 bg-autism-purple hover:bg-autism-purple/90 text-white rounded-md transition-colors duration-200"
            >
              Select File
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.pdf,.csv,.1d"
              className="hidden"
              onChange={handleFileChange}
            />
          </div>
        ) : (
          <div className="flex flex-col space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="bg-gray-100 dark:bg-gray-700 p-2 rounded-md">
                  <File className="h-6 w-6 text-autism-purple" />
                </div>
                <div className="flex flex-col">
                  <p className="text-sm font-medium text-gray-800 dark:text-gray-200 truncate">
                    {file.name}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {(file.size / 1024).toFixed(2)} KB
                  </p>
                </div>
              </div>
              <button
                onClick={handleRemoveFile}
                className="p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400"
                aria-label="Remove file"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm rounded-md flex items-center">
          <AlertCircle className="h-4 w-4 mr-2 flex-shrink-0" />
          {error}
        </div>
      )}

      {file && !loading && !result && (
        <div className="mt-6 flex justify-center">
          <button
            onClick={handleSubmit}
            className="px-6 py-2 bg-autism-purple hover:bg-autism-purple/90 text-white rounded-md font-medium transition-colors duration-200 shadow-md hover:shadow-lg"
          >
            Process File
          </button>
        </div>
      )}

      {loading && (
        <div className="mt-6 flex flex-col items-center space-y-4">
          <UploadProgress progress={uploadProgress} />
          <p className="text-gray-600 dark:text-gray-400">Processing your file...</p>
        </div>
      )}

      {result && (
        <div className="mt-6">
          <div className={`p-6 rounded-md shadow-md ${
            result.asdProbability > result.tdProbability 
              ? "bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800/30" 
              : "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800/30"
          }`}>
            <div className="flex items-center mb-4">
              {result.asdProbability > result.tdProbability ? (
                <AlertCircle className="h-6 w-6 mr-2 text-orange-500 dark:text-orange-400" />
              ) : (
                <CheckCircle className="h-6 w-6 mr-2 text-green-500 dark:text-green-400" />
              )}
              <h3 className="text-lg font-semibold">Analysis Result</h3>
            </div>
            
            <div className="space-y-3">
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Assessment</p>
                <p className="text-lg font-medium">
                  {result.asdProbability > result.tdProbability 
                    ? "⚠️ You may have signs of Autism Spectrum Disorder (ASD)" 
                    : "✅ You are doing well. Your brain seems to be developing typically "
                  }
                </p>
              </div>
              
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Probabilities</p>
                <div className="flex items-center mt-2">
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5 mr-2">
                    <div 
                      className="h-2.5 rounded-full bg-orange-500" 
                      style={{ width: `${result.asdProbability * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium w-24">ASD: {(result.asdProbability * 100).toFixed(1)}%</span>
                </div>
                <div className="flex items-center mt-2">
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5 mr-2">
                    <div 
                      className="h-2.5 rounded-full bg-green-500" 
                      style={{ width: `${result.tdProbability * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium w-24">TD: {(result.tdProbability * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
            
            <div className="mt-6 text-sm">
              {result.asdProbability > result.tdProbability ? (
                <div className="space-y-2 text-gray-700 dark:text-gray-300">
                  <h4 className="font-medium">Suggestions:</h4>
                  <ul className="space-y-1 list-disc pl-5">
                    <li>Please talk to a doctor or specialist for proper advice.</li>
                    <li>Start learning and social activities early.</li>
                    <li>Spend time talking and playing with others every day.</li>
                  </ul>
                </div>
              ) : (
                <div className="space-y-2 text-gray-700 dark:text-gray-300">
                  <h4 className="font-medium">Healthy Tips:</h4>
                  <ul className="space-y-1 list-disc pl-5">
                    <li>Eat healthy food and drink water.</li>
                    <li>Play, walk, or do some fun exercise every day.</li>
                    <li>Talk with friends and family, and try fun brain games.</li>
                  </ul>
                </div>
              )}
              
              <p className="mt-4 italic text-gray-600 dark:text-gray-400">
                <strong>Note:</strong> This is an automated screening tool and not a medical diagnosis.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUploader;
