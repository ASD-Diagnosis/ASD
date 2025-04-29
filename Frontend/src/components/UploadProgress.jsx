
import { useEffect, useState } from "react";

const UploadProgress = ({ progress }) => {
  const [percentage, setPercentage] = useState(0);

  useEffect(() => {
    setPercentage(Math.min(100, Math.max(0, progress)));
  }, [progress]);

  return (
    <div className="w-full">
      <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-1">
        <span>Upload Progress</span>
        <span>{percentage.toFixed(0)}%</span>
      </div>
      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
        <div 
          className="h-2 rounded-full bg-autism-purple transition-all duration-300 ease-out"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

export default UploadProgress;
