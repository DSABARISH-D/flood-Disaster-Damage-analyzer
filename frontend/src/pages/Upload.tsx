import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadCard from '../components/UploadCard';
import ImagePreview from '../components/ImagePreview';
import Loading from '../components/Loading';
import { apiService } from '../services/api';

const Upload = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const navigate = useNavigate();

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    try {
      const result = await apiService.uploadImage(selectedFile);
      
      // Store the image URL and result in session storage to pass to the result page
      // In a real app, this might go to a global store like Redux/Zustand or get fetched by ID
      const imageUrl = URL.createObjectURL(selectedFile);
      sessionStorage.setItem('analysisImage', imageUrl);
      sessionStorage.setItem('analysisResult', JSON.stringify(result));
      
      navigate('/result');
    } catch (error) {
      console.error('Failed to analyze image:', error);
      alert('Failed to analyze image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
  };

  return (
    <div className="max-w-3xl mx-auto py-10 px-4">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-navy mb-2">Upload Imagery</h1>
        <p className="text-gray-600">
          Select a satellite or drone image of the affected area to begin the damage assessment.
        </p>
      </div>

      {isAnalyzing ? (
        <div className="bg-white rounded-xl shadow-sm border border-border p-12 text-center">
          <Loading message="AI is processing the image. This may take a moment..." />
        </div>
      ) : (
        <div className="space-y-6">
          <UploadCard 
            onFileSelect={setSelectedFile} 
            selectedFile={selectedFile} 
            onClear={handleClear} 
          />

          {selectedFile && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <ImagePreview file={selectedFile} />
              
              <div className="flex justify-end">
                <button
                  onClick={handleAnalyze}
                  className="bg-orange-500 hover:bg-orange-600 text-white font-bold py-3 px-8 rounded-lg shadow transition-colors w-full sm:w-auto text-lg"
                >
                  Analyze Image
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Upload;
