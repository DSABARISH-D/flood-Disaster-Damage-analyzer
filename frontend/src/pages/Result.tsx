import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, AlertTriangle, CheckCircle, Activity } from 'lucide-react';
import ResultCard from '../components/ResultCard';
import StatisticsCard from '../components/StatisticsCard';
import RecommendationCard from '../components/RecommendationCard';
import type { PredictionResponse } from '../services/api';

const Result = () => {
  const navigate = useNavigate();
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    const savedImage = sessionStorage.getItem('analysisImage');
    const savedResult = sessionStorage.getItem('analysisResult');

    if (!savedImage || !savedResult) {
      navigate('/upload');
      return;
    }

    setImageUrl(savedImage);
    setResult(JSON.parse(savedResult));
  }, [navigate]);

  const handleDownloadPDF = () => {
    // Mock PDF download
    alert("Downloading PDF Report...");
  };

  if (!result || !imageUrl) return null;

  // We now have a direct recommendation from the AI
  const mockRecommendations = [];
  if (result.recommendation) {
    mockRecommendations.push({ text: result.recommendation });
  }

  // Derive color based on severity
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'Severe': return 'text-red-600';
      case 'High': return 'text-orange-500';
      case 'Medium': return 'text-yellow-600';
      case 'Low': return 'text-green-600';
      default: return 'text-darkgray';
    }
  };

  return (
    <div className="max-w-5xl mx-auto py-8 px-4">
      <div className="flex items-center justify-between mb-8">
        <div>
          <Link to="/upload" className="flex items-center gap-2 text-gray-500 hover:text-navy transition-colors text-sm font-medium mb-2">
            <ArrowLeft className="h-4 w-4" /> Back to Upload
          </Link>
          <h1 className="text-3xl font-bold text-navy">Analysis Results</h1>
        </div>
        <button 
          onClick={() => {
            if (result.pdf_url) {
              window.open(result.pdf_url, '_blank');
            } else {
              alert('PDF is still generating. Please try again in a moment.');
            }
          }}
          className="flex items-center gap-2 bg-navy hover:bg-slate-800 text-white px-4 py-2 rounded-md font-medium transition-colors"
        >
          <Download className="h-4 w-4" />
          <span className="hidden sm:inline">Download PDF</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Images */}
        <div className="lg:col-span-2 space-y-6">
          <ResultCard title="Imagery">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium text-gray-500 mb-2">Original Image</p>
                <div className="rounded border border-border overflow-hidden bg-gray-50 aspect-video flex items-center justify-center relative">
                  <img src={imageUrl} alt="Original" className="w-full h-full object-cover" />
                </div>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-500 mb-2">Processed Image (Mask)</p>
                <div className="rounded border border-border overflow-hidden bg-gray-50 aspect-video flex items-center justify-center relative">
                  <img src={result.processed_image || imageUrl} alt="Processed" className="w-full h-full object-cover" />
                </div>
              </div>
            </div>
          </ResultCard>

          <ResultCard title="Detected Objects">
            <div className="flex flex-wrap gap-2">
              {Object.entries(result.objects).map(([key, count], i) => (
                <span key={i} className="bg-navy text-white text-sm px-3 py-1.5 rounded-full font-medium capitalize">
                  {key}: {count}
                </span>
              ))}
              {Object.values(result.objects).every(v => v === 0) && (
                <span className="text-gray-500 italic">No significant objects detected in the flood zone.</span>
              )}
            </div>
          </ResultCard>
          
          <RecommendationCard recommendations={mockRecommendations} />
        </div>

        {/* Right Column: Stats & Status */}
        <div className="space-y-6">
          <ResultCard title="Status Overview">
            <div className="flex flex-col gap-4">
              <div className={`p-4 rounded-lg flex items-center gap-4 ${result.flood ? 'bg-red-50 border border-red-200 text-red-700' : 'bg-green-50 border border-green-200 text-green-700'}`}>
                {result.flood ? <AlertTriangle className="h-8 w-8" /> : <CheckCircle className="h-8 w-8" />}
                <div>
                  <p className="text-sm font-medium opacity-80">Flood Status</p>
                  <p className="text-2xl font-bold">{result.flood ? 'Flood Detected' : 'No Flood'}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <StatisticsCard 
                  label="Severity" 
                  value={result.severity} 
                  highlight={result.severity === 'High' || result.severity === 'Severe'}
                  valueColor={getSeverityColor(result.severity)}
                />
                <StatisticsCard 
                  label="Confidence" 
                  value={`${result.confidence.toFixed(1)}%`} 
                  valueColor="text-navy"
                />
              </div>
              
              {result.resnet_classification && (
                <div className="mt-2 p-3 bg-gray-50 border border-gray-200 rounded-md">
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">ImageNet Feature Extraction</p>
                  <p className="text-sm font-medium text-navy mt-1 capitalize">{result.resnet_classification}</p>
                </div>
              )}
              
              <div className="mt-2">
                <div className="flex justify-between items-end mb-1">
                  <p className="text-sm font-medium text-gray-500">Flood Coverage Area</p>
                  <p className="text-lg font-bold text-navy">{result.flood_percentage}%</p>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div 
                    className="bg-orange-500 h-2.5 rounded-full" 
                    style={{ width: `${result.flood_percentage}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </ResultCard>

        </div>
      </div>
    </div>
  );
};

export default Result;
