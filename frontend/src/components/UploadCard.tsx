import React, { useCallback, useState } from 'react';
import { UploadCloud, Image as ImageIcon, X } from 'lucide-react';

interface UploadCardProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
}

const UploadCard = ({ onFileSelect, selectedFile, onClear }: UploadCardProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onFileSelect(file);
      } else {
        alert("Please upload an image file");
      }
    }
  }, [onFileSelect]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  if (selectedFile) {
    return (
      <div className="bg-lightgray p-6 rounded-lg border-2 border-border shadow-sm flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="bg-navy p-3 rounded-full text-white">
            <ImageIcon className="h-6 w-6" />
          </div>
          <div>
            <p className="font-medium text-darkgray truncate max-w-[200px] sm:max-w-xs">{selectedFile.name}</p>
            <p className="text-sm text-gray-500">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
          </div>
        </div>
        <button 
          onClick={onClear}
          className="text-gray-400 hover:text-red-500 transition-colors p-2"
          title="Remove file"
        >
          <X className="h-6 w-6" />
        </button>
      </div>
    );
  }

  return (
    <div
      className={`relative border-2 border-dashed rounded-lg p-10 text-center transition-colors ${
        isDragging ? 'border-orange-500 bg-orange-50' : 'border-gray-300 bg-lightgray hover:bg-gray-50'
      }`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
      />
      <div className="flex flex-col items-center justify-center pointer-events-none">
        <UploadCloud className={`h-12 w-12 mb-4 ${isDragging ? 'text-orange-500' : 'text-gray-400'}`} />
        <p className="text-lg font-medium text-darkgray">
          Drag & Drop your image here
        </p>
        <p className="text-sm text-gray-500 mt-2">
          or click to browse from your computer
        </p>
        <p className="text-xs text-gray-400 mt-4">
          Supports JPG, PNG, TIFF
        </p>
      </div>
    </div>
  );
};

export default UploadCard;
