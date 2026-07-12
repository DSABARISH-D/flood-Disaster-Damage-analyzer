import { useEffect, useState } from 'react';

interface ImagePreviewProps {
  file: File;
}

const ImagePreview = ({ file }: ImagePreviewProps) => {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  if (!previewUrl) return null;

  return (
    <div className="rounded-lg overflow-hidden border border-border shadow-sm bg-white">
      <img 
        src={previewUrl} 
        alt="Preview" 
        className="w-full h-auto max-h-[400px] object-contain"
      />
    </div>
  );
};

export default ImagePreview;
