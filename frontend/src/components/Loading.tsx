import { Loader2 } from 'lucide-react';

interface LoadingProps {
  message?: string;
}

const Loading = ({ message = 'Processing...' }: LoadingProps) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-darkgray">
      <Loader2 className="h-10 w-10 text-orange-500 animate-spin mb-4" />
      <p className="text-lg font-medium">{message}</p>
    </div>
  );
};

export default Loading;
