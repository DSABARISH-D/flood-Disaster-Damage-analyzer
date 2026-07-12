import { AlertTriangle, CheckCircle, Info } from 'lucide-react';

interface RecommendationCardProps {
  recommendations: { priority?: string, text: string }[];
}

const RecommendationCard = ({ recommendations }: RecommendationCardProps) => {
  if (!recommendations || recommendations.length === 0) {
    return (
      <div className="bg-green-50 text-green-800 p-4 rounded-lg border border-green-200 flex items-start gap-3">
        <CheckCircle className="h-5 w-5 text-green-600 mt-0.5" />
        <div>
          <h4 className="font-semibold mb-1">No Immediate Action Required</h4>
          <p className="text-sm">The AI did not detect any significant damage requiring immediate attention.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-lightgray p-4 rounded-lg border border-border">
      <h4 className="font-semibold text-navy mb-3 flex items-center gap-2">
        <Info className="h-5 w-5 text-orange-500" />
        Recommended Actions
      </h4>
      <ul className="space-y-3">
        {recommendations.map((rec, index) => (
          <li key={index} className="flex items-start gap-2 text-sm text-darkgray bg-white p-3 rounded border border-border">
            <AlertTriangle className="h-4 w-4 text-orange-500 mt-0.5 flex-shrink-0" />
            <span>{rec.text}</span>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default RecommendationCard;
