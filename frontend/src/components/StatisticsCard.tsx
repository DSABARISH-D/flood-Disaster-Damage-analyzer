interface StatisticsCardProps {
  label: string;
  value: string | number;
  highlight?: boolean;
  valueColor?: string;
}

const StatisticsCard = ({ label, value, highlight = false, valueColor = 'text-darkgray' }: StatisticsCardProps) => {
  return (
    <div className={`p-4 rounded-lg border ${highlight ? 'border-orange-500 bg-orange-50' : 'border-border bg-lightgray'} flex flex-col items-center justify-center text-center h-full`}>
      <p className="text-sm font-medium text-gray-500 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${valueColor}`}>{value}</p>
    </div>
  );
};

export default StatisticsCard;
