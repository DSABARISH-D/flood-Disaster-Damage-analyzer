interface ResultCardProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

const ResultCard = ({ title, children, className = '' }: ResultCardProps) => {
  return (
    <div className={`bg-white rounded-lg border border-border shadow-sm overflow-hidden ${className}`}>
      <div className="bg-lightgray px-4 py-3 border-b border-border">
        <h3 className="font-semibold text-navy">{title}</h3>
      </div>
      <div className="p-4">
        {children}
      </div>
    </div>
  );
};

export default ResultCard;
