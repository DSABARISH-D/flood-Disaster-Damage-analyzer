import { Link } from 'react-router-dom';
import { ShieldAlert, Map, BarChart3 } from 'lucide-react';

const Home = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-140px)] py-12">
      <div className="text-center max-w-3xl px-4">
        <h1 className="text-4xl md:text-5xl font-extrabold text-navy tracking-tight mb-6">
          AI-Powered Flood Damage <span className="text-orange-500">Assessment</span>
        </h1>
        <p className="text-lg md:text-xl text-gray-600 mb-10 leading-relaxed">
          Upload satellite or drone imagery to instantly analyze flood extent, detect affected structures, 
          and generate actionable damage reports for rapid emergency response.
        </p>
        
        <Link 
          to="/upload" 
          className="inline-block bg-orange-500 hover:bg-orange-600 text-white font-bold text-lg px-8 py-4 rounded-lg shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-1"
        >
          Start Analysis
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-20 px-4 max-w-5xl w-full">
        <div className="bg-lightgray p-6 rounded-xl text-center flex flex-col items-center">
          <div className="bg-navy text-white p-3 rounded-full mb-4">
            <Map className="h-6 w-6" />
          </div>
          <h3 className="font-bold text-lg text-navy mb-2">Automated Mapping</h3>
          <p className="text-gray-600 text-sm">Instantly highlight flooded areas from raw imagery.</p>
        </div>
        <div className="bg-lightgray p-6 rounded-xl text-center flex flex-col items-center">
          <div className="bg-navy text-white p-3 rounded-full mb-4">
            <ShieldAlert className="h-6 w-6" />
          </div>
          <h3 className="font-bold text-lg text-navy mb-2">Damage Detection</h3>
          <p className="text-gray-600 text-sm">Identify submerged buildings, roads, and vehicles.</p>
        </div>
        <div className="bg-lightgray p-6 rounded-xl text-center flex flex-col items-center">
          <div className="bg-navy text-white p-3 rounded-full mb-4">
            <BarChart3 className="h-6 w-6" />
          </div>
          <h3 className="font-bold text-lg text-navy mb-2">Severity Scoring</h3>
          <p className="text-gray-600 text-sm">Actionable insights and severity classification.</p>
        </div>
      </div>
    </div>
  );
};

export default Home;
