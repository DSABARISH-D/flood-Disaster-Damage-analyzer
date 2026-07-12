import {
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { AlertTriangle, CheckCircle, Map, Info, Building, Car, Navigation } from 'lucide-react';
import StatisticsCard from '../components/StatisticsCard';
import ResultCard from '../components/ResultCard';

const Dashboard = () => {
  // Mock Data for Charts
  const pieData = [
    { name: 'Buildings', value: 45, color: '#0F172A' }, // navy
    { name: 'Roads', value: 25, color: '#F97316' },     // orange
    { name: 'Vehicles', value: 30, color: '#94A3B8' },  // slate-400
  ];

  const lineData = [
    { day: 'Mon', severity: 20 },
    { day: 'Tue', severity: 40 },
    { day: 'Wed', severity: 65 },
    { day: 'Thu', severity: 85 },
    { day: 'Fri', severity: 90 },
    { day: 'Sat', severity: 70 },
    { day: 'Sun', severity: 45 },
  ];

  const barData = [
    { region: 'North', incidents: 12 },
    { region: 'South', incidents: 35 },
    { region: 'East', incidents: 8 },
    { region: 'West', incidents: 18 },
  ];

  const timelineData = [
    { time: '08:00 AM', event: 'Satellite pass over Sector 4', status: 'normal' },
    { time: '10:30 AM', event: 'Minor flooding detected in lowlands', status: 'warning' },
    { time: '02:15 PM', event: 'River banks breached. Severity High.', status: 'critical' },
    { time: '04:45 PM', event: 'Rescue teams dispatched to South Region', status: 'info' },
  ];

  return (
    <div className="max-w-7xl mx-auto py-8 px-4">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-navy">Global Dashboard</h1>
        <p className="text-gray-600">Real-time overview of aggregated flood damage assessments.</p>
      </div>

      {/* --- CARDS SECTION --- */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-red-50 border border-red-200 p-4 rounded-lg flex items-center gap-4">
          <AlertTriangle className="h-8 w-8 text-red-600" />
          <div>
            <p className="text-sm text-red-800 font-medium">Status</p>
            <p className="text-xl font-bold text-red-900">Flood Detected</p>
          </div>
        </div>
        
        <StatisticsCard label="Avg Confidence" value="94%" />
        <StatisticsCard label="Total Flood Area" value="1,240 acres" highlight={true} />
        
        <div className="bg-lightgray border border-border p-4 rounded-lg flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-500 font-medium">Risk Level</p>
            <p className="text-xl font-bold text-orange-500">CRITICAL</p>
          </div>
          <AlertTriangle className="h-8 w-8 text-orange-500 opacity-50" />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-white border border-border p-4 rounded-lg flex items-center gap-4 shadow-sm">
          <div className="bg-navy p-3 rounded-full text-white"><Building className="h-6 w-6" /></div>
          <div>
            <p className="text-sm text-gray-500 font-medium">Buildings Submerged</p>
            <p className="text-2xl font-bold text-navy">142</p>
          </div>
        </div>
        <div className="bg-white border border-border p-4 rounded-lg flex items-center gap-4 shadow-sm">
          <div className="bg-navy p-3 rounded-full text-white"><Navigation className="h-6 w-6" /></div>
          <div>
            <p className="text-sm text-gray-500 font-medium">Roads Compromised</p>
            <p className="text-2xl font-bold text-navy">38 km</p>
          </div>
        </div>
        <div className="bg-white border border-border p-4 rounded-lg flex items-center gap-4 shadow-sm">
          <div className="bg-navy p-3 rounded-full text-white"><Car className="h-6 w-6" /></div>
          <div>
            <p className="text-sm text-gray-500 font-medium">Vehicles Stranded</p>
            <p className="text-2xl font-bold text-navy">87</p>
          </div>
        </div>
      </div>

      {/* --- CHARTS SECTION --- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        
        {/* Pie Chart */}
        <ResultCard title="Infrastructure Damage Breakdown">
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  label
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </ResultCard>

        {/* Line Chart */}
        <ResultCard title="Flood Severity Trend (7 Days)">
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lineData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <Line type="monotone" dataKey="severity" stroke="#F97316" strokeWidth={3} dot={{ r: 4 }} />
                <CartesianGrid stroke="#ccc" strokeDasharray="5 5" vertical={false} />
                <XAxis dataKey="day" axisLine={false} tickLine={false} />
                <YAxis axisLine={false} tickLine={false} />
                <Tooltip />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </ResultCard>

        {/* Bar Chart */}
        <ResultCard title="Incidents by Region">
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="region" axisLine={false} tickLine={false} />
                <YAxis axisLine={false} tickLine={false} />
                <Tooltip cursor={{ fill: '#F3F4F6' }} />
                <Bar dataKey="incidents" fill="#0F172A" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ResultCard>

        {/* Timeline */}
        <ResultCard title="Live Event Timeline">
          <div className="h-64 overflow-y-auto pr-2 space-y-4">
            {timelineData.map((item, i) => (
              <div key={i} className="flex gap-4">
                <div className="flex flex-col items-center">
                  <div className={`h-3 w-3 rounded-full mt-1.5 flex-shrink-0 ${
                    item.status === 'critical' ? 'bg-red-500' :
                    item.status === 'warning' ? 'bg-orange-400' :
                    item.status === 'info' ? 'bg-blue-500' : 'bg-gray-300'
                  }`} />
                  {i !== timelineData.length - 1 && <div className="w-0.5 h-full bg-gray-200 mt-1" />}
                </div>
                <div className="pb-4">
                  <p className="text-xs font-bold text-gray-400 mb-1">{item.time}</p>
                  <p className="text-sm text-darkgray bg-gray-50 border border-gray-100 p-2 rounded">{item.event}</p>
                </div>
              </div>
            ))}
          </div>
        </ResultCard>
      </div>

      {/* --- RECOMMENDATION BAR --- */}
      <div className="bg-navy text-white p-6 rounded-lg shadow-md flex items-start gap-4">
        <Info className="h-6 w-6 text-orange-500 flex-shrink-0 mt-1" />
        <div>
          <h3 className="text-lg font-bold mb-2">Global Recommendation</h3>
          <p className="text-gray-300">
            Based on current data aggregates, the highest priority is the South Region. Immediate deployment of 
            amphibious rescue units and structural engineers is recommended. Ensure continuous monitoring of the 
            river banks in Sector 4.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
