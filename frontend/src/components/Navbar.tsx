import { Link } from 'react-router-dom';
import { Waves } from 'lucide-react';

const Navbar = () => {
  return (
    <header className="bg-navy text-white shadow-md">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <Waves className="h-6 w-6 text-orange-500" />
          <span className="font-bold text-xl tracking-tight hidden sm:block">Flood Damage Assessment</span>
          <span className="font-bold text-xl tracking-tight sm:hidden">FDA</span>
        </Link>
        <nav className="flex items-center gap-6">
          <Link to="/" className="text-gray-300 hover:text-white transition-colors">
            Home
          </Link>
          <Link to="/dashboard" className="text-gray-300 hover:text-white transition-colors">
            Dashboard
          </Link>
          <Link 
            to="/upload" 
            className="bg-orange-500 hover:bg-orange-600 text-white px-4 py-2 rounded-md font-medium transition-colors"
          >
            Upload
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Navbar;
