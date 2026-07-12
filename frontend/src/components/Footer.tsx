const Footer = () => {
  return (
    <footer className="bg-navy text-gray-400 py-6 mt-auto">
      <div className="container mx-auto px-4 text-center">
        <p className="text-sm">
          &copy; {new Date().getFullYear()} Flood Damage Assessment using Satellite & Drone Images.
        </p>
        <p className="text-sm mt-1">
          Developed for Hackathon Project
        </p>
      </div>
    </footer>
  );
};

export default Footer;
