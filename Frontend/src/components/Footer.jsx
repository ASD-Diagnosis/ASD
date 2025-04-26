
import { Link } from "react-router-dom";

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="bg-gray-100 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="md:col-span-1">
            <div className="flex items-center space-x-2">
              <div className="h-8 w-8 bg-gradient-to-br from-autism-purple to-autism-blue rounded-full flex items-center justify-center">
                <span className="text-white font-bold">A</span>
              </div>
              <span className="text-xl font-bold text-gray-900 dark:text-white">AutismAid</span>
            </div>
            <p className="mt-4 text-gray-600 dark:text-gray-400 max-w-md">
              Providing accessible autism screening solutions through advanced machine learning technology.
            </p>
          </div>
          
          <div className="md:col-span-1">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white uppercase tracking-wider">Navigation</h3>
            <div className="mt-4 space-y-2">
              <Link to="/" className="block text-gray-600 dark:text-gray-400 hover:text-autism-purple dark:hover:text-autism-purple">
                Home
              </Link>
              <Link to="/upload" className="block text-gray-600 dark:text-gray-400 hover:text-autism-purple dark:hover:text-autism-purple">
                Screening
              </Link>
              <Link to="/resources" className="block text-gray-600 dark:text-gray-400 hover:text-autism-purple dark:hover:text-autism-purple">
                Resources
              </Link>
              <Link to="/about" className="block text-gray-600 dark:text-gray-400 hover:text-autism-purple dark:hover:text-autism-purple">
                About
              </Link>
            </div>
          </div>
          
          <div className="md:col-span-1">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white uppercase tracking-wider">Contact</h3>
            <div className="mt-4 space-y-2">
              <a href="mailto:contact@autismaid.com" className="block text-gray-600 dark:text-gray-400 hover:text-autism-purple dark:hover:text-autism-purple">
                contact@autismaid.com
              </a>
              <p className="text-gray-600 dark:text-gray-400">
                We're here to help with any questions about our screening tool.
              </p>
            </div>
          </div>
        </div>
        
        <div className="mt-8 pt-4 border-t border-gray-200 dark:border-gray-800">
          <p className="text-gray-500 dark:text-gray-400 text-sm text-center">
            © {currentYear} AutismAid. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
