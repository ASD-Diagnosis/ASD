
import { useState } from "react";
import { Link } from "react-router-dom";
import ThemeToggle from "./ThemeToggle";
import { Menu, X } from "lucide-react";

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <header className="sticky top-0 z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-800 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <div className="h-8 w-8 bg-gradient-to-br from-autism-purple to-autism-blue rounded-full flex items-center justify-center">
                <span className="text-white font-bold">A</span>
              </div>
              <span className="text-xl font-bold text-gray-900 dark:text-white">AutismAid</span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-6">
            <Link
              to="/"
              className="text-gray-700 dark:text-gray-200 hover:text-autism-purple dark:hover:text-autism-purple font-medium transition-colors"
            >
              Home
            </Link>
            <Link
              to="/upload"
              className="text-gray-700 dark:text-gray-200 hover:text-autism-purple dark:hover:text-autism-purple font-medium transition-colors"
            >
              Screening
            </Link>
            <Link
              to="/resources"
              className="text-gray-700 dark:text-gray-200 hover:text-autism-purple dark:hover:text-autism-purple font-medium transition-colors"
            >
              Resources
            </Link>
            <Link
              to="/about"
              className="text-gray-700 dark:text-gray-200 hover:text-autism-purple dark:hover:text-autism-purple font-medium transition-colors"
            >
              About
            </Link>
          </nav>

          <div className="flex items-center space-x-4">
            <ThemeToggle />
            
            {/* Login button (optional as per requirements) */}
            <Link
              to="#"
              className="hidden md:inline-block text-sm px-4 py-2 border border-autism-purple text-autism-purple hover:bg-autism-purple hover:text-white rounded-md transition-colors duration-200"
            >
              Login
            </Link>

            {/* Mobile menu button */}
            <button
              onClick={toggleMenu}
              className="md:hidden text-gray-700 dark:text-gray-200"
              aria-label="Toggle menu"
            >
              {isMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="md:hidden py-4 space-y-4 slide-in">
            <Link
              to="/"
              className="block text-gray-700 dark:text-gray-200 hover:text-autism-purple dark:hover:text-autism-purple font-medium"
              onClick={toggleMenu}
            >
              Home
            </Link>
            <Link
              to="/upload"
              className="block text-gray-700 dark:text-gray-200 hover:text-autism-purple dark:hover:text-autism-purple font-medium"
              onClick={toggleMenu}
            >
              Screening
            </Link>
            <Link
              to="/resources"
              className="block text-gray-700 dark:text-gray-200 hover:text-autism-purple dark:hover:text-autism-purple font-medium"
              onClick={toggleMenu}
            >
              Resources
            </Link>
            <Link
              to="/about"
              className="block text-gray-700 dark:text-gray-200 hover:text-autism-purple dark:hover:text-autism-purple font-medium"
              onClick={toggleMenu}
            >
              About
            </Link>
            {/* Mobile login button */}
            <Link
              to="#"
              className="block w-full text-center text-sm px-4 py-2 border border-autism-purple text-autism-purple hover:bg-autism-purple hover:text-white rounded-md transition-colors duration-200"
            >
              Login
            </Link>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;
