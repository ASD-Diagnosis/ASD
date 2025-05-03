
import React from 'react';
import { Users, Mail } from 'lucide-react';

const Team = () => {
  // Define team members array inside the component
  const teamMembers = [
    {
      name: "Abhishek Singh",
      role: "Machine Learning Developer",
      bio: "Abhishek was responsible for building and training the machine learning model that powers the core functionality of our application.",
      id: "2023CA07",
      imageUrl: "https://images.unsplash.com/photo-1568602471122-7832951cc4c5?auto=format&fit=crop&w=500&h=500&q=80",
      linkedin: "#"
    },
    {
      name: "Manish Yadav",
      role: "Frontend Developer",
      bio: "Manish led the frontend development and successfully integrated the trained model into the user interface, ensuring a seamless user experience.",
      id: "2023CA60",
      imageUrl: "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?auto=format&fit=crop&w=500&h=500&q=80",
      linkedin: "#"
    },
    {
      name: "Ashish Singh",
      role: "Documentation & Model Integration",
      bio: "Ashish took charge of all the documentation work and closely collaborated with Abhishek on model development and integration.",
      id: "2023CA25",
      imageUrl: "https://images.unsplash.com/photo-1557862921-37829c790f19?auto=format&fit=crop&w=500&h=500&q=80",
      linkedin: "#"
    }
  ];

  return (
    <div className="fade-in">
      {/* Hero Section */}
      <section className="bg-white dark:bg-gray-900 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h1 className="text-3xl md:text-4xl font-bold mb-4">Meet Our Team</h1>
            <p className="text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              We are a team of three passionate and dedicated developers from Motilal Nehru National Institute of Technology (MNNIT), Allahabad, 
              working collaboratively to build this project.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8 mt-8">
            {teamMembers.map((member, index) => (
              <div 
                key={index} 
                className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden transform transition-all duration-300 hover:scale-105 hover:shadow-xl"
              >
                <div className="relative">
                  <img 
                    src={member.imageUrl} 
                    alt={member.name} 
                    className="w-full h-64 object-cover object-center"
                  />
                </div>
                <div className="p-6 text-center">
                  <h3 className="text-xl font-semibold mb-2 text-gray-800 dark:text-white">
                    {member.name}
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                    {member.id}
                  </p>
                  <p className="text-autism-purple font-medium mb-4">
                    {member.role}
                  </p>
                  <p className="text-gray-600 dark:text-gray-400">
                    {member.bio}
                  </p>
                </div>
              </div>
            ))}
          </div>
          
          <div className="text-center mt-12">
            <p className="text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Our team combines expertise in machine learning, software development, and UI/UX design to create an accessible
              and effective tool for autism screening. We are committed to making a positive impact in healthcare technology.
            </p>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section className="bg-gray-50 dark:bg-gray-800 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">
              <span className="inline-block mr-2"><Mail className="h-7 w-7 inline-block" /></span>
              Get In Touch
            </h2>
            <p className="text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Have questions about our autism screening tool or want to collaborate with our team?
              We'd love to hear from you.
            </p>
          </div>
          
          <div className="max-w-3xl mx-auto">
            <div className="bg-white dark:bg-gray-900 rounded-xl shadow-md p-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-left">
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Name
                  </label>
                  <input
                    type="text"
                    id="name"
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-700 rounded-md focus:ring-autism-purple focus:border-autism-purple bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                    placeholder="Your name"
                  />
                </div>
                
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Email
                  </label>
                  <input
                    type="email"
                    id="email"
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-700 rounded-md focus:ring-autism-purple focus:border-autism-purple bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                    placeholder="your.email@example.com"
                  />
                </div>
                
                <div className="md:col-span-2">
                  <label htmlFor="message" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Message
                  </label>
                  <textarea
                    id="message"
                    rows={4}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-700 rounded-md focus:ring-autism-purple focus:border-autism-purple bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                    placeholder="Your message..."
                  ></textarea>
                </div>
                
                <div className="md:col-span-2">
                  <button 
                    type="submit"
                    className="w-full md:w-auto px-6 py-3 bg-autism-purple text-white font-medium rounded-md hover:bg-autism-purple/90 transition-colors duration-200"
                  >
                    Send Message
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Team;
