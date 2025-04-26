import { Link } from "react-router-dom";
import { ArrowRight, Brain, FileUp, Users, BookOpen } from "lucide-react";

const Home = () => {
  return (
    <div className="fade-in">
      {/* Hero Section */}
      <section className="relative bg-white dark:bg-gray-900 overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 md:py-24">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
                <span className="gradient-text">Advanced Autism</span> Screening Technology
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-300 mb-8">
                AutismAid uses machine learning to provide accessible, accurate autism
                spectrum disorder screening - helping identify signs early when 
                intervention can be most effective.
              </p>
              <div className="flex flex-wrap gap-4">
                <Link
                  to="/upload"
                  className="button-primary"
                >
                  Start Screening
                  <ArrowRight className="ml-2 h-4 w-4 inline" />
                </Link>
                <Link
                  to="/about"
                  className="button-secondary"
                >
                  Learn More
                </Link>
              </div>
            </div>
            <div className="relative">
              <div className="glass-card p-6 md:ml-10">
                <img
                  src="https://images.unsplash.com/photo-1581091226033-d5c48150dbaa?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&h=400&q=80"
                  alt="Child development"
                  className="rounded-lg shadow-lg w-full h-auto"
                />
              </div>
              <div className="absolute -bottom-6 -left-6 bg-autism-purple/90 rounded-lg p-4 shadow-lg hidden md:block">
                <div className="text-white font-semibold">
                  <span className="block text-2xl">80%+</span>
                  <span className="text-sm opacity-90">Screening Accuracy</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Decorative elements */}
        <div className="hidden lg:block absolute top-0 right-0 -mt-16 -mr-16">
          <div className="w-64 h-64 bg-autism-purple/10 rounded-full"></div>
        </div>
        <div className="hidden lg:block absolute bottom-0 left-0 -mb-24 -ml-24">
          <div className="w-96 h-96 bg-autism-blue/10 rounded-full"></div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="bg-gray-50 dark:bg-gray-800 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">How AutismAid Works</h2>
            <p className="text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Our advanced neural network provides a simple, three-step process for preliminary autism screening
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 mt-12">
            <div className="bg-white dark:bg-gray-900 p-6 rounded-xl shadow-md hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 bg-autism-purple/20 rounded-full flex items-center justify-center mb-4">
                <FileUp className="h-6 w-6 text-autism-purple" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Upload Your File</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Upload audio recordings, images, or documents containing behavioral data for analysis
              </p>
            </div>

            <div className="bg-white dark:bg-gray-900 p-6 rounded-xl shadow-md hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 bg-autism-blue/20 rounded-full flex items-center justify-center mb-4">
                <Brain className="h-6 w-6 text-autism-blue" />
              </div>
              <h3 className="text-xl font-semibold mb-2">AI Analysis</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Our neural network analyzes patterns and identifies potential autism spectrum indicators
              </p>
            </div>

            <div className="bg-white dark:bg-gray-900 p-6 rounded-xl shadow-md hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 bg-autism-lightPurple/30 rounded-full flex items-center justify-center mb-4">
                <BookOpen className="h-6 w-6 text-autism-purple" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Get Results</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Receive a detailed report with confidence scores and recommendations for next steps
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* About Autism Section */}
      <section className="py-16 bg-white dark:bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div className="order-2 md:order-1">
              <img
                src="https://images.unsplash.com/photo-1604881991720-f91add269bed?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
                alt="Understanding autism"
                className="rounded-lg shadow-lg w-full h-auto"
              />
            </div>
            <div className="order-1 md:order-2">
              <h2 className="text-3xl font-bold mb-6">Understanding Autism</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition affecting communication, behavior, and social interaction. Each person with autism has a unique set of strengths and challenges.
              </p>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                Early detection is crucial for accessing appropriate support and intervention programs, which can significantly improve quality of life and developmental outcomes.
              </p>
              <div className="space-y-4">
                <div className="flex items-start">
                  <div className="flex-shrink-0 mt-1">
                    <div className="w-5 h-5 bg-autism-purple rounded-full"></div>
                  </div>
                  <div className="ml-3">
                    <h4 className="text-lg font-medium">Early intervention</h4>
                    <p className="text-gray-600 dark:text-gray-400">
                      Can lead to significant improvements in cognitive and social development
                    </p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="flex-shrink-0 mt-1">
                    <div className="w-5 h-5 bg-autism-blue rounded-full"></div>
                  </div>
                  <div className="ml-3">
                    <h4 className="text-lg font-medium">Personalized support</h4>
                    <p className="text-gray-600 dark:text-gray-400">
                      Early diagnosis enables tailored educational and therapeutic approaches
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-gray-50 dark:bg-gray-800 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold mb-6">Ready to Begin?</h2>
          <p className="text-lg text-gray-600 dark:text-gray-300 mb-8 max-w-3xl mx-auto">
            Start your autism screening process today - no registration required. 
            Our AI-powered tool provides instant preliminary results.
          </p>
          <Link
            to="/upload"
            className="button-primary inline-flex items-center"
          >
            Start Screening
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </div>
      </section>
    </div>
  );
};

export default Home;
