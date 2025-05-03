import React from 'react';
import { Brain, Code, ChevronRight, Check, Server } from 'lucide-react';

const About = () => {
  return (
    <div className="fade-in">
      {/* Previous sections remain unchanged... */}

      {/* FAQ Section */}
      <section className="bg-white dark:bg-gray-900 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Frequently Asked Questions</h2>
            <p className="text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Learn more about our autism screening tool and how it can help you.
            </p>
          </div>

          <div className="max-w-3xl mx-auto divide-y divide-gray-200 dark:divide-gray-700">
            {[{
                question: "How accurate is the screening tool?",
                answer: "Our neural network model achieves over 90% accuracy in preliminary screenings when compared to clinical diagnoses. However, it is designed to be a first step and not a replacement for professional medical assessment."
              },
              {
                question: "What types of files can I upload?",
                answer: (
                  <>
                    The system only accepts .1D files. If you want to convert your document into .1D format,{' '}
                    <a 
                      href='https://portal.cbrain.mcgill.ca/login' 
                      className='text-purple-600 dark:text-purple-400 hover:text-purple-800 dark:hover:text-purple-300 font-medium underline' 
                      target='_blank' 
                      rel='noopener noreferrer'
                    >
                      Click Here
                    </a>.
                  </>
                )
              },
              {
                question: "Is my data secure?",
                answer: "Yes, we take data security very seriously. All uploads are processed securely, and we do not store your data beyond the analysis session unless you explicitly opt in to save your results."
              },
              {
                question: "Do I need to create an account?",
                answer: "No, account creation is optional. You can use our screening tool without registration, though creating an account allows you to save your screening history for future reference."
              },
              {
                question: "What should I do after receiving my screening results?",
                answer: "If the screening indicates potential autism spectrum characteristics, we recommend consulting with a healthcare professional for a comprehensive evaluation. Our resources page provides information on finding appropriate specialists."
              }
            ].map((faq, index) => (
              <div key={index} className="py-6">
                <details className="group">
                  <summary className="flex justify-between items-center font-medium cursor-pointer list-none">
                    <span className="text-lg font-semibold text-gray-800 dark:text-white">{faq.question}</span>
                    <ChevronRight className="h-5 w-5 text-autism-purple transition-transform group-open:rotate-90" />
                  </summary>
                  <div className="text-gray-600 dark:text-gray-400 mt-4 pl-4">
                    {faq.answer}
                  </div>
                </details>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default About;