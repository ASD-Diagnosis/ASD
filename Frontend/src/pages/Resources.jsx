
import { Link } from "react-router-dom";
import { ExternalLink, BookOpen, FileText, Video, Users } from "lucide-react";

const Resources = () => {
  const resourceCategories = [
    {
      title: "Educational Resources",
      icon: BookOpen,
      resources: [
        {
          title: "Understanding Autism Spectrum Disorder",
          description: "A comprehensive guide to autism spectrum disorders, symptoms, and diagnostic criteria.",
          link: "https://www.cdc.gov/ncbddd/autism/index.html",
        },
        {
          title: "Early Signs of Autism",
          description: "Learn about the early developmental signs that may indicate autism spectrum disorder.",
          link: "https://www.autismspeaks.org/signs-autism",
        },
        {
          title: "Autism Research Updates",
          description: "The latest scientific research and advancements in understanding autism.",
          link: "https://www.spectrumnews.org/",
        },
      ]
    },
    {
      title: "Support Networks",
      icon: Users,
      resources: [
        {
          title: "Autism Support Groups",
          description: "Connect with other families affected by autism for shared experiences and advice.",
          link: "https://www.autismspeaks.org/family-services",
        },
        {
          title: "Parent Resources",
          description: "Tools and strategies for parents of children with autism spectrum disorder.",
          link: "https://www.autism.org/resources-for-families/",
        },
        {
          title: "Adult Autism Support",
          description: "Resources specifically designed for adults on the autism spectrum.",
          link: "https://autisticadvocacy.org/resources/",
        },
      ]
    },
    {
      title: "Intervention & Therapies",
      icon: FileText,
      resources: [
        {
          title: "Evidence-Based Interventions",
          description: "An overview of therapies and interventions with scientific support for autism treatment.",
          link: "https://www.nichd.nih.gov/health/topics/autism/conditioninfo/treatments",
        },
        {
          title: "Applied Behavior Analysis (ABA)",
          description: "Information about ABA therapy, one of the most widely used autism interventions.",
          link: "https://www.autismspeaks.org/applied-behavior-analysis",
        },
        {
          title: "Speech & Occupational Therapy Resources",
          description: "How speech and occupational therapy can benefit individuals with autism.",
          link: "https://www.asha.org/public/speech/disorders/autism/",
        },
      ]
    },
    {
      title: "Video Resources",
      icon: Video,
      resources: [
        {
          title: "Autism Explained",
          description: "A series of educational videos explaining different aspects of autism spectrum disorder.",
          link: "https://www.youtube.com/c/AutismSpeaks",
        },
        {
          title: "Personal Experiences",
          description: "Videos of individuals sharing their personal experiences living with autism.",
          link: "https://www.youtube.com/results?search_query=living+with+autism+personal+stories",
        },
        {
          title: "Expert Interviews",
          description: "Interviews with leading experts in the field of autism research and treatment.",
          link: "https://www.youtube.com/c/SpectrumNews",
        },
      ]
    },
  ];

  return (
    <div className="fade-in max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-12">
        <h1 className="text-3xl md:text-4xl font-bold mb-4">Autism Resources</h1>
        <p className="text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
          A curated collection of valuable resources to help understand, support, and navigate autism spectrum disorders.
        </p>
      </div>

      <div className="grid gap-8">
        {resourceCategories.map((category, index) => (
          <div key={index} className="bg-white dark:bg-gray-800 rounded-xl shadow-md overflow-hidden">
            <div className="p-6 border-b border-gray-100 dark:border-gray-700">
              <div className="flex items-center">
                <div className="w-10 h-10 bg-autism-purple/10 rounded-full flex items-center justify-center">
                  <category.icon className="h-5 w-5 text-autism-purple" />
                </div>
                <h2 className="ml-3 text-2xl font-semibold text-gray-800 dark:text-white">{category.title}</h2>
              </div>
            </div>
            
            <div className="divide-y divide-gray-100 dark:divide-gray-700">
              {category.resources.map((resource, resourceIndex) => (
                <div key={resourceIndex} className="p-6 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors duration-150">
                  <h3 className="text-lg font-medium text-gray-800 dark:text-white mb-2">
                    {resource.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    {resource.description}
                  </p>
                  <a 
                    href={resource.link} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="inline-flex items-center text-autism-purple hover:text-autism-blue transition-colors duration-200"
                  >
                    Visit Resource <ExternalLink className="ml-1 h-4 w-4" />
                  </a>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-16 bg-gradient-to-r from-autism-purple/10 to-autism-blue/10 rounded-xl p-8 text-center">
        <h2 className="text-2xl font-bold mb-4">Need More Information?</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-6 max-w-3xl mx-auto">
          Our resource list is continuously updated. If you're looking for specific information or would like to 
          suggest a valuable resource, please contact us.
        </p>
        <Link
          to="/about"
          className="button-primary inline-block"
        >
          Contact Us
        </Link>
      </div>
    </div>
  );
};

export default Resources;
