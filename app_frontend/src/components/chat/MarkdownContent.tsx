import React from 'react';
import ReactMarkdown from 'react-markdown';
import './MarkdownContent.css';

interface MarkdownContentProps {
  content?: string;
  className?: string;
}

export const MarkdownContent: React.FC<MarkdownContentProps> = ({ content, className = '' }) => {
  if (!content) {
    return null;
  }

  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        children={content}
        components={{
          ul: ({ ...props }) => <ul className="my-2 list-disc pl-5" {...props} />,
          ol: ({ ...props }) => <ol className="my-2 list-decimal pl-5" {...props} />,
          li: ({ ...props }) => <li className="my-1" {...props} />,
          strong: ({ ...props }) => <strong {...props} />,
        }}
      />
    </div>
  );
};
