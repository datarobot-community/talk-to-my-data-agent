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
                    ul: ({ ...props }) => <ul className="list-disc pl-5 my-2" {...props} />,
                    ol: ({ ...props }) => <ol className="list-decimal pl-5 my-2" {...props} />,
                    li: ({ ...props }) => <li className="my-1" {...props} />,
                }}
            />
        </div>
    );
};
