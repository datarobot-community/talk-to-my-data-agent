import React from 'react';

interface HighlightTextProps {
  text: string;
  searchText: string;
  className?: string;
}

export const HighlightText: React.FC<HighlightTextProps> = ({
  text,
  searchText,
  className = '',
}) => {
  if (!searchText?.trim() || !text) {
    return <span className={className}>{text}</span>;
  }

  const regex = new RegExp(`(${searchText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  const parts = text.split(regex);

  return (
    <span className={className}>
      {parts.map((part, index) => {
        const isMatch = regex.test(part);
        // Reset regex lastIndex to avoid issues with global flag
        regex.lastIndex = 0;

        return isMatch ? (
          <mark key={index} className="bg-yellow-200 dark:bg-yellow-800 px-0.5 rounded">
            {part}
          </mark>
        ) : (
          <span key={index}>{part}</span>
        );
      })}
    </span>
  );
};
