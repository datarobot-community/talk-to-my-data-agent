import React from 'react';

interface HeaderSectionProps {
  title: string;
  children: React.ReactNode;
}

export const HeaderSection: React.FC<HeaderSectionProps> = ({ title, children }) => {
  return (
    <>
      <div className="mn-label-large text-primary">{title}</div>
      <div className="mt-1 mb-4 body-secondary text-primary">{children}</div>
    </>
  );
};
