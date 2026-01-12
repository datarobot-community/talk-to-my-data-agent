import React from 'react';

interface HeaderSectionProps {
  title: string;
  children: React.ReactNode;
}

export const HeaderSection: React.FC<HeaderSectionProps> = ({ title, children }) => {
  return (
    <>
      <div className="text-primary mn-label-large">{title}</div>
      <div className="text-primary body-secondary mt-1 mb-4">{children}</div>
    </>
  );
};
