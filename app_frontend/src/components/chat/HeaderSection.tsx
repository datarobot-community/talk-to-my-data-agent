import React from 'react';

interface HeaderSectionProps {
    title: string;
    children: React.ReactNode;
}

export const HeaderSection: React.FC<HeaderSectionProps> = ({ title, children }) => {
    return (
        <>
            <div className="text-primary text-base font-semibold leading-tight">{title}</div>
            <div className="text-primary text-sm font-normal leading-tight mt-1 mb-4">
                {children}
            </div>
        </>
    );
};
