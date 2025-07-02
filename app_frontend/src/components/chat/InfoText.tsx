import React from 'react';

interface InfoTextProps {
    children: React.ReactNode;
    className?: string;
}

export const InfoText: React.FC<InfoTextProps> = ({ children, className = 'mb-4' }) => {
    return (
        <div className={`text-primary/66 text-sm font-normal leading-tight ${className}`}>
            {children}
        </div>
    );
};
