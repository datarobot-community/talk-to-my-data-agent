import React from 'react';
import { cn } from '@/lib/utils';

interface TruncatedTextProps {
    text?: string;
    maxLength?: number;
    tooltip?: boolean;
    className?: string;
    children?: string;
}

export const TruncatedText: React.FC<TruncatedTextProps> = ({
    text,
    className,
    maxLength = 18,
    tooltip = true,
    children,
}) => {
    text = text || children?.toString() || '';
    const isTruncated = text.length > maxLength;
    const truncatedText = isTruncated ? `${text.slice(0, maxLength)}...` : text;

    return (
        <span
            className={cn('truncate', className)}
            title={tooltip && isTruncated ? text : undefined}
        >
            {truncatedText}
        </span>
    );
};
