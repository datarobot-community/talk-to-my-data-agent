import React from 'react';

interface MessageContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  testId?: string;
}

export const MessageContainer = React.memo(
  React.forwardRef<HTMLDivElement, MessageContainerProps>(({ children, testId, ...props }, ref) => {
    return (
      <div
        className="mr-2 mb-8 flex min-w-0 flex-col items-start justify-start gap-3 rounded bg-card p-3"
        data-testid={testId}
        ref={ref}
        {...props}
      >
        {children}
      </div>
    );
  })
);
